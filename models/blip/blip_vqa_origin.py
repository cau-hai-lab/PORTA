from models.blip.med import BertConfig, BertModel, BertLMHeadModel
from models.blip.blip_captioning import create_vit, init_tokenizer, load_checkpoint
from utils.prune_utils import inherit_encoder_momentum_masks
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.prune_utils import inherit_encoder_decoder_masks
from models.blip.vit import VisionTransformer, interpolate_pos_embed
from types import SimpleNamespace
class BLIPVQA(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/blip/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                   
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """         
        print("[Debug] blip_vqa.py : BLIPVQA 클래스 init() 함수 호출")      
        super().__init__()
        print(">>> BLIPVQA __init__ received image_size =", image_size)
        # Vision encoder 생성
        print("[Debug] blip_vqa.py : init()함수 -> Vision encoder 생성")
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1)
        self.tokenizer = init_tokenizer()
        # Text encoder 생성
        print("[Debug] blip_vqa.py : init()함수 -> Text encoder 생성")
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False) 
        # Text decoder 생성
        print("[Debug] blip_vqa.py : init()함수 -> Text decoder 생성")
        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)
        print(self.visual_encoder.pos_embed.shape)



    def forward(self, image, question, answer=None, k=None, weights=None, train=True, inference='rank', k_test=None, **kwargs):
        print("[Debug] blip_vqa.py : BLIPVQA 클래스 forward() 함수 실행")
        # 1) 혹시 PEFT/어댑터 경유로 top-level에 input_ids/attention_mask가 넘어올 경우 처리
        if question is None and ('input_ids' in kwargs or 'attention_mask' in kwargs):
            q_ids = kwargs.get('input_ids', None)
            q_att = kwargs.get('attention_mask', None)
            if q_ids is not None:
                if q_att is None:
                    q_att = torch.ones_like(q_ids)
                question = SimpleNamespace(input_ids=q_ids, attention_mask=q_att)

        # 2) question/answer가 dict(BatchEncoding)로 들어와도 동작
        if isinstance(question, dict):
            question = SimpleNamespace(**question)
        if isinstance(answer, dict):
            answer = SimpleNamespace(**answer)
        
        image_embeds = self.visual_encoder(image)   # Image 임베딩 -> Question 임베딩과 cross-attention에서 상호작용할 때 쓰임

        # 이미지 임베딩 벡터의 어텐션 마스크 : 중요한 임베딩에 1로 표시하여 나타냄
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device) 
        
        # question = self.tokenizer(question, padding='longest', truncation=True, max_length=35, 
        #                           return_tensors="pt").to(image.device)

        # Question 전처리 : Tokenize된 Question 텍스트의 첫 번째 시작 token을 enc_token_id로 교체 -> 질문의 시작을 특수한 Token으로 명시하기 위한 전략
        question.input_ids[:,0] = self.tokenizer.enc_token_id
        
        if train:  # Train 모드가 True일 때           
            '''
            n: number of answers for each question
            weights: weight for each answer
            '''                     
            # answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(image.device) 
            # Answer 전처리 : Tokenize된 Answer 텍스트의 첫 번째 시작 token을 bos_token_id로 교체 -> 정답 시퀀스의 시작점임을 명시하기 위한 전략
            answer.input_ids[:,0] = self.tokenizer.bos_token_id
            # 패딩 token의 위치는 학습 시에 loss에 반영하지 않기 위해 -100으로 설정(Pytorch의 CELoss에서는 -100은 손실 계산에서 제외함)
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)      

            # question 토큰을 Text encoder에 입력하여 인코딩 -> 인코딩된 question 토큰과 이미지 임베딩과 cross-attention을 통해 상호작용
            # 결과가 딕셔너리 형태로 반환됨(이미지 정보와 질문의 연관성을 내포한 임베딩 벡터)
            question_output = self.text_encoder(question.input_ids,  # question tokens
                                                attention_mask = question.attention_mask, # question mask 
                                                encoder_hidden_states = image_embeds, # image embeddings
                                                encoder_attention_mask = image_atts, # image mask         
                                                return_dict = True)    

            question_states = []                
            question_atts = []  
            # decoder에 답변 후보와 질문을 줄 때, 한 질문에 대해 n개 답변 후보가 있을 때, 한 번 계산된 질문 인코딩을 n번 복제해 “질문–답변” 쌍끼리 매핑

            for b, n in enumerate(k): # Batch 내 각 질문의 개수(b)와 각 질문에 맞는 답변의 개수(n) -> EarthVQA는 1질문-1정답임
                question_states += [question_output.last_hidden_state[b]]*n # b(1)개 질문에 대해 각 질문에 대한 인코딩 결과를 답변 개수 n(1개)만큼 복제해서 리스트에 추가 
                question_atts += [question.attention_mask[b]]*n # 각 질문에 대한 답변 개수 만큼 어텐션 마스크 리스트에 복제              
            question_states = torch.stack(question_states,0) # 리스트에 있는 모든 텐서를 하나의 텐서로 합친다
            question_atts = torch.stack(question_atts,0)   # 리스트에 있는 모든 텐서를 하나의 텐서로 합친다

            # Decoder에 정답 시퀀스와 해당 attention mask(실제 단어와 패딩 구분), 질문에 대한 인코딩 결과를 넣어 모델이 정답을 예측하도록 함
            # 주어진 답변 후보 시퀀스를 기반으로, 실제 answer와의 loss를 계산하는 역할 
            answer_output = self.text_decoder(answer.input_ids, 
                                              attention_mask = answer.attention_mask, # 실제 정답의 어텐션 마스크
                                              encoder_hidden_states = question_states, # encoder의 output(이미지 정보와 질문의 연관성을 내포한 임베딩 벡터)을 후보 개수만큼 담은 list
                                              encoder_attention_mask = question_atts, # encoder의 output의 어텐션 마스크                
                                              labels = answer_targets, # 실제 정답을 전달하여 예측한 정답과의 loss를 계산하도록 함
                                              return_dict = True,   
                                              reduction = 'none',
                                             ) 
                 
            # 후보 정답 개수별로 encoder를 통해 출력된 질문 임베딩을 decoder에 넣어 디코딩을 통해 예측 정답에 대한 로짓을 계산
            # 각 질문 임베딩별 출력된 예측 로짓과 후보 정답 값을 CE Loss를 통해 각각 loss를 구한다.

            #텍스트 디코더는 “질문+이미지” 인코딩 정보를 기반으로 답변 후보를 예측하고, Label을 이용하여 정확하게 예측되었는지 평가, loss계산


            loss = weights * answer_output.loss # 계산된 각 정답 후보별 loss에 가중치를 곱하여 더 중요한 후보의 loss를 크게 반영되도록 함
            loss = loss.sum()/image.size(0) # batch내 모든 loss 값을 합산하여 batch로 나누어 batch당 평균 손실을 구함

            return loss
            

        else: # Train 모드가 아닐 때(추론 모드)
            # question 토큰을 Text encoder에 입력하여 인코딩 -> 인코딩된 question 토큰과 이미지 임베딩과 cross-attention을 통해 상호작용
            question_output = self.text_encoder(question.input_ids, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                                    
                                                return_dict = True) 
            
            # if inference=='generate': # 디코더를 통해 답변을 생성
            #     num_beams = 3  # Beam search에서 3가지 후보를 고려
            #     #Beam search : 순차적인 데이터 생성할 때 사용하는 탐색 알고리즘 
            #     # 생성할 다음 후보 단어들에 대해 1가지만 선택하는 것이 아니라 3개의 후보에 대해 모두 고려하여 탐색하는 방법

            #     # encoder를 통해 각 batch 내 질문 시퀀스들에 대한 답변 후보 임베딩들을 num_beams만큼 반복하여 저장
            #     # 각 질문마다 여러 후보를 동시에 평가하기 위함
            #     question_states = question_output.last_hidden_state.repeat_interleave(num_beams, dim=0)
                
            #     # Batch 내 토큰에 대해 모두 1로 마스킹하여 모두 유효한 정보임을 표시함
            #     question_atts = torch.ones(question_states.size()[:-1],dtype=torch.long).to(question_states.device)
                
            #     # Decoder에 추가적으로 전달할 인자들을 딕셔너리 형태로 저장함
            #     model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask":question_atts}
                
            #     # Batch 크기만큼, 각 샘플마다 시작 토큰을 정의함
            #     bos_ids = torch.full((image.size(0),1),fill_value=self.tokenizer.bos_token_id,device=image.device)
                
            #     # Decoder generate 함수 호출
            #     outputs = self.text_decoder.generate(input_ids=bos_ids,
            #                                          max_length=10,
            #                                          min_length=1,
            #                                          num_beams=num_beams,
            #                                          eos_token_id=self.tokenizer.sep_token_id,
            #                                          pad_token_id=self.tokenizer.pad_token_id, 
            #                                          **model_kwargs)
                
            #     answers = []    
            #     for output in outputs: # 생성된 각 답변 토큰 시퀀스를 순회하며 특수 토큰을 제외하여 텍스트 문자열로 반환한다.
            #         answer = self.tokenizer.decode(output, skip_special_tokens=True)    
            #         answers.append(answer)
            #     return answers
            if inference == 'generate':
                num_beams = 3

                # 1) encoder hidden state / mask는 B 크기 그대로 사용
                #    (BLIP_Decoder.generate에서 image_embeds를 쓰는 부분과 동일한 역할)
                question_states = question_output.last_hidden_state           # [B, T_q, D]
                question_atts   = question.attention_mask                     # [B, T_q]

                model_kwargs = {
                    "encoder_hidden_states": question_states,
                    "encoder_attention_mask": question_atts,
                }

                # 2) decode 시작 토큰: encoder batch 크기와 동일하게 맞추기
                bos_ids = torch.full(
                    (question_states.size(0), 1),
                    fill_value=self.tokenizer.bos_token_id,
                    device=image.device,
                )

                # 3) generate 호출 (beam search는 HF 쪽에 맡김)
                outputs = self.text_decoder.generate(
                    input_ids=bos_ids,
                    max_length=10,
                    min_length=1,
                    num_beams=num_beams,
                    eos_token_id=self.tokenizer.sep_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **model_kwargs,
                )

                answers = []
                for output in outputs:
                    answer = self.tokenizer.decode(output, skip_special_tokens=True)
                    answers.append(answer)
                return answers

            elif inference=='rank': # 미리 주어진 후보 답변들 중에서 질문과 가장 잘 맞는 답변을 평가하고 선택하는 과정
                max_ids = self.rank_answer(question_output.last_hidden_state, question.attention_mask, 
                                           answer.input_ids, answer.attention_mask, k_test) # 상위 k_test개만큼의 답변에 대해 평가
                return max_ids # 가장 높은 점수를 받은 답변 정보를 반환한다.
 
                
                
    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0) # Batch 내 질문 개수
        start_ids = answer_ids[0,0].repeat(num_ques,1) # 전처리한 answer의 bos token을 가져와 각 question에 대해 bos token을 복제하여 텐서를 만듦
        # 후보 정답 집합에서 첫 토큰(BOS 토큰)을 각 질문마다 반복해서 준비(모든 질문에 대해 ‘시작’ 버튼을 누른다고 생각)
        
        # bos token을 입력으로 디코더를 실행하여, 각 질문에 대한 output 출력
        # 질문마다 bos token(시작 버튼)을 넣어 각 질문에 대한 예측 결과를 얻는다
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none')              
        logits = start_output.logits[:,0,:] # 각 질문에 대한 답변으로 후보 답변들의 첫 토큰이 나올 예측 점수를 추출
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1] # 각 후보 답변들의 두 번째 토큰(bos token다음에 오늘 실제 첫 단어)를 가져온다.

        # decoder가 각 질문에 대해 후보 답변들 중 어떤 단어가 가장 가능성이 높은지 확률로 계산한다.
        # 후보 답변들이 첫 토큰(bos_token)에서 얼마나 유망한지를 평가할 수 있게 된다.
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) #각 질문에 대해 상위 k개의 후보를 선택하고, 이 후보들의 인덱스를 기록
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        #각 질문마다 상위 k개로 선택된 토큰들의 전체 토큰 시퀀스를 찾아내어 하나의 텐서로 결합한다.
        for b, topk_id in enumerate(topk_ids): # 각 질문별로 상위 k개의 후보 답변의 인덱스를 담은 텐서
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        # 후보 답변 시퀀스에서, 패딩 토큰은 loss 계산에서 무시되도록 마스킹한다.
        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)


        # repeat encoder's output for top-k answers
        #각 질문에 대해 선택된 후보 답변과 질문의 encoding 결과를 일대일로 대응시킴
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        # 각 답변 후보 전체 시퀀스와 인코딩된 질문을 decoder에 넣어 해당 답변이 질문과 얼마나 잘 맞는지 평가한다.
        output = self.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, 
                                   reduction = 'none')   
        
        # 각 질문에 대한 후보 답변과 정답에 대해 계산된 CELoss로, 손실이 낮을 수록 후보 답변의 예측이 좋다
        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques,k)

        max_topk_ids = log_probs_sum.argmax(dim=1) 
        max_ids = topk_ids[max_topk_ids>=0,max_topk_ids]

        return max_ids # 각 질문에 대해 가장 높은 점수를 받은 후보 답변의 인덱스를 반환
    
    
    # def load_from_pruned_pretrained(self, pretraining_weights, mask, config, is_eval=False):
    #     print("[Debug] blip_vqa.py : load_from_pruned_pretrained() 함수 호출 -> pruning mask 적용")
    #     self.load_pretrained(pretraining_weights, config)

    #     print(f"Loading from mask at: {mask}")
    #     mask = torch.load(mask, map_location="cpu")
    #     mask = inherit_encoder_decoder_masks(mask)
    #     msg = self.load_state_dict(mask, strict=False) # pruning mask를 가중치에 적용함
    #     relevant_missing_keys = [k for k in msg.missing_keys if "bias" not in k and "layernorm" not in k.lower() and "pruning_mask" in k]
    #     if len(relevant_missing_keys) > 0:
    #         print(f"missing keys: {relevant_missing_keys}")
        
    #     keys_to_exclude = ["bias", "layernorm", "pruning_mask", "text_encoder_m", "visual_encoder_m"]
    #     relevant_unexpected_keys = [k for k in msg.unexpected_keys if not any([x in k.lower() for x in keys_to_exclude])]
    #     if len(relevant_unexpected_keys) > 0:
    #         print(f"unexpected keys: {relevant_unexpected_keys}")


    # def load_pretrained(self, weights_ckpt, config, is_eval=False):
    #     print("[Debug] blip_vqa.py : load_pretrained() 함수 호출 -> pre-trained된 가중치 정보 출력")
    #     print("Loaded params from: ", weights_ckpt)
    #     _, msg = load_checkpoint(self, weights_ckpt)
    #     relevant_missing_keys = [k for k in msg.missing_keys if "pruning_mask" not in k]
    #     if len(relevant_missing_keys) > 0:
    #         print(f"missing keys: {relevant_missing_keys}")

    #     # the checkpoint also contains the weights of the momentum encoders, which are not to be loaded 
    #     keys_to_exclude = ["visual_encoder_m", "text_encoder_m", "vision_proj_m", "text_proj_m", "text_decoder"]
    #     relevant_unexpected_keys = [k for k in msg.unexpected_keys if not any([x in k for x in keys_to_exclude])]
    #     if len(relevant_unexpected_keys) > 0:
    #         print(f"unexpected keys: {relevant_unexpected_keys}")
    # def load_pretrained(self, weights_ckpt, config, is_eval=False):
    #     print("<blip/blip_retrieval.py -> BLIPRetrieval.load_pretrained()>")
    #     print("Loaded params from:", weights_ckpt)

    #     # 기존 load_checkpoint(...) 대신 직접 로드 + 리매핑
    #     raw = torch.load(weights_ckpt, map_location="cpu")
    #     sd  = raw.get("model", raw)  # 포맷에 따라
    #     sd  = adapt_blip_weights_state_dict(sd, self)

    #     msg = self.load_state_dict(sd, strict=False)

    #     print("missing keys:")
    #     print([k for k in msg.missing_keys if "pruning_mask" not in k])

    #     # momentum/text_decoder는 제외하고 나머지만 보여주기
    #     keys_to_exclude = ["visual_encoder_m", "text_encoder_m", "vision_proj_m", "text_proj_m", "text_decoder"]
    #     print("unexpected keys:")
    #     print([k for k in msg.unexpected_keys if not any(x in k for x in keys_to_exclude)])


    def load_from_pruned_pretrained(self, pretraining_weights, mask, config, is_eval=False):
        print("\n" + "-"*100)
        print("[Debug] blip_retrieval.py : load_from_pruned_pretrained() -> pruning mask 적용")
        # 1) 프리트레인 가중치 로드(리매핑 포함)
        self.load_pretrained(pretraining_weights, config, is_eval)
        print("-"*100 + "\n")

        # 2) 마스크 로드 + 리매핑
        print("-"*100 + f"\nLoading mask from: {mask}")
        mask_sd = torch.load(mask, map_location="cpu")
        mask_sd = adapt_blip_mask_state_dict(mask_sd, self)   # ★ 추가
        mask_sd = inherit_encoder_momentum_masks(mask_sd)     # 기존 로직 유지

        msg = self.load_state_dict(mask_sd, strict=False)
        
        print("missing keys (mask):")
        print([k for k in msg.missing_keys
            if "pruning_mask" in k and "bias" not in k and "layernorm" not in k.lower()])

        print("unexpected keys (mask):")
        print([k for k in msg.unexpected_keys
            if "pruning_mask" in k and "text_decoder" not in k and "layernorm" not in k.lower()])
        print("-"*100)
    from models.blip.vit import interpolate_pos_embed  # 이미 추가했다고 가정

    def load_pretrained(self, weights_ckpt, *args, **kwargs):
        print("[Debug] blip_captioning.py : load_pretrained()")
        print("Loaded params from: ", weights_ckpt)

        import torch
        ckpt = torch.load(weights_ckpt, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)  # ckpt 구조에 따라 조정

        new_state_dict = {}

        for k, v in state_dict.items():
            # 1) ViT 블록 qkv → q_proj / k_proj / v_proj로 분해
            if "visual_encoder.blocks" in k and ".attn.qkv." in k:
                # k 예시: "visual_encoder.blocks.0.attn.qkv.weight"
                prefix, suffix = k.split(".attn.qkv.")
                # prefix = "visual_encoder.blocks.0"
                # suffix = "weight" 또는 "bias"
                base = prefix + ".attn."  # "visual_encoder.blocks.0.attn."

                if suffix == "weight":
                    # v.shape = (3*dim, dim)
                    dim3, d_in = v.shape
                    dim = dim3 // 3
                    new_state_dict[base + "q_proj.weight"] = v[0:dim, :]
                    new_state_dict[base + "k_proj.weight"] = v[dim:2*dim, :]
                    new_state_dict[base + "v_proj.weight"] = v[2*dim:3*dim, :]
                elif suffix == "bias":
                    # v.shape = (3*dim,)
                    dim3 = v.shape[0]
                    dim = dim3 // 3
                    new_state_dict[base + "q_proj.bias"] = v[0:dim]
                    new_state_dict[base + "k_proj.bias"] = v[dim:2*dim]
                    new_state_dict[base + "v_proj.bias"] = v[2*dim:3*dim]
                else:
                    # 혹시 모를 이상한 suffix 방어용 (거의 안 타겠지만)
                    print("[Warn] unexpected suffix for attn.qkv:", k)
            else:
                new_state_dict[k] = v

        # 2) pos_embed 크기 안 맞으면 interpolate 해서 맞춰주기
        pe_key = "visual_encoder.pos_embed"
        if pe_key in new_state_dict:
            ckpt_pe = new_state_dict[pe_key]
            model_pe = self.visual_encoder.pos_embed
            if ckpt_pe.shape != model_pe.shape:
                print(f"[Debug] interpolate pos_embed: ckpt {ckpt_pe.shape} -> model {model_pe.shape}")
                new_state_dict[pe_key] = interpolate_pos_embed(ckpt_pe, self.visual_encoder)

        # 3) 실제 로딩
        msg = self.load_state_dict(new_state_dict, strict=False)

        print("missing keys:")
        print([k for k in msg.missing_keys if "pruning_mask" not in k])

        keys_to_exclude = ["visual_encoder_m", "text_encoder_m", "vision_proj_m",
                        "text_proj_m", "text_decoder", "text_encoder"]
        print("unexpected keys:")
        print([k for k in msg.unexpected_keys if not any([x in k for x in keys_to_exclude])])



    
    
def blip_vqa(pretrained='',**kwargs):
    model = BLIPVQA(**kwargs)
    if pretrained:
        model, _ = load_checkpoint(model,pretrained)
#         assert(len(msg.missing_keys)==0)
    return model  


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    

import re, torch       
def _chunk_by_outdim(t: torch.Tensor):
    # ViT 계열은 q/k/v가 같은 out_dim이라 대부분 3등분으로 충분
    return torch.chunk(t, 3, dim=0)

def adapt_blip_weights_state_dict(sd: dict, model) -> dict:
    """
    - ...attn.qkv.{weight,bias} → q_proj/k_proj/v_proj.{weight,bias}
    - queue_ptr → ptr_queue (필요 시)
    - visual_encoder.* → visual_encoder_m.* 미러링(모델에 해당 키가 있을 때만)
    """
    new_sd = {}
    msd = model.state_dict()

    for k, v in sd.items():
        # queue 포인터 이름 보정
        if k.endswith('queue_ptr'):
            new_sd[k.replace('queue_ptr', 'ptr_queue')] = v
            continue

        # qkv → q/k/v
        m = re.match(r'^(visual_encoder(?:_m)?\.blocks\.\d+\.attn)\.qkv\.(weight|bias)$', k)
        if m:
            base, wb = m.groups()  # wb ∈ {weight, bias}
            q, k_, v_ = _chunk_by_outdim(v)
            new_sd[f'{base}.q_proj.{wb}'] = q.contiguous()
            new_sd[f'{base}.k_proj.{wb}'] = k_.contiguous()
            new_sd[f'{base}.v_proj.{wb}'] = v_.contiguous()
            continue

        new_sd[k] = v

    # visual_encoder_m가 모델에 있다면, visual_encoder.*를 미러링(없는 키만)
    if any(k.startswith('visual_encoder_m.') for k in msd.keys()):
        for k, v in list(new_sd.items()):
            if k.startswith('visual_encoder.'):
                km = 'visual_encoder_m.' + k[len('visual_encoder.'):]
                if (km in msd) and (km not in new_sd):
                    new_sd[km] = v.clone()

    return new_sd


def adapt_blip_mask_state_dict(mask_sd: dict, model) -> dict:
    """
    pruning_mask도 qkv 기반으로 저장된 경우가 있어, 동일 규칙으로 분해/리네임
    """
    new_sd = {}
    for k, v in mask_sd.items():
        # qkv_pruning_mask → q_/k_/v_ pruning_mask
        m = re.match(r'^(visual_encoder(?:_m)?\.blocks\.\d+\.attn)\.qkv_(pruning_mask)$', k)
        if m:
            base, suffix = m.groups()
            q, k_, v_ = torch.chunk(v, 3, dim=0)
            new_sd[f'{base}.q_proj_{suffix}'] = q.contiguous()
            new_sd[f'{base}.k_proj_{suffix}'] = k_.contiguous()
            new_sd[f'{base}.v_proj_{suffix}'] = v_.contiguous()
            continue

        # queue_ptr 같은 건 마스크엔 잘 없지만 혹시 몰라 그대로
        new_sd[k] = v
    return new_sd
