"""Evaluation functions with a shared signature for VQA, used in vqa.py"""
# 이 파일은 VQA(Vision Question Answering) 모델 평가를 위한 유틸리티를 제공합니다.
# blip_evaluation: BLIP 기반 VQA 평가
# xvlm_evaluation: XVLM 기반 VQA 평가
import os
import random
import json
import torch
import lightning as L
import re
from functools import partial
from utils.misc import mprint as print
import faiss
# 질문 ID에 따라 무작위 답변을 생성하는 디버그용 페이크 함수
# ques_id: 질문 식별자, answer_list: 가능한 답변 목록
def vqa_fake_result(ques_id, answer_list):
    return {"question_id": ques_id, "answer": random.choice(answer_list)}

def normalize_answer(ans: str) -> str:
    ans = ans.strip()
    ans = re.sub(r'\s*%\s*', '%', ans)
    ans = re.sub(r'\s*-\s*', '-', ans)
    ans = re.sub(r'\s+', ' ', ans)
    return ans

# @torch.inference_mode()
# # BLIP 모델을 이용하여 VQA 평가를 수행하는 함수
# # output_dir: 결과를 저장할 디렉토리 경로
# # model: 평가할 VQA 모델
# # data_loader: 이미지∙질문∙질문ID 배치를 제공하는 데이터 로더
# # tokenizer: 질문을 토크나이즈할 토크나이저
# # fabric: Lightning Fabric 객체 (분산 환경 설정)
# # config: 평가 설정을 담은 딕셔너리 (inference 방식, 출력 빈도, k_test 등)
# # split: 검증/테스트 데이터 구분 ("val" 또는 "test")
# # debug_mode: True일 경우 일부 샘플에 페이크 결과 사용
# def blip_evaluation(output_dir, model, data_loader, tokenizer, fabric, config, split="val", debug_mode=False):
#     model.eval()  # 모델을 평가 모드로 전환하여 드롭아웃 등 비활성화
#     device = fabric.device  # 연산에 사용할 디바이스(CPU/GPU)
#     rank = fabric.global_rank  # 분산 환경에서의 프로세스 랭크

#     result = []  # 평가 결과를 저장할 리스트
#     computed = 0  # 처리한 배치 수 카운터

#     # 디버그 모드: 페이크 결과 생성을 위한 partial 함수 준비
#     if debug_mode:
#         vqa_fake_result_partial = partial(vqa_fake_result, answer_list=data_loader.dataset.answer_list)
    
#     # 랭크 기반 추론시: 모든 답변 후보를 토크나이즈하여 미리 준비
#     if config['inference'] == 'rank':   
#         answer_list = data_loader.dataset.answer_list  # 가능한 답변 목록
#         answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)
#         # 모든 후보 입력의 첫 토큰을 BOS로 설정
#         answer_candidates.input_ids[:, 0] = model.tokenizer.bos_token_id
        
#     # 배치별 평가 수행
#     for batch_idx, (image, question, question_id) in enumerate(data_loader):       
#         # 일정 주기마다 진행 상황 출력
#         if batch_idx % config['print_freq'] == 0:
#             print(f"[Evaluation]\tBatch {batch_idx}/{len(data_loader)}") 
        
#         # 디버그 모드에서 일부 이후 샘플은 페이크 결과로 대체
#         if debug_mode and computed > 100:
#             result_for_this_batch = [vqa_fake_result_partial(ques_id.item()) for ques_id in question_id]
#             result += result_for_this_batch

#         # 생성 기반 추론: 모델이 직접 답변 시퀀스를 생성
#         elif config['inference'] == 'generate':
#             print("[Debug] Generation inference 수행")
#             question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)
            
#             answers = model(image, question_input, train=False, inference='generate')
#             # for answer, ques_id in zip(answers, question_id):
#             #     ques_id = int(ques_id.item())
#             #     result.append({"question_id": ques_id, "answer": answer})
#             for answer, ques_id in zip(answers, question_id):
#                # ques_id 가 tensor 면 .item(), 아니면 바로 int()
#                if hasattr(ques_id, "item"):
#                    qid = int(ques_id.item())
#                else:
#                    qid = int(ques_id)
#                # 정규화 적용
#                norm_answer = normalize_answer(answer)    
#                result.append({"question_id": qid, "answer": norm_answer})
#             computed += 1
        
#         # 랭크 기반 추론: 미리 준비한 후보 중 상위 k_test 답안 선택
#         elif config['inference'] == 'rank':
#             print("[Debug] Rank inference 수행")
#             question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)
#             answer_ids = model(image, question_input, answer_candidates, train=False, inference='rank', k_test=config['k_test'])
#             for ques_id, answer_id in zip(question_id, answer_ids):
#                 result.append({"question_id": int(ques_id), "answer": answer_list[answer_id]})
#             computed += 1

#     # 평가 결과를 JSON 파일로 저장
#     outpath = os.path.join(output_dir, f"vqa_{split}_{rank}.json")
#     print(f"Dumping eval file at {outpath}")
#     with open(outpath, "w") as f:
#         json.dump(result, f)
    
#     return result  # 최종 평가 결과 반환

@torch.no_grad()
def blip_evaluation(output_dir, model, data_loader, tokenizer, fabric, config, split="val", debug_mode=False) :
    # test
    model.eval()
    device = fabric.device
    rank = fabric.global_rank
    
    result = []
    computed = 0

    if debug_mode:
        vqa_fake_result_partial = partial(vqa_fake_result, answer_list=data_loader.dataset.answer_list)
    
    if config['inference'] == 'rank':   
        answer_list = data_loader.dataset.answer_list
        answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
        answer_candidates.input_ids[:, 0] = model.tokenizer.bos_token_id
        
    for batch_idx, (image, question, question_id) in enumerate(data_loader):       
        if batch_idx % config['print_freq'] == 0:
            print(f"[Evaluation]\tBatch {batch_idx}/{len(data_loader)}") 
        
        if debug_mode and computed > 100:
            result_for_this_batch = [vqa_fake_result_partial(ques_id.item()) for ques_id in question_id]
            result += result_for_this_batch

        # NOTE: the code for answer generation is not tested, 
        # in the main paper VQA was performed using rank-based inference
        elif config['inference'] == 'generate':
            question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)  
            answers = model(image, question_input, train=False, inference='generate') 
            for answer, ques_id in zip(answers, question_id):
                ques_id = int(ques_id.item())       
                result.append({"question_id":ques_id, "answer":answer})     
            computed += 1        
            
        elif config['inference'] == 'rank':
            question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)  
            answer_ids = model(image, question_input, answer_candidates, train=False, inference='rank', k_test=config['k_test'])
            for ques_id, answer_id in zip(question_id, answer_ids):
                result.append({"question_id":int(ques_id.item()), "answer": answer_list[answer_id]})   
            computed += 1


    outpath = os.path.join(output_dir, f"vqa_{split}_{rank}.json")
    print(f"Dumping eval file at {outpath}")
    with open(outpath, "w") as f:
        json.dump(result, f)
    
    return result


@torch.no_grad()
# XVLM 모델을 이용하여 VQA 평가를 수행하는 함수
# output_dir: 결과를 저장할 디렉토리 경로
# model: 평가할 VQA 모델
# data_loader: 이미지∙질문∙질문ID 배치를 제공하는 데이터 로더
# tokenizer: 질문을 토크나이즈할 토크나이저
# fabric: Lightning Fabric 객체 (분산 환경 설정)
# config: 평가 설정을 담은 딕셔너리 (inference 방식, 출력 빈도, k_test 등)
# split: 검증/테스트 데이터 구분 ("val" 또는 "test")
# debug_mode: True일 경우 일부 샘플에 페이크 결과 사용
def xvlm_evaluation(output_dir, model, data_loader, tokenizer, fabric: L.Fabric, config, split="val", debug_mode=False):
    assert split in ("val", "test")  # split 값이 "val" 또는 "test"인지 확인
    model.eval()  # 모델을 평가 모드로 전환하여 드롭아웃 등 비활성화
    device = fabric.device  # 연산에 사용할 디바이스(CPU/GPU)
    rank = fabric.global_rank  # 분산 환경에서의 프로세스 랭크
    
    # 가능한 답변 목록에 EOS 토큰 추가 후 토크나이즈
    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)

    result = []  # 평가 결과를 저장할 리스트
    computed = 0  # 처리한 배치 수 카운터

    # 디버그 모드: 페이크 결과 생성을 위한 partial 함수 준비
    if debug_mode:
        vqa_fake_result_partial = partial(vqa_fake_result, answer_list=data_loader.dataset.answer_list)
    
    # 배치별 평가 수행
    for batch_idx, (image, question, question_id) in enumerate(data_loader):    
        # 일정 주기마다 진행 상황 출력
        if batch_idx % config['print_freq'] == 0:
            print(f"[Evaluation]\tBatch {batch_idx}/{len(data_loader)}")
        
        # 디버그 모드에서 일부 이후 샘플은 페이크 결과로 대체
        if debug_mode and computed > 100:
            result_for_this_batch = [vqa_fake_result_partial(ques_id.item()) for ques_id in question_id]
            result += result_for_this_batch
        else:
            # 질문을 토크나이즈하여 모델에 입력
            question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)
            # 모델이 상위 k_test 답변 후보를 선택
            topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])
            
            for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
                ques_id = int(ques_id.item())
                _, pred = topk_prob.max(dim=0)  # 가장 높은 확률의 답변 선택
                ans = data_loader.dataset.answer_list[topk_id[pred]]
                result.append({"question_id": ques_id, "answer": ans})
                
            computed += 1

    # 평가 결과를 JSON 파일로 저장
    outpath = os.path.join(output_dir, f"vqa_{split}_{rank}.json")
    print(f"Dumping eval file at {outpath}")
    with open(outpath, "w") as f:
        json.dump(result, f)

    return result  # 최종 평가 결과 반환


def vqa_fake_result(ques_id, answer_list):
    import random
    return {"question_id": ques_id, "answer": random.choice(answer_list)}

def normalize_answer(ans: str) -> str:
    ans = ans.strip()
    ans = re.sub(r'\s*%\s*', '%', ans)
    ans = re.sub(r'\s*-\s*', '-', ans)
    ans = re.sub(r'\s+', ' ', ans)
    return ans

import os, json, re, random
import torch
import lightning as L

# ---------------------------
# 헬퍼: 다양한 encode_* 시그니처 호환
# ---------------------------
def _unwrap_clip(model):
    """래퍼(CLIPGVQA 등)면 .model을, 아니면 본체 그대로."""
    return getattr(model, "model", model)

@torch.inference_mode()
def _encode_text_any(model, tok):
    """
    - 래퍼: model.encode_text(input_ids, attention_mask)
    - 순정 HF: get_text_features(input_ids=..., attention_mask=...)
    둘 다 정규화는 호출부에서 처리 (여기선 raw feature만 반환)
    """
    if hasattr(model, "encode_text"):
        return model.encode_text(tok["input_ids"], tok["attention_mask"])
    m = _unwrap_clip(model)
    return m.get_text_features(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])

@torch.inference_mode()
def _encode_image_any(model, images):
    """
    - 래퍼: model.encode_image(pixel_values)
    - 순정 HF: get_image_features(pixel_values=...)
    """
    if hasattr(model, "encode_image"):
        return model.encode_image(images)
    m = _unwrap_clip(model)
    return m.get_image_features(pixel_values=images)

def _get_logit_scale_any(model) -> float | None:
    """
    래퍼/순정 모두에서 logit_scale(파라미터)을 찾아 exp까지 float로 반환.
    없으면 None.
    """
    m = _unwrap_clip(model)
    ls = getattr(m, "logit_scale", None)
    if ls is None:
        return None
    with torch.no_grad():
        return float(ls.exp().item())
def _normalize_answer_default(ans: str) -> str:
    ans = ans.strip()
    ans = re.sub(r'\s*%\s*', '%', ans)
    ans = re.sub(r'\s*-\s*', '-', ans)
    ans = re.sub(r'\s+', ' ', ans)
    return ans


# ---------------------------
# 실전 절충: Answer-only coarse top-K → Q+A 재랭크
# ---------------------------
@torch.inference_mode()
def clip_evaluation(
    output_dir,
    model,
    data_loader,
    tokenizer,
    fabric: L.Fabric,
    config,
    split: str = "test",
    debug_mode: bool = False,
):
    """
    Zero-shot CLIP VQA (실전 절충):
      1) Answer-only 텍스트 임베딩을 1회 사전 계산(txt_bank)
      2) 이미지 임베딩과 코사인 유사도로 coarse top-K 후보 뽑기
      3) 그 K개만 Q+A 프롬프트로 다시 인코딩해 재랭크(질문 반영)

    필수/옵션 config 키:
      - print_freq: int = 50          # 진행 로그 주기
      - max_tokens: int = 77          # 텍스트 토큰 길이
      - clip_text_batch: int = 1024   # 텍스트 인코딩 배치 크기
      - clip_answer_template: str = "Answer: {a}"
      - clip_qna_template: str = "Question: {q} Answer: {a}"
      - clip_refine_topk: int = 32    # 재랭크할 K (coarse 단계 top-K)
      - clip_keep_text_bank_on_gpu: bool = True  # txt_bank를 GPU에 유지
    """
    assert split in ("val", "test")
    model.eval()
    device = fabric.device
    rank = fabric.global_rank

    # config 값
    print_freq  = int(config.get("print_freq", 50))
    max_tokens  = int(config.get("max_tokens", 77))
    text_bs     = int(config.get("clip_text_batch", 1024))
    a_tmpl      = config.get("clip_answer_template", "Answer: {a}")
    qa_tmpl     = config.get("clip_qna_template", "Question: {q} Answer: {a}")
    K           = int(config.get("clip_refine_topk", 32))
    keep_on_gpu = bool(config.get("clip_keep_text_bank_on_gpu", True))

    # 정답 리스트
    answer_list = data_loader.dataset.answer_list
    A = len(answer_list)
    if A == 0:
        raise ValueError("answer_list is empty.")

    # 정규화 함수(모듈에 이미 있을 경우 그걸 쓰고, 없으면 로컬 기본 사용)
    normalize_answer = globals().get("normalize_answer", _normalize_answer_default)

    # logit_scale (CLIP 계열이면 있음)
    scale = _get_logit_scale_any(model)  # float | None
    use_scale = (scale is not None)

    # --------------------------------
    # 1) Answer-only 텍스트 임베딩 1회 사전 계산
    # --------------------------------
    print("[CLIP-RANK-RERANK] Pre-encoding answer candidates ...")
    texts = [a_tmpl.format(a=a) for a in answer_list]
    tok_all = tokenizer(
        texts, padding="longest", truncation=True, max_length=max_tokens, return_tensors="pt"
    )

    txt_feats = []
    for s in range(0, A, text_bs):
        e = s + text_bs
        tok = {k: v[s:e].to(device, non_blocking=True) for k, v in tok_all.items()}
        feat = _encode_text_any(model, tok)
        feat = torch.nn.functional.normalize(feat, dim=-1)  # (chunk, D)
        txt_feats.append(feat)
    txt_bank = torch.cat(txt_feats, dim=0)  # (A, D)
    if keep_on_gpu:
        txt_bank = txt_bank.to(device, non_blocking=True)
    else:
        txt_bank = txt_bank.cpu()

    # --------------------------------
    # 2) 배치 루프: coarse top-K → 3) 재랭크
    # --------------------------------
    result, computed = [], 0
    for bidx, (images, questions, qids) in enumerate(data_loader):
        if bidx % print_freq == 0:
            print(f"[Evaluation-CLIP-RR]\tBatch {bidx}/{len(data_loader)}")

        if debug_mode and computed > 100:
            # 디버그: 일부는 랜덤
            for qid in qids:
                qid_int = int(qid.item()) if hasattr(qid, "item") else int(qid)
                result.append({"question_id": qid_int, "answer": random.choice(answer_list)})
            computed += 1
            continue

        # 이미지 임베딩
        images = images.to(device, non_blocking=True) if torch.is_tensor(images) else images
        img_feat = _encode_image_any(model, images)  # (B,D) or (B,N,D)
        if img_feat.ndim == 3:
            img_feat = img_feat.mean(dim=1)
        img_feat = torch.nn.functional.normalize(img_feat, dim=-1)  # (B, D)
        B, D = img_feat.shape

        # ---- 2) coarse top-K (Answer-only)
        need_K = min(max(1, K), A)  # 안전 가드
        logits = img_feat @ txt_bank.t()  # (B, A)
        if use_scale:
            logits = logits * scale
        topk_vals, topk_idx = torch.topk(logits, k=need_K, dim=1)  # (B, K)

        # ---- 3) 재랭크 (Q+A)
        # B*K 개 프롬프트 생성
        qa_prompts = []
        for b, q in enumerate(questions):
            for k in range(need_K):
                a = answer_list[topk_idx[b, k].item()]
                qa_prompts.append(qa_tmpl.format(q=q, a=a))

        qa_feats = []
        for s in range(0, len(qa_prompts), text_bs):
            chunk = qa_prompts[s:s+text_bs]
            tok = tokenizer(chunk, padding="longest", truncation=True,
                            max_length=max_tokens, return_tensors="pt").to(device)
            txt = _encode_text_any(model, tok)
            qa_feats.append(torch.nn.functional.normalize(txt, dim=-1))
        qa_feats = torch.cat(qa_feats, dim=0).view(B, need_K, D)  # (B, K, D)

        refined = (img_feat.unsqueeze(1) * qa_feats).sum(dim=-1)  # (B, K), 코사인 점수
        if use_scale:
            refined = refined * scale
        best_k = torch.argmax(refined, dim=1)  # (B,)
        final_idx = topk_idx[torch.arange(B, device=device), best_k]  # (B,)

        # 저장
        for j, qid in enumerate(qids):
            qid_int = int(qid.item()) if hasattr(qid, "item") else int(qid)
            ans = normalize_answer(answer_list[int(final_idx[j].item())])
            result.append({"question_id": qid_int, "answer": ans})

        computed += 1

    # 덤프
    outpath = os.path.join(output_dir, f"vqa_{split}_{rank}.json")
    print(f"Dumping eval file at {outpath}")
    with open(outpath, "w") as f:
        json.dump(result, f)

    return result


import os, json, glob
from typing import Any, Dict, List, Optional, Union
import torch
from evaltools.vqa import vqa_eval
from .vqa import VQA
from .vqaEval import VQAEval
# (전제) 아래 두 클래스를 이미 import했다고 가정:
# from vqa_eval import VQAEval
# from vqa_api  import VQA

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _as_py_int_list(x):
    # qids가 Tensor / list / numpy 등 어떤 형태든 파이썬 int 리스트로
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    return [int(t) for t in x]

def _load_candidates(config: Dict[str, Any]) -> List[str]:
    if "candidates" in config and isinstance(config["candidates"], list):
        cand = config["candidates"]
    elif "answer_list" in config and isinstance(config["answer_list"], str):
        with open(config["answer_list"], "r", encoding="utf-8") as f:
            cand = json.load(f)
    else:
        raise ValueError("rank 모드에는 config['candidates'](list) 또는 config['answer_list'](json 경로)가 필요합니다.")

    # 파일 포맷이 [{"answer": "..."}] 형태면 보정
    if len(cand) > 0 and isinstance(cand[0], dict) and "answer" in cand[0]:
        cand = [c["answer"] for c in cand]
    if not isinstance(cand, list) or len(cand) == 0:
        raise ValueError("candidates가 비었거나 형식이 잘못되었습니다.")
    return cand

@torch.inference_mode()
def blip2_evaluation(
    output_dir: str,
    model,                       # BLIP2VQA 인스턴스
    data_loader,                 # test DataLoader: (images, questions, qids)
    fabric,                      # Lightning Fabric (분산/로그용)
    config: Dict[str, Any],
    split: str = "test",
):
    """
    Zero-shot BLIP-2 평가 파이프라인.
    - config['inference'] ∈ {'generate','rank'}
    - generate:  자유생성 → pred.json 저장 → VQAEval
    - rank:      후보 단어집 기반 top-1 선택 → pred.json 저장 → VQAEval

    필수 config 키:
      - 'inference': 'generate' | 'rank'
      - (rank) 'candidates': List[str]  또는  'answer_list': str(JSON 경로)
      - 'vqa_questions': str (공식 questions.json)
      - 'vqa_annotations': str (공식 annotations.json)

    선택 config 키:
      - 'k_test': int (rank 상위 k; 기본 1)
      - 'max_new_tokens': int (generate)
      - 'num_beams': int (generate)
      - 'pred_filename': str (저장 파일명; 기본 pred_{split}.json)
      - 'run_tag': str (샤드/폴더명에 덧붙일 태그)
    """
    fabric.print(f"[blip2_evaluation] split={split}, inference={config.get('inference')}")
    _ensure_dir(output_dir)

    # ===== 분산 환경 정보 =====
    # Fabric은 rank/world_size 제공. 없다면 기본값.
    rank = getattr(fabric, "global_rank", 0)
    world_size = getattr(fabric, "world_size", 1)
    is_zero = (rank == 0)

    # ===== 설정 파싱 =====
    inference_mode = config.get("inference", "generate")
    assert inference_mode in ("generate", "rank"), "config['inference'] must be 'generate' or 'rank'"

    k_test = int(config.get("k_test", 1))
    max_new_tokens = int(config.get("max_new_tokens", getattr(model, "max_new_tokens", 16)))
    num_beams      = int(config.get("num_beams", getattr(model, "num_beams", 3)))

    # 출력 파일명
    base_name = config.get("pred_filename", f"pred_{split}.json")
    run_tag   = config.get("run_tag", None)
    if run_tag:
        base_name = base_name.replace(".json", f"_{run_tag}.json")

    shard_path = os.path.join(output_dir, base_name.replace(".json", f".rank{rank}.json"))
    final_path = os.path.join(output_dir, base_name)

    # ===== 후보(랭크 전용) =====
    candidates: Optional[List[str]] = None
    if inference_mode == "rank":
        candidates = _load_candidates(config)
        fabric.print(f"[blip2_evaluation] candidates loaded: {len(candidates)}")

    # ===== 추론 루프 =====
    preds = []
    for batch in data_loader:
        # 기대: (images, questions, qids)
        if len(batch) < 3:
            raise RuntimeError("DataLoader는 (images, questions, qids) 3요소를 반환해야 합니다.")
        images, questions, qids = batch[0], batch[1], batch[2]
        qids = _as_py_int_list(qids)

        if inference_mode == "generate":
            texts = model(
                images=images,
                question=questions,
                inference="generate",
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            for qid, ans in zip(qids, texts):
                preds.append({"question_id": qid, "answer": ans})

        else:  # rank
            top_idx = model(
                images=images,
                question=questions,
                inference="rank",
                candidates=candidates,
                k_test=max(1, k_test),
            )
            if isinstance(top_idx, torch.Tensor):
                if top_idx.dim() == 2:
                    top_idx = top_idx[:, 0]  # top-1
                top_idx = top_idx.detach().cpu().tolist()
            for qid, idx in zip(qids, top_idx):
                preds.append({"question_id": qid, "answer": candidates[int(idx)]})

    # 샤드 저장
    with open(shard_path, "w", encoding="utf-8") as f:
        json.dump(preds, f)
    fabric.print(f"[blip2_evaluation] wrote shard -> {shard_path} (N={len(preds)})")

    # 모든 rank 동기화
    if hasattr(fabric, "barrier"):
        fabric.barrier()

    # ===== 샤드 머지 (rank0) =====
    if is_zero:
        merged = []
        pattern = shard_path.replace(f".rank{rank}.json", ".rank*.json")
        for p in sorted(glob.glob(pattern)):
            with open(p, "r", encoding="utf-8") as f:
                merged.extend(json.load(f))

        # question_id로 정렬(선택)
        merged.sort(key=lambda x: x["question_id"])
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(merged, f)
        fabric.print(f"[blip2_evaluation] merged -> {final_path} (total N={len(merged)})")

    # 동기화 후 rank0만 평가 수행
    if hasattr(fabric, "barrier"):
        fabric.barrier()

    results = {
        "overall": None,
        "perQuestionType": {},
        "perAnswerType": {},
        "pred_path": final_path,
        "num_samples": None,
    }

    if is_zero:
        q_path = config.get("vqa_questions") or config.get("vqa_question_file")
        a_path = config.get("vqa_annotations") or config.get("vqa_annotation_file")
        if not (q_path and a_path):
            fabric.print("[blip2_evaluation][WARN] VQA questions/annotations 경로가 없어 점수 계산을 건너뜁니다.")
            # 그래도 개수만 기록
            with open(final_path, "r", encoding="utf-8") as f:
                results["num_samples"] = len(json.load(f))
            return results

        vqa = VQA(annotation_file=a_path, question_file=q_path)
        # === (rank0, 평가 직전) pred ↔ expected 정렬/보정 ===

        with open(final_path, "r", encoding="utf-8") as f:
            merged_preds = json.load(f)

        # 1) 중복 정리 + 빈 답 메우기
        byid = {}
        for p in merged_preds:
            qid = int(p["question_id"])
            ans = str(p.get("answer", "")).strip() or "unknown"
            byid[qid] = {"question_id": qid, "answer": ans}

        # 2) expected qids 로드
        with open(q_path, "r", encoding="utf-8") as f:
            exp_q = json.load(f)["questions"]
        expected = [int(x["question_id"]) for x in exp_q]
        exp_set  = set(expected)
        pred_set = set(byid.keys())

        missing = sorted(exp_set - pred_set)
        extra   = sorted(pred_set - exp_set)
        fabric.print(f"[eval] expected={len(exp_set)} pred(unique)={len(pred_set)} "
                    f"missing={len(missing)} extra={len(extra)}")

        # 3) 누락은 'unknown'으로 채우고, 여분은 제거
        for qid in missing:
            byid[qid] = {"question_id": qid, "answer": "unknown"}
        for qid in extra:
            byid.pop(qid, None)

        # 4) 공식 질문 순서대로 재기록
        aligned = [byid[qid] for qid in expected]
        aligned_path = final_path.replace(".json", ".aligned.json")
        with open(aligned_path, "w", encoding="utf-8") as f:
            json.dump(aligned, f)
        fabric.print(f"[eval] aligned file -> {aligned_path} (N={len(aligned)})")

        vqaRes = vqa.loadRes(aligned_path, quesFile=q_path)
        evaluator = VQAEval(vqa, vqaRes)
        evaluator.evaluate()

        results.update({
            "overall": evaluator.accuracy.get("overall"),
            "perQuestionType": evaluator.accuracy.get("perQuestionType", {}),
            "perAnswerType": evaluator.accuracy.get("perAnswerType", {}),
        })
        with open(final_path, "r", encoding="utf-8") as f:
            results["num_samples"] = len(json.load(f))
        fabric.print(f"[blip2_evaluation] Overall: {results['overall']}")
    return results


@torch.no_grad()
def llava_vqa_evaluation(
    output_dir,
    model,
    data_loader,
    tokenizer,   # LLaVA에서는 안 써도 됨, signature 맞추기용
    fabric,
    config,
    split: str = "val",
    debug_mode: bool = False,
):
    model.eval()
    device = fabric.device
    rank = fabric.global_rank

    result = []
    computed = 0

    print_freq = config.get("print_freq", 100)

    for batch_idx, (image, question, question_id) in enumerate(data_loader):
        if batch_idx % print_freq == 0:
            print(f"[LLaVA Evaluation]\tBatch {batch_idx}/{len(data_loader)}")

        # question: list[str] 그대로 모델에 넘김
        answers = model(
            image=image,          # tensor (B, C, H, W)
            question=question,    # List[str]
            train=False,
            inference="generate",
        )

        for ans, qid in zip(answers, question_id):
            result.append({
                "question_id": int(qid.item()),
                "answer": ans,
            })
        computed += 1

    outpath = os.path.join(output_dir, f"vqa_{split}_{rank}.json")
    print(f"[LLaVA Evaluation] Dumping eval file at {outpath}")
    with open(outpath, "w") as f:
        json.dump(result, f)

    return result

#(mf) hai_kms@ubuntu2204:/data/hai_kms/multiflow/images/scienceqa$ python -c "import transformers; print(transformers.__version__)"
#4.54.1
@torch.no_grad()
def llava_scienceqa_evaluation(
    output_dir,
    model,
    data_loader,
    tokenizer,   # 안 써도 signature 맞추기용
    fabric,
    config,
    split: str = "test",
    debug_mode: bool = False,
):
    model.eval()
    device = fabric.device
    rank = fabric.global_rank

    detailed_results = []   # 메모리 안에서 디버그용으로 쓸 full 정보
    json_results = []       # 파일에 저장할 최소 정보 (question_id, answer)

    total = 0
    correct = 0

    print_freq = config.get("print_freq", 100)

    for batch_idx, (image, prompt, label, question_id) in enumerate(data_loader):
        if batch_idx % print_freq == 0:
            print(f"[LLaVA ScienceQA Eval]\tBatch {batch_idx}/{len(data_loader)}")

        answers = model(
            image=image.to(device),
            question=prompt,   # list[str]
            train=False,
            inference="generate",
        )

        for ans, gt, qid in zip(answers, label, question_id):
            ans_str = str(ans).strip()
            ans_up = ans_str.upper()

            # GT를 문자로 통일
            if isinstance(gt, int):
                gt_letter = "ABCDE"[gt]
            else:
                gt_letter = str(gt).strip().upper()

            # --- 예측 문자 뽑기 (조금 더 안전하게) ---
            pred_letter = None

            # 우선 "A.", "A)", "A:" 등 앞부분 패턴부터 체크
            for ch in ["A", "B", "C", "D", "E"]:
                if ans_up.startswith(ch) or ans_up.startswith(ch + ".") or ans_up.startswith(ch + ")") or ans_up.startswith(ch + ":"):
                    pred_letter = ch
                    break

            # 그래도 못 찾으면 그냥 포함 여부로 fallback
            if pred_letter is None:
                for ch in ["A", "B", "C", "D", "E"]:
                    if ch in ans_up:
                        pred_letter = ch
                        break

            ok = (pred_letter == gt_letter)
            total += 1
            correct += int(ok)

            if hasattr(qid, "item"):
                qid_val = int(qid.item())
            else:
                qid_val = int(qid)

            detailed_results.append({
                "question_id": qid_val,
                "answer_raw": ans_str,
                "pred_letter": pred_letter,
                "gt_letter": gt_letter,
                "correct": bool(ok),
            })

            # ScienceQA 후처리용으로는 이것만 있으면 됨
            json_results.append({
                "question_id": qid_val,
                "answer": ans_str,
            })

    acc = correct / max(total, 1)
    print(f"[LLaVA ScienceQA] Accuracy = {acc * 100:.2f}% (rank={rank})")

    outpath = os.path.join(output_dir, f"vqa_{split}_{rank}.json")
    print(f"[LLaVA ScienceQA] Dumping eval file at {outpath}")
    with open(outpath, "w") as f:
        # ★ 여기서는 리스트만 저장한다 ★
        json.dump(json_results, f)

    # 반환값은 네가 디버깅하기 좋게 full 정보 돌려줘도 됨
    return acc, detailed_results
