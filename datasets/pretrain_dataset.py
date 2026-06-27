# Adapted from the following paper :)

# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import os
import sys
import json
import copy
import random
import traceback
import pandas as pd

from random import randint, shuffle
from random import random as rand

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
from transformers import BertTokenizer

from PIL import Image, ImageFile
from transformers import AutoTokenizer # CLIP Tokenizer
from models.blip.blip_captioning import init_tokenizer
from datasets.utils import pre_caption

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class TextMaskingGenerator:
    def __init__(self, tokenizer, mask_prob, mask_max, skipgram_prb=0.2, skipgram_size=3, mask_whole_word=True, use_roberta=False):
        self.id2token = {i: w for w, i in tokenizer.get_vocab().items()}

        self.use_roberta = use_roberta

        for i in range(len(self.id2token)):
            assert i in self.id2token.keys()  # check

        self.cls_token = tokenizer.cls_token
        self.mask_token = tokenizer.mask_token

        self.mask_max = mask_max
        self.mask_prob = mask_prob

        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word

    def get_random_word(self):
        i = randint(0, len(self.id2token) - 1)
        return self.id2token[i]

    def __call__(self, tokens: list):  # tokens: [CLS] + ...
        n_pred = min(self.mask_max, max(1, int(round(len(tokens) * self.mask_prob))))

        # candidate positions of masked tokens
        assert tokens[0] == self.cls_token
        special_pos = set([0])  # will not be masked
        cand_pos = list(range(1, len(tokens)))

        shuffle(cand_pos)
        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end

                if self.use_roberta:
                    while (new_st > 1) and (tokens[new_st][0] != 'Ġ'):
                        new_st -= 1
                    while (new_end < len(tokens)) and (tokens[new_end][0] != 'Ġ'):
                        new_end += 1
                else:
                    # bert, WordPiece
                    while (new_st >= 0) and tokens[new_st].startswith('##'):
                        new_st -= 1
                    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                        new_end += 1

                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        n_real_pred = len(masked_pos)
        if n_real_pred > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = self.mask_token
            elif rand() < 0.5:  # 10%
                tokens[pos] = self.get_random_word()

        return tokens, masked_pos


class XVLMPretrainDataset(Dataset):
    def __init__(self, config, transform, add_eos=False, mask_tokens=False, do_crop=True) -> None:
        super().__init__()

        # set the json files; each file should be a json containing a list of dicts
        self.annotations = []
        if 'coco_file' in config:
            self.coco_file = config['coco_file']
            self.coco_image_root = config['coco_image_root']
            
            # load the annotations into memory
            print(f"Loading COCO data from {self.coco_file}")
            self.coco_data = json.load(open(self.coco_file, 'r'))
            print(f"Loading successful! (loaded {len(self.coco_data)} files)\n")
            self.annotations += self.coco_data
        
        if 'nlvr2_file' in config:
            self.nlvr2_file = config['nlvr2_file']
            self.nlvr2_image_root = config['nlvr2_image_root']

            print(f"Loading nlvr2 data from {self.nlvr2_file}")
            self.nlvr2_data = json.load(open(self.nlvr2_file, 'r'))
            print(f"Loading successful! (loaded {len(self.nlvr2_data)} files)\n")
            self.annotations += self.nlvr2_data
        
        if 'vg_file' in config:
            self.vg_file = config['vg_file']
            self.vg_image_root = config['vg_image_root']

            print(f"Loading Visual Genome data from {self.vg_file}")
            self.vg_data = json.load(open(self.vg_file, 'r'))
            print(f"Loading successful! (loaded {len(self.vg_data)} files)\n")
            self.annotations += self.vg_data

        if 'cc3m_file' in config:
            self.cc3m_file = config['cc3m_file']
            self.cc3m_image_root = config['cc3m_image_root']

            print(f"Loading CC3M data from {self.cc3m_file}")
            self.cc3m_data = json.load(open(self.cc3m_file, 'r'))
            print(f"Loading successful! (loaded {len(self.cc3m_data)} files)\n")
            self.annotations += self.cc3m_data
        
        if 'sbu_file' in config:
            self.sbu_file = config['sbu_file']
            self.sbu_image_root = config['sbu_image_root']

            print(f"Loading SBU Captions data from {self.sbu_file}")
            self.sbu_data = json.load(open(self.sbu_file, 'r'))
            print(f"Loading successful! (loaded {len(self.sbu_data)} files)\n")
            self.annotations += self.sbu_data

        # setup the dictionary keys to retrieve data in __getitem__
        self.image_key = config['image_key']
        self.caption_key = config['caption_key']
        self.dataset_key = config['dataset_key']
        self.do_crop = do_crop

        # setup the transformations
        self.transform = transform
        self.add_eos = add_eos
        self.tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
        self.max_tokens = config['max_tokens']
        self.max_words = config['max_words']
        self.cls_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id

        # initialize the unique set of images of this dataset
        self.all_images = set([ann['image'] for ann in self.annotations])
        self.num_images = len(self.all_images)

        # flags which will decide what is returned by the __getitem__
        # the flag and code below sets everything for MLM
        self.mask_tokens = mask_tokens
        if self.mask_tokens:
            self.mask_generator = TextMaskingGenerator(
                self.tokenizer, config['mask_prob'],
                config['max_masks'], config['skipgram_prb'],
                config['skipgram_size'], config['mask_whole_word']
            )
            self.PAD_mask = -100 
            self.max_masks = config['max_masks']


        # this flag instruct to also return the index passed to __getitem__
        # it is useful for qualitative analysis
        self.return_index = config['return_index'] if 'return_index' in config else False
        self.image_res = config['image_res']


    def __mask_tokens__(self, tokens, text_ids):
        n_tokens = len(tokens)
        tokens_masked, masked_pos = self.mask_generator(copy.deepcopy(tokens))
        text_ids_masked = self.tokenizer.convert_tokens_to_ids(tokens_masked)  # list of int
        masked_ids = [text_ids[p] for p in masked_pos]

        # pad
        n_pad = self.max_tokens - n_tokens
        text_ids = text_ids + [self.pad_token_id] * n_pad
        text_atts = [1] * n_tokens + [0] * n_pad

        text_ids_masked = text_ids_masked + [self.pad_token_id] * n_pad
        n_pad = self.max_masks - len(masked_ids)
        masked_pos = masked_pos + [0] * n_pad
        masked_ids = masked_ids + [self.PAD_mask] * n_pad

        return text_ids, text_atts, text_ids_masked, masked_pos, masked_ids


    def get_image(self, index):
        # grab the correct annotation
        ann = self.annotations[index]

        # load the image into memory and transform it
        if ann[self.dataset_key] == "cc3m":
            image_root = self.cc3m_image_root
        elif ann[self.dataset_key] == "sbu":
            image_root = self.sbu_image_root
        elif ann[self.dataset_key] == "vg":
            image_root = self.vg_image_root
        elif ann[self.dataset_key] == "coco":
            image_root = self.coco_image_root
        
        image = Image.open(os.path.join(image_root, ann[self.image_key])).convert('RGB')

        # crop the image to the bounding box if needed
        if 'bb' in ann:
            [x, y, w, h] = [int(coord) for coord in ann['bb']]
            x_down, y_down = x + w, y + h
            image = image.crop(box=(x, y, x_down, y_down))

        return resize(image, size=(self.image_res, self.image_res), interpolation=Image.BICUBIC)


    def get_image_path(self, index):
        ann = self.annotations[index]
        # load the image into memory and transform it
        if ann[self.dataset_key] == "cc3m":
            image_root = self.cc3m_image_root
        elif ann[self.dataset_key] == "sbu":
            image_root = self.sbu_image_root
        elif ann[self.dataset_key] == "vg":
            image_root = self.vg_image_root
        elif ann[self.dataset_key] == "coco":
            image_root = self.coco_image_root
        return os.path.join(image_root, ann[self.image_key])


    def __private_getitem__(self, index):
        # grab the correct annotation
        ann = self.annotations[index]

        # load the image into memory and transform it
        if ann[self.dataset_key] == "cc3m":
            image_root = self.cc3m_image_root
        elif ann[self.dataset_key] == "sbu":
            image_root = self.sbu_image_root
        elif ann[self.dataset_key] == "vg":
            image_root = self.vg_image_root
        elif ann[self.dataset_key] == "coco":
            image_root = self.coco_image_root
        
        # prepend the root folder only if the image path is relative
        if not ann[self.image_key].startswith(image_root):
            image = Image.open(os.path.join(image_root, ann[self.image_key])).convert('RGB')
        else:
            image = Image.open(ann[self.image_key]).convert('RGB')

        # load the text into memory and transform it
        if isinstance(ann[self.caption_key], list):
            random_index = random.randint(0, len(ann[self.caption_key])-1)
            caption = ann[self.caption_key][random_index]
        elif isinstance(ann[self.caption_key], str):
            caption = ann[self.caption_key]
            random_index = None
        else:
            raise ValueError(f"{self.caption_key} is neither str or a list of str in {ann}. Please fix this.")

        # crop the image to the bounding box if configured
        if 'bb' in ann and self.do_crop:

            if random_index is not None:
                assert isinstance(ann['bb'], list)
                [x, y, w, h] = [int(coord) for coord in ann['bb'][random_index]]
            elif isinstance(ann['bb'][0], (int, float)):
                [x, y, w, h] = [int(coord) for coord in ann['bb']]
            else:
                raise ValueError(f"{ann['bb']} is neither a list of list or a list of int in {ann}. Please fix this.")
            
            x_down, y_down = x + w, y + h
            image = image.crop(box=(x, y, x_down, y_down))

        image = self.transform(image)
        
        caption = pre_caption(caption, self.max_words)
        tokens = self.tokenizer.tokenize(caption)
        tokens = [self.cls_token] + tokens[:self.max_tokens - 1]
        if self.add_eos:
            tokens = tokens[:self.max_tokens - 1]
            tokens += [self.eos_token]
        
        text_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        if self.mask_tokens:
            text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.__mask_tokens__(tokens, text_ids)
            if self.return_index:
                return image, torch.tensor(text_ids), torch.tensor(text_atts), torch.tensor(text_ids_masked), torch.tensor(masked_pos), torch.tensor(masked_ids), index
            else: 
                return image, torch.tensor(text_ids), torch.tensor(text_atts), torch.tensor(text_ids_masked), torch.tensor(masked_pos), torch.tensor(masked_ids)
        
        else:
            n_pad = self.max_tokens - len(tokens)
            text_ids = text_ids + [self.pad_token_id] * n_pad
            text_atts = [1] * len(tokens) + [0] * n_pad
            if self.return_index:
                return image, torch.tensor(text_ids), torch.tensor(text_atts), index
            else:
                return image, torch.tensor(text_ids), torch.tensor(text_atts)
            

    def __getitem__(self, index):
        # call the __private_getitem__ method, but make it
        # more robust to errors with try-catching, retries and logging
        for _ in range(10):
            try:
                return self.__private_getitem__(index)
            except Exception as e:
                print(traceback.format_exc())
                print('encounter broken data: %s' % e)
                print('-'*20)
                sys.stdout.flush()
        
        error_path = self.get_image_path(index)
        raise ValueError(f"Failed to load data from {error_path} after 10 retries. Please check the data.")
    

    def __len__(self):
        return len(self.annotations)


class BlipPretrainDataset(Dataset):
    def __init__(self, config, transform) -> None:
        super().__init__()
        print("<pretrain_dataset.py -> BLIPPreTrainingDataset클래스 init()함수 실행>")
        # set the json files; each file should be a json containing a list of dicts
        self.annotations = []
        if 'vg_file' in config:
            self.vg_file = config['vg_file']
            self.vg_image_root = config['vg_image_root']

            print(f"Loading Visual Genome data from {self.vg_file}")
            self.vg_data = json.load(open(self.vg_file, 'r'))
            print(f"Loading successful! (loaded {len(self.vg_data)} files)\n")
            self.annotations += self.vg_data
        
        
        if 'nlvr2_file' in config:
            self.nlvr2_file = config['nlvr2_file']
            self.nlvr2_image_root = config['nlvr2_image_root']

            print(f"Loading nlvr2 data from {self.nlvr2_file}")
            self.nlvr2_data = json.load(open(self.nlvr2_file, 'r'))
            print(f"Loading successful! (loaded {len(self.nlvr2_data)} files)\n")
            self.annotations += self.nlvr2_data


        if 'coco_file' in config:
            self.coco_file = config['coco_file']
            self.coco_image_root = config['coco_image_root']
            
            # load the annotations into memory
            print(f"Loading COCO data from {self.coco_file}")
            self.coco_data = json.load(open(self.coco_file, 'r'))
            print(f"Loading successful! (loaded {len(self.coco_data)} files)\n")
            self.annotations += self.coco_data
        
        if 'SkyScript_file' in config:
            self.SkyScript_file = config['SkyScript_file']
            self.SkyScript_image_root = config['SkyScript_image_root']

            # load the annotations into memory
            print(f"Loading SkyScript data from {self.SkyScript_file}")
            self.SkyScript_data = json.load(open(self.SkyScript_file, 'r'))
            print(f"Loading successful! (loaded {len(self.SkyScript_data)} files)\n")
            self.annotations += self.SkyScript_data
        
        if 'nwpu_file' in config:
            self.nwpu_file = config['nwpu_file']
            self.nwpu_image_root = config['nwpu_image_root']

            # load the annotations into memory
            print(f"Loading nwpu data from {self.nwpu_file}")
            self.nwpu_data = json.load(open(self.nwpu_file, 'r'))
            print(f"Loading successful! (loaded {len(self.nwpu_data)} files)\n")
            self.annotations += self.nwpu_data    

        if 'rsicd_file' in config:
            self.rsicd_file = config['rsicd_file']
            self.rsicd_image_root = config['rsicd_image_root']

            # load the annotations into memory
            print(f"Loading rsicd data from {self.rsicd_file}")
            self.rsicd_data = json.load(open(self.rsicd_file, 'r'))
            print(f"Loading successful! (loaded {len(self.rsicd_data)} files)\n")
            self.annotations += self.rsicd_data    
        
        if 'rsitmd_file' in config:
            self.rsitmd_file = config['rsitmd_file']
            self.rsitmd_image_root = config['rsitmd_image_root']

            # load the annotations into memory
            print(f"Loading rsitmd data from {self.rsitmd_file}")
            self.rsitmd_data = json.load(open(self.rsitmd_file, 'r'))
            print(f"Loading successful! (loaded {len(self.rsitmd_data)} files)\n")
            self.annotations += self.rsitmd_data

        # if 'cc3m_file' in config:
        #     self.cc3m_file = config['cc3m_file']
        #     self.cc3m_image_root = config['cc3m_image_root']

        #     print(f"Loading CC3M data from {self.cc3m_file}")
        #     self.cc3m_data = json.load(open(self.cc3m_file, 'r'))
        #     print(f"Loading successful! (loaded {len(self.cc3m_data)} files)\n")
        #     self.annotations += self.cc3m_data
        
        if 'flickr30k_file' in config:
            self.flickr30k_file = config['flickr30k_file']
            self.flickr30k_image_root = config['flickr30k_image_root']

            print(f"Loading flickr30k Captions data from {self.flickr30k_file}")
            self.flickr30k_data = json.load(open(self.flickr30k_file, 'r'))
            print(f"Loading successful! (loaded {len(self.flickr30k_data)} files)\n")
            self.annotations += self.flickr30k_data

        if 'sbu_file' in config:
            self.sbu_file = config['sbu_file']
            self.sbu_image_root = config['sbu_image_root']

            print(f"Loading SBU Captions data from {self.sbu_file}")
            self.sbu_data = json.load(open(self.sbu_file, 'r'))
            print(f"Loading successful! (loaded {len(self.sbu_data)} files)\n")
            self.annotations += self.sbu_data

        # setup the dictionary keys to retrieve data in __getitem__
        self.image_key = config['image_key']
        self.caption_key = config['caption_key']
        self.dataset_key = config['dataset_key']

        # setup the image transformations
        self.transform = transform
        
        # setup the text transformations
        self.max_tokens = config['max_tokens']
        self.max_words = config['max_words']
        self.tokenizer = init_tokenizer()
        
        self.cls_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id

        # initialize the unique set of images of this dataset
        self.all_images = set([ann['image'] for ann in self.annotations])
        self.num_images = len(self.all_images)

        # this flag instruct to also return the index passed to __getitem__
        # it is useful for qualitative analysis
        self.return_index = config['return_index'] if 'return_index' in config else False
        self.image_res = config['image_res']


    def get_image_path(self, index):
        ann = self.annotations[index]
        # load the image into memory and transform it
        # if ann[self.dataset_key] == "cc3m":
        #     image_root = self.cc3m_image_root
        if ann[self.dataset_key] == "SkyScript":
            image_root = self.SkyScript_image_root
        elif ann[self.dataset_key] == "nwpu":
            image_root = self.nwpu_image_root
        elif ann[self.dataset_key] == "rsicd":
            image_root = self.rsicd_image_root
        elif ann[self.dataset_key] == "rsitmd":
            image_root = self.rsitmd_image_root
        elif ann[self.dataset_key] == "flickr30k":
            image_root = self.flickr30k_image_root
        elif ann[self.dataset_key] == "sbu":
            image_root = self.sbu_image_root
        elif ann[self.dataset_key] == "vg":
            image_root = self.vg_image_root
        elif ann[self.dataset_key] == "coco":
            image_root = self.coco_image_root
        elif ann[self.dataset_key] == "nlvr2":
            image_root = self.nlvr2_image_root
        return os.path.join(image_root, ann[self.image_key])


    def __private_getitem__(self, index):
        # grab the correct annotation
        ann = self.annotations[index]

        # load the image into memory and transform it
        # if ann[self.dataset_key] == "cc3m":
        #     image_root = self.cc3m_image_root
        if ann[self.dataset_key] == "SkyScript":
            image_root = self.SkyScript_image_root
        elif ann[self.dataset_key] == "nwpu":
             image_root = self.nwpu_image_root
        elif ann[self.dataset_key] == "rsicd":
            image_root = self.rsicd_image_root
        elif ann[self.dataset_key] == "rsitmd":
            image_root = self.rsitmd_image_root
        elif ann[self.dataset_key] == "flickr30k":
            image_root = self.flickr30k_image_root
        elif ann[self.dataset_key] == "sbu":
            image_root = self.sbu_image_root
        elif ann[self.dataset_key] == "vg":
            image_root = self.vg_image_root
        elif ann[self.dataset_key] == "coco":
            image_root = self.coco_image_root
        elif ann[self.dataset_key] == "nlvr2":
            image_root = self.nlvr2_image_root


        # prepend the root folder only if the image path is relative
        if not ann[self.image_key].startswith(image_root):
            image = Image.open(os.path.join(image_root, ann[self.image_key])).convert('RGB')
        else:
            image = Image.open(ann[self.image_key]).convert('RGB')

        # load the text into memory and transform it
        if isinstance(ann[self.caption_key], list):
            random_index = random.randint(0, len(ann[self.caption_key])-1)
            caption = ann[self.caption_key][random_index]
        elif isinstance(ann[self.caption_key], str):
            caption = ann[self.caption_key]
        else:
            raise ValueError(f"{self.caption_key} is neither str or a list of str in {ann}. Please fix this.")

        image = self.transform(image)
        caption = pre_caption(caption, self.max_words)
        if self.return_index:
            return image, caption, index
        else:
            return image, caption
        

    def __getitem__(self, index):
        # call the __private_getitem__ method, but make it
        # more robust to errors with try-catching, retries and logging
        for _ in range(10):
            try:
                return self.__private_getitem__(index)
            except Exception as e:
                print(traceback.format_exc())
                print('encounter broken data: %s' % e)
                print('-'*20)
                sys.stdout.flush()
        
        # if we get here, it means the 10 retries failed
        error_path = self.get_image_path(index)
        raise ValueError(f"Failed to load data from {error_path} after 10 retries. Please check the data.")
    

    def __len__(self):
        return len(self.annotations)
    
    def collate_fn(self, batch):
        print("[Debug] datasets/pretrain_dataset.py -> collate_fn()함수 호출 : dataloader에서 사용할 data 가공")
        images = torch.stack([b[0] for b in batch])
        captions = [b[1] for b in batch]
        text_input = self.tokenizer(
            captions,
            padding='max_length', 
            truncation=True, 
            max_length=30, 
            return_tensors="pt"
        )
        
        if self.return_index:
            indices = [b[-1] for b in batch]
            return images, text_input.input_ids, text_input.attention_mask, torch.tensor(indices, dtype=torch.int64)
        else:
            return images, text_input.input_ids, text_input.attention_mask

class ClipPretrainDataset(Dataset):
    def __init__(self, config, transform) -> None:
        super().__init__()
        print("<pretrain_dataset.py -> ClipPreTrainingDataset클래스 init()함수 실행>")
        # set the json files; each file should be a json containing a list of dicts
        self.annotations = []
        if 'vg_file' in config:
            self.vg_file = config['vg_file']
            self.vg_image_root = config['vg_image_root']

            print(f"Loading Visual Genome data from {self.vg_file}")
            self.vg_data = json.load(open(self.vg_file, 'r'))
            print(f"Loading successful! (loaded {len(self.vg_data)} files)\n")
            self.annotations += self.vg_data
        
        if 'coco_file' in config:
            self.coco_file = config['coco_file']
            self.coco_image_root = config['coco_image_root']
            
            # load the annotations into memory
            print(f"Loading COCO data from {self.coco_file}")
            self.coco_data = json.load(open(self.coco_file, 'r'))
            print(f"Loading successful! (loaded {len(self.coco_data)} files)\n")
            self.annotations += self.coco_data
        
        if 'SkyScript_file' in config:
            self.SkyScript_file = config['SkyScript_file']
            self.SkyScript_image_root = config['SkyScript_image_root']

            # load the annotations into memory
            print(f"Loading SkyScript data from {self.SkyScript_file}")
            self.SkyScript_data = json.load(open(self.SkyScript_file, 'r'))
            print(f"Loading successful! (loaded {len(self.SkyScript_data)} files)\n")
            self.annotations += self.SkyScript_data
        
        if 'nwpu_file' in config:
            self.nwpu_file = config['nwpu_file']
            self.nwpu_image_root = config['nwpu_image_root']

            # load the annotations into memory
            print(f"Loading nwpu data from {self.nwpu_file}")
            self.nwpu_data = json.load(open(self.nwpu_file, 'r'))
            print(f"Loading successful! (loaded {len(self.nwpu_data)} files)\n")
            self.annotations += self.nwpu_data    

        if 'rsicd_file' in config:
            self.rsicd_file = config['rsicd_file']
            self.rsicd_image_root = config['rsicd_image_root']

            # load the annotations into memory
            print(f"Loading rsicd data from {self.rsicd_file}")
            self.rsicd_data = json.load(open(self.rsicd_file, 'r'))
            print(f"Loading successful! (loaded {len(self.rsicd_data)} files)\n")
            self.annotations += self.rsicd_data    
        
        if 'rsitmd_file' in config:
            self.rsitmd_file = config['rsitmd_file']
            self.rsitmd_image_root = config['rsitmd_image_root']

            # load the annotations into memory
            print(f"Loading rsitmd data from {self.rsitmd_file}")
            self.rsitmd_data = json.load(open(self.rsitmd_file, 'r'))
            print(f"Loading successful! (loaded {len(self.rsitmd_data)} files)\n")
            self.annotations += self.rsitmd_data

        if 'wit_file' in config:
            self.wit_file = config['wit_file']
            self.wit_image_root = config['wit_image_root']

            print(f"Loading WIT data from {self.wit_file}")
            self.wit_data = json.load(open(self.wit_file, 'r'))
            print(f"Loading successful! (loaded {len(self.wit_data)} files)\n")
            self.annotations += self.wit_data

        if 'laion_file' in config:
            self.laion_file = config['laion_file']
            self.laion_image_root = config['laion_image_root']

            print(f"Loading LAION data from {self.laion_file}")
            self.laion_data = json.load(open(self.laion_file, 'r'))
            print(f"Loading successful! (loaded {len(self.laion_data)} files)\n")
            self.annotations += self.laion_data

        if 'nocaps_file' in config:
            self.nocaps_file = config['nocaps_file']
            self.nocaps_image_root = config['nocaps_image_root']

            print(f"Loading NoCaps data from {self.nocaps_file}")
            self.nocaps_data = json.load(open(self.nocaps_file, 'r'))
            print(f"Loading successful! (loaded {len(self.nocaps_data)} files)\n")
            self.annotations += self.nocaps_data

        if 'textcaps_file' in config:
            self.textcaps_file = config['textcaps_file']
            self.textcaps_image_root = config['textcaps_image_root']

            print(f"Loading TextCaps data from {self.textcaps_file}")
            self.textcaps_data = json.load(open(self.textcaps_file, 'r'))
            print(f"Loading successful! (loaded {len(self.textcaps_data)} files)\n")
            self.annotations += self.textcaps_data

        if 'pascal50s_file' in config:
            self.pascal50s_file = config['pascal50s_file']
            self.pascal50s_image_root = config['pascal50s_image_root']

            print(f"Loading PASCAL-50S data from {self.pascal50s_file}")
            self.pascal50s_data = json.load(open(self.pascal50s_file, 'r'))
            print(f"Loading successful! (loaded {len(self.pascal50s_data)} files)\n")
            self.annotations += self.pascal50s_data

        if 'flickr30k_file' in config:
            self.flickr30k_file = config['flickr30k_file']
            self.flickr30k_image_root = config['flickr30k_image_root']

            print(f"Loading flickr30k Captions data from {self.flickr30k_file}")
            self.flickr30k_data = json.load(open(self.flickr30k_file, 'r'))
            print(f"Loading successful! (loaded {len(self.flickr30k_data)} files)\n")
            self.annotations += self.flickr30k_data

        if 'sbu_file' in config:
            self.sbu_file = config['sbu_file']
            self.sbu_image_root = config['sbu_image_root']

            print(f"Loading SBU Captions data from {self.sbu_file}")
            self.sbu_data = json.load(open(self.sbu_file, 'r'))
            print(f"Loading successful! (loaded {len(self.sbu_data)} files)\n")
            self.annotations += self.sbu_data

        # setup the dictionary keys to retrieve data in __getitem__
        self.image_key = config['image_key']
        self.caption_key = config['caption_key']
        self.dataset_key = config['dataset_key']

        # setup the image transformations
        self.transform = transform
        
        # setup the text transformations
        self.max_tokens = config['max_tokens']
        self.max_words = config['max_words']
        self.tokenizer =  AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14', use_fast=True)
        self.cls_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id

        # initialize the unique set of images of this dataset
        self.all_images = set([ann['image'] for ann in self.annotations])
        self.num_images = len(self.all_images)

        # this flag instruct to also return the index passed to __getitem__
        # it is useful for qualitative analysis
        self.return_index = config['return_index'] if 'return_index' in config else False
        self.image_res = config['image_res']


    def get_image_path(self, index):
        ann = self.annotations[index]
        # load the image into memory and transform it
        # if ann[self.dataset_key] == "cc3m":
        #     image_root = self.cc3m_image_root
        if ann[self.dataset_key] == "SkyScript":
            image_root = self.SkyScript_image_root
        elif ann[self.dataset_key] == "nwpu":
            image_root = self.nwpu_image_root
        elif ann[self.dataset_key] == "rsicd":
            image_root = self.rsicd_image_root
        elif ann[self.dataset_key] == "rsitmd":
            image_root = self.rsitmd_image_root
        elif ann[self.dataset_key] == "wit":
            image_root = self.wit_image_root
        elif ann[self.dataset_key] == "laion":
            image_root = self.laion_image_root
        elif ann[self.dataset_key] == "nocaps":
            image_root = self.nocaps_image_root
        elif ann[self.dataset_key] == "textcaps":
            image_root = self.textcaps_image_root
        elif ann[self.dataset_key] == "pascal50s":
            image_root = self.pascal50s_image_root
        elif ann[self.dataset_key] == "flickr30k":
            image_root = self.flickr30k_image_root
        elif ann[self.dataset_key] == "sbu":
            image_root = self.sbu_image_root
        elif ann[self.dataset_key] == "vg":
            image_root = self.vg_image_root
        elif ann[self.dataset_key] == "coco":
            image_root = self.coco_image_root
        return os.path.join(image_root, ann[self.image_key])


    def __private_getitem__(self, index):
        # grab the correct annotation
        ann = self.annotations[index]

        # load the image into memory and transform it
        # if ann[self.dataset_key] == "cc3m":
        #     image_root = self.cc3m_image_root
        if ann[self.dataset_key] == "SkyScript":
            image_root = self.SkyScript_image_root
        elif ann[self.dataset_key] == "nwpu":
             image_root = self.nwpu_image_root
        elif ann[self.dataset_key] == "rsicd":
            image_root = self.rsicd_image_root
        elif ann[self.dataset_key] == "rsitmd":
            image_root = self.rsitmd_image_root
        elif ann[self.dataset_key] == "wit":
            image_root = self.wit_image_root
        elif ann[self.dataset_key] == "laion":
            image_root = self.laion_image_root
        elif ann[self.dataset_key] == "nocaps":
            image_root = self.nocaps_image_root
        elif ann[self.dataset_key] == "textcaps":
            image_root = self.textcaps_image_root
        elif ann[self.dataset_key] == "pascal50s":
            image_root = self.pascal50s_image_root
        elif ann[self.dataset_key] == "flickr30k":
            image_root = self.flickr30k_image_root
        elif ann[self.dataset_key] == "sbu":
            image_root = self.sbu_image_root
        elif ann[self.dataset_key] == "vg":
            image_root = self.vg_image_root
        elif ann[self.dataset_key] == "coco":
            image_root = self.coco_image_root


        # prepend the root folder only if the image path is relative
        if not ann[self.image_key].startswith(image_root):
            image = Image.open(os.path.join(image_root, ann[self.image_key])).convert('RGB')
        else:
            image = Image.open(ann[self.image_key]).convert('RGB')

        # load the text into memory and transform it
        if isinstance(ann[self.caption_key], list):
            random_index = random.randint(0, len(ann[self.caption_key])-1)
            caption = ann[self.caption_key][random_index]
        elif isinstance(ann[self.caption_key], str):
            caption = ann[self.caption_key]
        else:
            raise ValueError(f"{self.caption_key} is neither str or a list of str in {ann}. Please fix this.")

        image = self.transform(image)
        caption = pre_caption(caption, self.max_words) if 'pre_caption' in globals() else caption
        
        if self.return_index:
            return image, caption, index
        else:
            return image, caption
        

    def __getitem__(self, index):
        # call the __private_getitem__ method, but make it
        # more robust to errors with try-catching, retries and logging
        for _ in range(10):
            try:
                return self.__private_getitem__(index)
            except Exception as e:
                print(traceback.format_exc())
                print('encounter broken data: %s' % e)
                print('-'*20)
                sys.stdout.flush()
        
        # if we get here, it means the 10 retries failed
        error_path = self.get_image_path(index)
        raise ValueError(f"Failed to load data from {error_path} after 10 retries. Please check the data.")
    

    def __len__(self):
        return len(self.annotations)
    
    def collate_fn(self, batch):
        print("[Debug] datasets/pretrain_dataset.py -> CLIP collate_fn()함수 호출 : dataloader에서 사용할 data 가공")
        images = torch.stack([b[0] for b in batch])
        captions = [b[1] for b in batch]
        text_input = self.tokenizer(
            captions,
            padding='max_length', 
            truncation=True, 
            max_length=self.max_tokens, 
            return_tensors="pt"
        )
        
        if self.return_index:
            indices = [b[-1] for b in batch]
            return images, text_input.input_ids, text_input.attention_mask, torch.tensor(indices, dtype=torch.int64)
        else:
            return images, text_input.input_ids, text_input.attention_mask

class ClipGPretrainDataset(Dataset):
    def __init__(self, config, transform) -> None:
        super().__init__()
        print("<pretrain_dataset.py -> ClipPreTrainingDataset클래스 init()함수 실행>")
        # set the json files; each file should be a json containing a list of dicts
        self.annotations = []
        if 'vg_file' in config:
            self.vg_file = config['vg_file']
            self.vg_image_root = config['vg_image_root']

            print(f"Loading Visual Genome data from {self.vg_file}")
            self.vg_data = json.load(open(self.vg_file, 'r'))
            print(f"Loading successful! (loaded {len(self.vg_data)} files)\n")
            self.annotations += self.vg_data
        
        if 'coco_file' in config:
            self.coco_file = config['coco_file']
            self.coco_image_root = config['coco_image_root']
            
            # load the annotations into memory
            print(f"Loading COCO data from {self.coco_file}")
            self.coco_data = json.load(open(self.coco_file, 'r'))
            print(f"Loading successful! (loaded {len(self.coco_data)} files)\n")
            self.annotations += self.coco_data
        
        if 'SkyScript_file' in config:
            self.SkyScript_file = config['SkyScript_file']
            self.SkyScript_image_root = config['SkyScript_image_root']

            # load the annotations into memory
            print(f"Loading SkyScript data from {self.SkyScript_file}")
            self.SkyScript_data = json.load(open(self.SkyScript_file, 'r'))
            print(f"Loading successful! (loaded {len(self.SkyScript_data)} files)\n")
            self.annotations += self.SkyScript_data
        
        if 'nwpu_file' in config:
            self.nwpu_file = config['nwpu_file']
            self.nwpu_image_root = config['nwpu_image_root']

            # load the annotations into memory
            print(f"Loading nwpu data from {self.nwpu_file}")
            self.nwpu_data = json.load(open(self.nwpu_file, 'r'))
            print(f"Loading successful! (loaded {len(self.nwpu_data)} files)\n")
            self.annotations += self.nwpu_data    

        if 'rsicd_file' in config:
            self.rsicd_file = config['rsicd_file']
            self.rsicd_image_root = config['rsicd_image_root']

            # load the annotations into memory
            print(f"Loading rsicd data from {self.rsicd_file}")
            self.rsicd_data = json.load(open(self.rsicd_file, 'r'))
            print(f"Loading successful! (loaded {len(self.rsicd_data)} files)\n")
            self.annotations += self.rsicd_data    
        
        if 'rsitmd_file' in config:
            self.rsitmd_file = config['rsitmd_file']
            self.rsitmd_image_root = config['rsitmd_image_root']

            # load the annotations into memory
            print(f"Loading rsitmd data from {self.rsitmd_file}")
            self.rsitmd_data = json.load(open(self.rsitmd_file, 'r'))
            print(f"Loading successful! (loaded {len(self.rsitmd_data)} files)\n")
            self.annotations += self.rsitmd_data

        if 'wit_file' in config:
            self.wit_file = config['wit_file']
            self.wit_image_root = config['wit_image_root']

            print(f"Loading WIT data from {self.wit_file}")
            self.wit_data = json.load(open(self.wit_file, 'r'))
            print(f"Loading successful! (loaded {len(self.wit_data)} files)\n")
            self.annotations += self.wit_data

        if 'laion_file' in config:
            self.laion_file = config['laion_file']
            self.laion_image_root = config['laion_image_root']

            print(f"Loading LAION data from {self.laion_file}")
            self.laion_data = json.load(open(self.laion_file, 'r'))
            print(f"Loading successful! (loaded {len(self.laion_data)} files)\n")
            self.annotations += self.laion_data

        if 'nocaps_file' in config:
            self.nocaps_file = config['nocaps_file']
            self.nocaps_image_root = config['nocaps_image_root']

            print(f"Loading NoCaps data from {self.nocaps_file}")
            self.nocaps_data = json.load(open(self.nocaps_file, 'r'))
            print(f"Loading successful! (loaded {len(self.nocaps_data)} files)\n")
            self.annotations += self.nocaps_data

        if 'textcaps_file' in config:
            self.textcaps_file = config['textcaps_file']
            self.textcaps_image_root = config['textcaps_image_root']

            print(f"Loading TextCaps data from {self.textcaps_file}")
            self.textcaps_data = json.load(open(self.textcaps_file, 'r'))
            print(f"Loading successful! (loaded {len(self.textcaps_data)} files)\n")
            self.annotations += self.textcaps_data

        if 'pascal50s_file' in config:
            self.pascal50s_file = config['pascal50s_file']
            self.pascal50s_image_root = config['pascal50s_image_root']

            print(f"Loading PASCAL-50S data from {self.pascal50s_file}")
            self.pascal50s_data = json.load(open(self.pascal50s_file, 'r'))
            print(f"Loading successful! (loaded {len(self.pascal50s_data)} files)\n")
            self.annotations += self.pascal50s_data

        if 'flickr30k_file' in config:
            self.flickr30k_file = config['flickr30k_file']
            self.flickr30k_image_root = config['flickr30k_image_root']

            print(f"Loading flickr30k Captions data from {self.flickr30k_file}")
            self.flickr30k_data = json.load(open(self.flickr30k_file, 'r'))
            print(f"Loading successful! (loaded {len(self.flickr30k_data)} files)\n")
            self.annotations += self.flickr30k_data

        if 'sbu_file' in config:
            self.sbu_file = config['sbu_file']
            self.sbu_image_root = config['sbu_image_root']

            print(f"Loading SBU Captions data from {self.sbu_file}")
            self.sbu_data = json.load(open(self.sbu_file, 'r'))
            print(f"Loading successful! (loaded {len(self.sbu_data)} files)\n")
            self.annotations += self.sbu_data

        if 'nlvr2_file' in config:
            self.nlvr2_file = config['nlvr2_file']
            self.nlvr2_image_root = config['nlvr2_image_root']

            print(f"Loading nlvr2 data from {self.nlvr2_file}")
            self.nlvr2_data = json.load(open(self.nlvr2_file, 'r'))
            print(f"Loading successful! (loaded {len(self.nlvr2_data)} files)\n")
            self.annotations += self.nlvr2_data

        # setup the dictionary keys to retrieve data in __getitem__
        self.image_key = config['image_key']
        self.caption_key = config['caption_key']
        self.dataset_key = config['dataset_key']

        # setup the image transformations
        self.transform = transform
        
        # setup the text transformations
        self.max_tokens = config['max_tokens']
        self.max_words = config['max_words']
        self.tokenizer =  AutoTokenizer.from_pretrained('laion/CLIP-ViT-bigG-14-laion2B-39B-b160k', use_fast=True)
        self.cls_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id

        # initialize the unique set of images of this dataset
        self.all_images = set([ann['image'] for ann in self.annotations])
        self.num_images = len(self.all_images)

        # this flag instruct to also return the index passed to __getitem__
        # it is useful for qualitative analysis
        self.return_index = config['return_index'] if 'return_index' in config else False
        self.image_res = config['image_res']


    def get_image_path(self, index):
        ann = self.annotations[index]
        # load the image into memory and transform it
        # if ann[self.dataset_key] == "cc3m":
        #     image_root = self.cc3m_image_root
        if ann[self.dataset_key] == "SkyScript":
            image_root = self.SkyScript_image_root
        elif ann[self.dataset_key] == "nwpu":
            image_root = self.nwpu_image_root
        elif ann[self.dataset_key] == "rsicd":
            image_root = self.rsicd_image_root
        elif ann[self.dataset_key] == "rsitmd":
            image_root = self.rsitmd_image_root
        elif ann[self.dataset_key] == "wit":
            image_root = self.wit_image_root
        elif ann[self.dataset_key] == "laion":
            image_root = self.laion_image_root
        elif ann[self.dataset_key] == "nocaps":
            image_root = self.nocaps_image_root
        elif ann[self.dataset_key] == "textcaps":
            image_root = self.textcaps_image_root
        elif ann[self.dataset_key] == "pascal50s":
            image_root = self.pascal50s_image_root
        elif ann[self.dataset_key] == "flickr30k":
            image_root = self.flickr30k_image_root
        elif ann[self.dataset_key] == "sbu":
            image_root = self.sbu_image_root
        elif ann[self.dataset_key] == "vg":
            image_root = self.vg_image_root
        elif ann[self.dataset_key] == "coco":
            image_root = self.coco_image_root
        elif ann[self.dataset_key] == "nlvr2":
            image_root = self.nlvr2_image_root
        return os.path.join(image_root, ann[self.image_key])


    def __private_getitem__(self, index):
        # grab the correct annotation
        ann = self.annotations[index]

        # load the image into memory and transform it
        # if ann[self.dataset_key] == "cc3m":
        #     image_root = self.cc3m_image_root
        if ann[self.dataset_key] == "SkyScript":
            image_root = self.SkyScript_image_root
        elif ann[self.dataset_key] == "nwpu":
             image_root = self.nwpu_image_root
        elif ann[self.dataset_key] == "rsicd":
            image_root = self.rsicd_image_root
        elif ann[self.dataset_key] == "rsitmd":
            image_root = self.rsitmd_image_root
        elif ann[self.dataset_key] == "wit":
            image_root = self.wit_image_root
        elif ann[self.dataset_key] == "laion":
            image_root = self.laion_image_root
        elif ann[self.dataset_key] == "nocaps":
            image_root = self.nocaps_image_root
        elif ann[self.dataset_key] == "textcaps":
            image_root = self.textcaps_image_root
        elif ann[self.dataset_key] == "pascal50s":
            image_root = self.pascal50s_image_root
        elif ann[self.dataset_key] == "flickr30k":
            image_root = self.flickr30k_image_root
        elif ann[self.dataset_key] == "sbu":
            image_root = self.sbu_image_root
        elif ann[self.dataset_key] == "vg":
            image_root = self.vg_image_root
        elif ann[self.dataset_key] == "coco":
            image_root = self.coco_image_root
        elif ann[self.dataset_key] == "nlvr2":
            image_root = self.nlvr2_image_root


        # prepend the root folder only if the image path is relative
        if not ann[self.image_key].startswith(image_root):
            image = Image.open(os.path.join(image_root, ann[self.image_key])).convert('RGB')
        else:
            image = Image.open(ann[self.image_key]).convert('RGB')

        # load the text into memory and transform it
        if isinstance(ann[self.caption_key], list):
            random_index = random.randint(0, len(ann[self.caption_key])-1)
            caption = ann[self.caption_key][random_index]
        elif isinstance(ann[self.caption_key], str):
            caption = ann[self.caption_key]
        else:
            raise ValueError(f"{self.caption_key} is neither str or a list of str in {ann}. Please fix this.")

        image = self.transform(image)
        caption = pre_caption(caption, self.max_words) if 'pre_caption' in globals() else caption
        
        if self.return_index:
            return image, caption, index
        else:
            return image, caption
        

    def __getitem__(self, index):
        # call the __private_getitem__ method, but make it
        # more robust to errors with try-catching, retries and logging
        for _ in range(10):
            try:
                return self.__private_getitem__(index)
            except Exception as e:
                print(traceback.format_exc())
                print('encounter broken data: %s' % e)
                print('-'*20)
                sys.stdout.flush()
        
        # if we get here, it means the 10 retries failed
        error_path = self.get_image_path(index)
        raise ValueError(f"Failed to load data from {error_path} after 10 retries. Please check the data.")
    

    def __len__(self):
        return len(self.annotations)
    
    def collate_fn(self, batch):
        print("[Debug] datasets/pretrain_dataset.py -> CLIP collate_fn()함수 호출 : dataloader에서 사용할 data 가공")
        images = torch.stack([b[0] for b in batch])
        captions = [b[1] for b in batch]
        text_input = self.tokenizer(
            captions,
            padding='max_length', 
            truncation=True, 
            max_length=self.max_tokens, 
            return_tensors="pt"
        )
        
        if self.return_index:
            indices = [b[-1] for b in batch]
            return images, text_input.input_ids, text_input.attention_mask, torch.tensor(indices, dtype=torch.int64)
        else:
            return images, text_input.input_ids, text_input.attention_mask

# utils/transforms_blip2.py
from transformers import AutoImageProcessor
from torchvision import transforms




class Blip2PretrainDataset(Dataset):
    def __init__(self, config, transform) -> None:
        super().__init__()
        print("<pretrain_dataset.py -> BLIP2PreTrainingDataset클래스 init()함수 실행>")
        # set the json files; each file should be a json containing a list of dicts
        self.annotations = []
        if 'vg_file' in config:
            self.vg_file = config['vg_file']
            self.vg_image_root = config['vg_image_root']

            print(f"Loading Visual Genome data from {self.vg_file}")
            self.vg_data = json.load(open(self.vg_file, 'r'))
            print(f"Loading successful! (loaded {len(self.vg_data)} files)\n")
            self.annotations += self.vg_data
        
        if 'coco_file' in config:
            self.coco_file = config['coco_file']
            self.coco_image_root = config['coco_image_root']
            
            # load the annotations into memory
            print(f"Loading COCO data from {self.coco_file}")
            self.coco_data = json.load(open(self.coco_file, 'r'))
            print(f"Loading successful! (loaded {len(self.coco_data)} files)\n")
            self.annotations += self.coco_data
        
        if 'SkyScript_file' in config:
            self.SkyScript_file = config['SkyScript_file']
            self.SkyScript_image_root = config['SkyScript_image_root']

            # load the annotations into memory
            print(f"Loading SkyScript data from {self.SkyScript_file}")
            self.SkyScript_data = json.load(open(self.SkyScript_file, 'r'))
            print(f"Loading successful! (loaded {len(self.SkyScript_data)} files)\n")
            self.annotations += self.SkyScript_data
        
        if 'nwpu_file' in config:
            self.nwpu_file = config['nwpu_file']
            self.nwpu_image_root = config['nwpu_image_root']

            # load the annotations into memory
            print(f"Loading nwpu data from {self.nwpu_file}")
            self.nwpu_data = json.load(open(self.nwpu_file, 'r'))
            print(f"Loading successful! (loaded {len(self.nwpu_data)} files)\n")
            self.annotations += self.nwpu_data    

        if 'rsicd_file' in config:
            self.rsicd_file = config['rsicd_file']
            self.rsicd_image_root = config['rsicd_image_root']

            # load the annotations into memory
            print(f"Loading rsicd data from {self.rsicd_file}")
            self.rsicd_data = json.load(open(self.rsicd_file, 'r'))
            print(f"Loading successful! (loaded {len(self.rsicd_data)} files)\n")
            self.annotations += self.rsicd_data    
        
        if 'rsitmd_file' in config:
            self.rsitmd_file = config['rsitmd_file']
            self.rsitmd_image_root = config['rsitmd_image_root']

            # load the annotations into memory
            print(f"Loading rsitmd data from {self.rsitmd_file}")
            self.rsitmd_data = json.load(open(self.rsitmd_file, 'r'))
            print(f"Loading successful! (loaded {len(self.rsitmd_data)} files)\n")
            self.annotations += self.rsitmd_data

        # if 'cc3m_file' in config:
        #     self.cc3m_file = config['cc3m_file']
        #     self.cc3m_image_root = config['cc3m_image_root']

        #     print(f"Loading CC3M data from {self.cc3m_file}")
        #     self.cc3m_data = json.load(open(self.cc3m_file, 'r'))
        #     print(f"Loading successful! (loaded {len(self.cc3m_data)} files)\n")
        #     self.annotations += self.cc3m_data
        
        if 'flickr30k_file' in config:
            self.flickr30k_file = config['flickr30k_file']
            self.flickr30k_image_root = config['flickr30k_image_root']

            print(f"Loading flickr30k Captions data from {self.flickr30k_file}")
            self.flickr30k_data = json.load(open(self.flickr30k_file, 'r'))
            print(f"Loading successful! (loaded {len(self.flickr30k_data)} files)\n")
            self.annotations += self.flickr30k_data

        if 'sbu_file' in config:
            self.sbu_file = config['sbu_file']
            self.sbu_image_root = config['sbu_image_root']

            print(f"Loading SBU Captions data from {self.sbu_file}")
            self.sbu_data = json.load(open(self.sbu_file, 'r'))
            print(f"Loading successful! (loaded {len(self.sbu_data)} files)\n")
            self.annotations += self.sbu_data

        # setup the dictionary keys to retrieve data in __getitem__
        self.image_key = config['image_key']
        self.caption_key = config['caption_key']
        self.dataset_key = config['dataset_key']

        # setup the image transformations
        self.transform = transform
        
        # setup the text transformations
        self.max_tokens = config['max_tokens']
        self.max_words = config['max_words']
        self.model_name = config['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        
        self.cls_token = getattr(self.tokenizer, 'cls_token', None)  # T5면 None
        self.eos_token = getattr(self.tokenizer, 'eos_token', None)  # 일반적으로 "</s>"
        self.pad_token_id = int(getattr(self.tokenizer, 'pad_token_id', 0) or 0)
        # initialize the unique set of images of this dataset
        self.all_images = set([ann['image'] for ann in self.annotations])
        self.num_images = len(self.all_images)

        # this flag instruct to also return the index passed to __getitem__
        # it is useful for qualitative analysis
        self.return_index = config['return_index'] if 'return_index' in config else False
        self.image_res = config['image_res']


    def get_image_path(self, index):
        ann = self.annotations[index]
        # load the image into memory and transform it
        # if ann[self.dataset_key] == "cc3m":
        #     image_root = self.cc3m_image_root
        if ann[self.dataset_key] == "SkyScript":
            image_root = self.SkyScript_image_root
        elif ann[self.dataset_key] == "nwpu":
            image_root = self.nwpu_image_root
        elif ann[self.dataset_key] == "rsicd":
            image_root = self.rsicd_image_root
        elif ann[self.dataset_key] == "rsitmd":
            image_root = self.rsitmd_image_root
        elif ann[self.dataset_key] == "flickr30k":
            image_root = self.flickr30k_image_root
        elif ann[self.dataset_key] == "sbu":
            image_root = self.sbu_image_root
        elif ann[self.dataset_key] == "vg":
            image_root = self.vg_image_root
        elif ann[self.dataset_key] == "coco":
            image_root = self.coco_image_root
        return os.path.join(image_root, ann[self.image_key])


    def __private_getitem__(self, index):
        # grab the correct annotation
        ann = self.annotations[index]

        # load the image into memory and transform it
        # if ann[self.dataset_key] == "cc3m":
        #     image_root = self.cc3m_image_root
        if ann[self.dataset_key] == "SkyScript":
            image_root = self.SkyScript_image_root
        elif ann[self.dataset_key] == "nwpu":
             image_root = self.nwpu_image_root
        elif ann[self.dataset_key] == "rsicd":
            image_root = self.rsicd_image_root
        elif ann[self.dataset_key] == "rsitmd":
            image_root = self.rsitmd_image_root
        elif ann[self.dataset_key] == "flickr30k":
            image_root = self.flickr30k_image_root
        elif ann[self.dataset_key] == "sbu":
            image_root = self.sbu_image_root
        elif ann[self.dataset_key] == "vg":
            image_root = self.vg_image_root
        elif ann[self.dataset_key] == "coco":
            image_root = self.coco_image_root


        # prepend the root folder only if the image path is relative
        if not ann[self.image_key].startswith(image_root):
            image = Image.open(os.path.join(image_root, ann[self.image_key])).convert('RGB')
        else:
            image = Image.open(ann[self.image_key]).convert('RGB')

        # load the text into memory and transform it
        if isinstance(ann[self.caption_key], list):
            random_index = random.randint(0, len(ann[self.caption_key])-1)
            caption = ann[self.caption_key][random_index]
        elif isinstance(ann[self.caption_key], str):
            caption = ann[self.caption_key]
        else:
            raise ValueError(f"{self.caption_key} is neither str or a list of str in {ann}. Please fix this.")

        image = self.transform(image)
        caption = pre_caption(caption, self.max_words)
        if self.return_index:
            return image, caption, index
        else:
            return image, caption
        

    def __getitem__(self, index):
        # call the __private_getitem__ method, but make it
        # more robust to errors with try-catching, retries and logging
        for _ in range(10):
            try:
                return self.__private_getitem__(index)
            except Exception as e:
                print(traceback.format_exc())
                print('encounter broken data: %s' % e)
                print('-'*20)
                sys.stdout.flush()
        
        # if we get here, it means the 10 retries failed
        error_path = self.get_image_path(index)
        raise ValueError(f"Failed to load data from {error_path} after 10 retries. Please check the data.")
    

    def __len__(self):
        return len(self.annotations)
    
    def collate_fn(self, batch):
        print("[Debug] datasets/pretrain_dataset.py -> collate_fn()함수 호출 : dataloader에서 사용할 data 가공")
        images = torch.stack([b[0] for b in batch])
        captions = [b[1] for b in batch]
        text_input = self.tokenizer(
            captions,
            padding='longest', 
            truncation=True, 
            max_length=self.max_tokens, 
            return_tensors="pt"
        )
        
        if self.return_index:
            indices = [b[-1] for b in batch]
            return images, text_input.input_ids, text_input.attention_mask, torch.tensor(indices, dtype=torch.int64)
        else:
            return images, text_input.input_ids, text_input.attention_mask

            
from transformers import AutoTokenizer  # 파일 상단에 이미 있으면 생략

class LLaVAPretrainDataset(Dataset):
    def __init__(self, config, transform) -> None:
        super().__init__()
        print("<pretrain_dataset.py -> LLaVAPretrainDataset init() 함수 실행>")

        # -----------------------
        # 1) annotation 로딩 부분
        #    (Blip2PretrainDataset랑 동일하게 가져가면 됨)
        # -----------------------
        self.annotations = []
        if 'vg_file' in config:
            self.vg_file = config['vg_file']
            self.vg_image_root = config['vg_image_root']

            print(f"Loading Visual Genome data from {self.vg_file}")
            self.vg_data = json.load(open(self.vg_file, 'r'))
            print(f"Loading successful! (loaded {len(self.vg_data)} files)\n")
            self.annotations += self.vg_data
        
        if 'coco_file' in config:
            self.coco_file = config['coco_file']
            self.coco_image_root = config['coco_image_root']
            
            print(f"Loading COCO data from {self.coco_file}")
            self.coco_data = json.load(open(self.coco_file, 'r'))
            print(f"Loading successful! (loaded {len(self.coco_data)} files)\n")
            self.annotations += self.coco_data
        
        if 'SkyScript_file' in config:
            self.SkyScript_file = config['SkyScript_file']
            self.SkyScript_image_root = config['SkyScript_image_root']

            print(f"Loading SkyScript data from {self.SkyScript_file}")
            self.SkyScript_data = json.load(open(self.SkyScript_file, 'r'))
            print(f"Loading successful! (loaded {len(self.SkyScript_data)} files)\n")
            self.annotations += self.SkyScript_data
        
        if 'nwpu_file' in config:
            self.nwpu_file = config['nwpu_file']
            self.nwpu_image_root = config['nwpu_image_root']

            print(f"Loading nwpu data from {self.nwpu_file}")
            self.nwpu_data = json.load(open(self.nwpu_file, 'r'))
            print(f"Loading successful! (loaded {len(self.nwpu_data)} files)\n")
            self.annotations += self.nwpu_data    

        if 'rsicd_file' in config:
            self.rsicd_file = config['rsicd_file']
            self.rsicd_image_root = config['rsicd_image_root']

            print(f"Loading rsicd data from {self.rsicd_file}")
            self.rsicd_data = json.load(open(self.rsicd_file, 'r'))
            print(f"Loading successful! (loaded {len(self.rsicd_data)} files)\n")
            self.annotations += self.rsicd_data    
        
        if 'rsitmd_file' in config:
            self.rsitmd_file = config['rsitmd_file']
            self.rsitmd_image_root = config['rsitmd_image_root']

            print(f"Loading rsitmd data from {self.rsitmd_file}")
            self.rsitmd_data = json.load(open(self.rsitmd_file, 'r'))
            print(f"Loading successful! (loaded {len(self.rsitmd_data)} files)\n")
            self.annotations += self.rsitmd_data

        if 'flickr30k_file' in config:
            self.flickr30k_file = config['flickr30k_file']
            self.flickr30k_image_root = config['flickr30k_image_root']

            print(f"Loading flickr30k Captions data from {self.flickr30k_file}")
            self.flickr30k_data = json.load(open(self.flickr30k_file, 'r'))
            print(f"Loading successful! (loaded {len(self.flickr30k_data)} files)\n")
            self.annotations += self.flickr30k_data

        if 'sbu_file' in config:
            self.sbu_file = config['sbu_file']
            self.sbu_image_root = config['sbu_image_root']

            print(f"Loading SBU Captions data from {self.sbu_file}")
            self.sbu_data = json.load(open(self.sbu_file, 'r'))
            print(f"Loading successful! (loaded {len(self.sbu_data)} files)\n")
            self.annotations += self.sbu_data

        # -----------------------
        # 2) 기본 설정
        # -----------------------
        self.image_key = config['image_key']
        self.caption_key = config['caption_key']
        self.dataset_key = config['dataset_key']

        self.transform = transform
        
        self.max_tokens = config['max_tokens']
        self.max_words = config['max_words']

        # LLaVA 쪽 토크나이저 (config에 HF ID 넣어두는 걸 추천)
        # 예: config['model_name'] = "llava-hf/llava-1.5-7b-hf" 또는 "lmsys/vicuna-7b-v1.5"
        self.model_name = config['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        # LLaMA 계열은 pad_token이 없을 수 있어서 안전하게 처리
        self.cls_token = getattr(self.tokenizer, 'cls_token', None)
        self.eos_token = getattr(self.tokenizer, 'eos_token', None)
        self.pad_token_id = int(getattr(self.tokenizer, 'pad_token_id', 0) or 0)

        self.all_images = set([ann['image'] for ann in self.annotations])
        self.num_images = len(self.all_images)

        self.return_index = config['return_index'] if 'return_index' in config else False
        self.image_res = config['image_res']

    def get_image_path(self, index):
        ann = self.annotations[index]
        if ann[self.dataset_key] == "SkyScript":
            image_root = self.SkyScript_image_root
        elif ann[self.dataset_key] == "nwpu":
            image_root = self.nwpu_image_root
        elif ann[self.dataset_key] == "rsicd":
            image_root = self.rsicd_image_root
        elif ann[self.dataset_key] == "rsitmd":
            image_root = self.rsitmd_image_root
        elif ann[self.dataset_key] == "flickr30k":
            image_root = self.flickr30k_image_root
        elif ann[self.dataset_key] == "sbu":
            image_root = self.sbu_image_root
        elif ann[self.dataset_key] == "vg":
            image_root = self.vg_image_root
        elif ann[self.dataset_key] == "coco":
            image_root = self.coco_image_root
        return os.path.join(image_root, ann[self.image_key])

    def __private_getitem__(self, index):
        ann = self.annotations[index]

        # 이미지 루트 선택
        if ann[self.dataset_key] == "SkyScript":
            image_root = self.SkyScript_image_root
        elif ann[self.dataset_key] == "nwpu":
            image_root = self.nwpu_image_root
        elif ann[self.dataset_key] == "rsicd":
            image_root = self.rsicd_image_root
        elif ann[self.dataset_key] == "rsitmd":
            image_root = self.rsitmd_image_root
        elif ann[self.dataset_key] == "flickr30k":
            image_root = self.flickr30k_image_root
        elif ann[self.dataset_key] == "sbu":
            image_root = self.sbu_image_root
        elif ann[self.dataset_key] == "vg":
            image_root = self.vg_image_root
        elif ann[self.dataset_key] == "coco":
            image_root = self.coco_image_root

        # 이미지 로드
        if not ann[self.image_key].startswith(image_root):
            image = Image.open(os.path.join(image_root, ann[self.image_key])).convert('RGB')
        else:
            image = Image.open(ann[self.image_key]).convert('RGB')

        # 캡션 선택
        if isinstance(ann[self.caption_key], list):
            random_index = random.randint(0, len(ann[self.caption_key]) - 1)
            caption = ann[self.caption_key][random_index]
        elif isinstance(ann[self.caption_key], str):
            caption = ann[self.caption_key]
        else:
            raise ValueError(f"{self.caption_key} is neither str or a list of str in {ann}. Please fix this.")

        image = self.transform(image)
        caption = pre_caption(caption, self.max_words)

        if self.return_index:
            return image, caption, index
        else:
            return image, caption

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.__private_getitem__(index)
            except Exception as e:
                print(traceback.format_exc())
                print('encounter broken data: %s' % e)
                print('-' * 20)
                sys.stdout.flush()

        error_path = self.get_image_path(index)
        raise ValueError(f"Failed to load data from {error_path} after 10 retries. Please check the data.")

    def __len__(self):
        return len(self.annotations)

    def collate_fn(self, batch):
        print("[Debug] datasets/pretrain_dataset.py -> LLaVA collate_fn() 함수 호출 : dataloader에서 사용할 data 가공")
        images = torch.stack([b[0] for b in batch])
        captions = [b[1] for b in batch]

        # LLaMA/ Vicuna 계열은 padding='longest' + max_length 로 잘 동작
        text_input = self.tokenizer(
            captions,
            padding='longest',
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt"
        )

        if self.return_index:
            indices = [b[-1] for b in batch]
            return images, text_input.input_ids, text_input.attention_mask, torch.tensor(indices, dtype=torch.int64)
        else:
            return images, text_input.input_ids, text_input.attention_mask

import os, sys, json, random, traceback
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

# 있으면 그대로 사용
# from datasets.utils import pre_caption


class FlamingoPretrainDataset(Dataset):
    """
    Flamingo(OpenFlamingo) 캘리브레이션/프루닝용 dataset

    반환 기본 형태:
      __getitem__  -> (image_tensor, prompt_str) or (image_tensor, prompt_str, index)
      collate_fn   -> images: (B, N, C, H, W), input_ids: (B, T), attention_mask: (B, T) (+ indices)

    config에서 기대하는 키(네 yaml에 맞춤):
      - max_tokens, max_words
      - image_key, caption_key, dataset_key
      - return_index, image_res
      - num_images (default=1)
      - num_shots (default=0)
      - prompt_template (default="{caption}")
      - (옵션) multi_image_strategy: "repeat" | "random"
    """

    def __init__(self, config, transform) -> None:
        super().__init__()
        print("<pretrain_dataset.py -> FlamingoPretrainDataset init()>")

        self.annotations: List[dict] = []

        # ---- (1) 여러 파일 로딩 로직은 CLIP-G 코드 흐름 그대로 유지 ----
        # 필요하면 계속 추가 가능
        self._maybe_load(config, "vg_file", "vg_image_root", name="Visual Genome", store_attr="vg")
        self._maybe_load(config, "coco_file", "coco_image_root", name="COCO", store_attr="coco")
        self._maybe_load(config, "SkyScript_file", "SkyScript_image_root", name="SkyScript", store_attr="SkyScript")
        self._maybe_load(config, "nwpu_file", "nwpu_image_root", name="NWPU", store_attr="nwpu")
        self._maybe_load(config, "rsicd_file", "rsicd_image_root", name="RSICD", store_attr="rsicd")
        self._maybe_load(config, "rsitmd_file", "rsitmd_image_root", name="RSITMD", store_attr="rsitmd")
        self._maybe_load(config, "flickr30k_file", "flickr30k_image_root", name="Flickr30k", store_attr="flickr30k")
        self._maybe_load(config, "sbu_file", "sbu_image_root", name="SBU", store_attr="sbu")
        self._maybe_load(config, "nlvr2_file", "nlvr2_image_root", name="NLVR2", store_attr="nlvr2")

        # ---- (2) 공통 키 ----
        self.image_key = config["image_key"]
        self.caption_key = config["caption_key"]
        self.dataset_key = config["dataset_key"]
        self.transform = transform
        # ---- (3) 이미지 변환 ----
        self.tokenizer = config.get("tokenizer_obj", None)
        self.image_processor = config.get("image_processor_obj", None)

        if self.tokenizer is None:
            # 폴백: config에 tokenizer_name_or_path를 넣어둔 경우
            tok_name = config.get("tokenizer_name_or_path", None)
            if tok_name is None:
                raise ValueError("FlamingoPretrainDataset needs tokenizer_obj or tokenizer_name_or_path in config.")
            self.tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)

        # pad 토큰 안정화 (Flamingo/LM계열에서 pad가 없을 수 있음)
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # 최후의 수단
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # ---- (4) 텍스트 변환: Flamingo는 LM tokenizer를 써야 함 ----
        self.max_tokens = int(config.get("max_tokens", 256))
        self.max_words = int(config.get("max_words", 40))

        # ---- (5) Flamingo 입력 구성 ----
        self.num_images = int(config.get("num_images", 1))
        self.num_shots = int(config.get("num_shots", 0))
        self.prompt_template = str(config.get("prompt_template", "{caption}"))
        self.multi_image_strategy = str(config.get("multi_image_strategy", "repeat"))

        self.return_index = bool(config.get("return_index", False))
        self.image_res = int(config.get("image_res", 224))

        # ---- 통계 ----
        self.all_images = set([ann.get(self.image_key) for ann in self.annotations if self.image_key in ann])
        self.num_unique_images = len(self.all_images)

    def _maybe_load(self, config: Dict[str, Any], file_key: str, root_key: str, name: str, store_attr: str):
        if file_key in config:
            setattr(self, file_key, config[file_key])
            setattr(self, root_key, config[root_key])

            path = config[file_key]
            print(f"Loading {name} data from {path}")
            data = json.load(open(path, "r"))
            print(f"Loading successful! (loaded {len(data)} files)\n")
            self.annotations += data

    def _get_image_root(self, ann: dict) -> str:
        ds = ann[self.dataset_key]
        if ds == "SkyScript":
            return self.SkyScript_image_root
        elif ds == "nwpu":
            return self.nwpu_image_root
        elif ds == "rsicd":
            return self.rsicd_image_root
        elif ds == "rsitmd":
            return self.rsitmd_image_root
        elif ds == "flickr30k":
            return self.flickr30k_image_root
        elif ds == "sbu":
            return self.sbu_image_root
        elif ds == "vg":
            return self.vg_image_root
        elif ds == "coco":
            return self.coco_image_root
        elif ds == "nlvr2":
            return self.nlvr2_image_root
        else:
            raise ValueError(f"Unknown dataset_key value: {ds}")

    def get_image_path(self, index: int) -> str:
        ann = self.annotations[index]
        image_root = self._get_image_root(ann)
        return os.path.join(image_root, ann[self.image_key])

    def _load_one_image(self, image_root: str, image_path: str):
        # 상대경로면 root 붙임
        if not image_path.startswith(image_root):
            full = os.path.join(image_root, image_path)
        else:
            full = image_path
        img = Image.open(full).convert("RGB")
        return img

    def _pick_caption(self, ann: dict) -> str:
        cap = ann[self.caption_key]
        if isinstance(cap, list):
            return cap[random.randint(0, len(cap) - 1)]
        elif isinstance(cap, str):
            return cap
        raise ValueError(f"{self.caption_key} is neither str nor list[str]. ann={ann}")

    def _format_prompt(self, caption: str) -> str:
        # pre_caption 있으면 유지
        if "pre_caption" in globals():
            caption = pre_caption(caption, self.max_words)
        return self.prompt_template.format(caption=caption)

    def __private_getitem__(self, index: int):
        ann = self.annotations[index]
        image_root = self._get_image_root(ann)

        # ---- (A) 이미지 로드: Flamingo는 N장 지원. 기본은 1장. ----
        # annotation이 image_key에 list를 담고 있으면 그걸 사용.
        raw_img_field = ann[self.image_key]
        img_paths: List[str]
        if isinstance(raw_img_field, list):
            img_paths = raw_img_field
        else:
            img_paths = [raw_img_field]

        # num_images에 맞추기
        if self.num_images <= 1:
            chosen = [img_paths[0]]
        else:
            if len(img_paths) >= self.num_images:
                chosen = img_paths[:self.num_images]
            else:
                # 부족하면 전략에 따라 채움
                if self.multi_image_strategy == "repeat":
                    chosen = (img_paths * (self.num_images // len(img_paths) + 1))[:self.num_images]
                elif self.multi_image_strategy == "random":
                    # 현재 샘플 + 랜덤 다른 샘플 이미지로 채움
                    chosen = img_paths[:]
                    while len(chosen) < self.num_images:
                        ridx = random.randint(0, len(self.annotations) - 1)
                        rann = self.annotations[ridx]
                        chosen.append(rann[self.image_key] if isinstance(rann[self.image_key], str) else rann[self.image_key][0])
                else:
                    raise ValueError(f"Unknown multi_image_strategy={self.multi_image_strategy}")

        images = []
        for p in chosen:
            pil = self._load_one_image(image_root, p)
            if self.transform is not None:
                img_t = self.transform(pil)          # (C,H,W)
            elif self.image_processor is not None:
                out = self.image_processor(images=pil, return_tensors="pt")
                img_t = out["pixel_values"][0]       # (C,H,W)
            else:
                raise ValueError("Either transform or image_processor must be provided.")
            images.append(img_t)

        # (N, C, H, W)
        image_tensor = torch.stack(images, dim=0)

        # ---- (B) 텍스트 로드 ----
        caption = self._pick_caption(ann)
        prompt = self._format_prompt(caption)

        if self.return_index:
            return image_tensor, prompt, index
        else:
            return image_tensor, prompt

    def __getitem__(self, index: int):
        for _ in range(10):
            try:
                return self.__private_getitem__(index)
            except Exception as e:
                print(traceback.format_exc())
                print(f"encounter broken data: {e}")
                print("-" * 20)
                sys.stdout.flush()

        error_path = self.get_image_path(index)
        raise ValueError(f"Failed to load data from {error_path} after 10 retries. Please check the data.")

    def __len__(self):
        return len(self.annotations)

    def collate_fn(self, batch):
        """
        batch item:
          - (N,C,H,W), prompt_str [, index]

        return:
          images: (B,N,C,H,W)
          input_ids: (B,T)
          attention_mask: (B,T)
          (+ indices)
        """
        images = torch.stack([b[0] for b in batch], dim=0)   # (B,N,C,H,W)
        prompts = [b[1] for b in batch]

        text = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )

        if self.return_index:
            indices = [b[-1] for b in batch]
            return images, text.input_ids, text.attention_mask, torch.tensor(indices, dtype=torch.int64)
        else:
            return images, text.input_ids, text.attention_mask
