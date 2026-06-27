from models.xvlm.xvlm import XVLMBase
from models.xvlm.xvlm import build_mlp
from models.xvlm.xvlm import load_pretrained, load_pretrained_weights_and_masks

from models.xvlm.xvlm_pretrain import XVLM as XVLMPretrain
from models.xvlm.xvlm_captioning_pretrain import XVLM as XVLMCaptioningPretrain

from models.xvlm.xvlm_captioning import XVLM as XVLMCaptioning
from models.xvlm.xvlm_vqa import XVLM as XVLMVQA
from models.xvlm.xvlm_retrieval import XVLM as XVLMRetrieval
from models.xvlm.tokenization_bert import BertTokenizer as BertTokenizerForXVLM

from models.blip.blip_captioning import BLIPCaptioning
from models.blip.blip_retrieval import BLIPRetrieval
from models.blip.blip_vqa import BLIPVQA
from models.blip.blip_pretrain import BLIPPretrain

from models.blip2.blip2_retrieval import BLIP2Retrieval
from models.blip2.blip2_captioning import BLIP2Captioning
from models.blip2.blip2_vqa import BLIP2VQA

from models.clip.clip_retrieval import CLIPRetrieval
from transformers import CLIPTokenizer

try:
    from models.clip.clip_diffusion import SD15Generator, SDXLGenerator
except Exception as e:
    SD15Generator, SDXLGenerator = None, None
    print(f"[Warn] diffusers/xformers import failed (ignored unless you use sd*): {e}")

from models.clip.clip_vqa import CLIPVQA
from models.clip.clip_classification import CLIPClassification
from models.clipG.clipG_classification import CLIPGClassification
from models.clipG.clipG_retrieval import CLIPGRetrieval
from models.clipG.clipG_vqa import CLIPGVQA

from models.llava.llava_vqa import LLaVAVQA
from models.qwen_vl import QwenVLVQA
