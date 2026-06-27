
from .mag import Mag
from .rand import Rand
from .snip import SNIP
from .itersnip import IterSNIP
from .lamp import Lamp
from .chita import Chita
from .tamt import TAMT
from .l2 import LayerWiseL2Norm
from .multiflow import MultiFlow
from .smoothflow import SmoothFlow
from .ecoflap import ECoFLaP
from .dlp import DLP
from .varmin import Varmin
from .wanda_llm import WandaPruner, SparseGPTPruner  # ← NEW
from .smoothflow_dakdak import SmoothFlowDakDak
from .varmin_col import Varmin_Col
from .smoothflow_dakdak_col import SmoothFlowDakDakCol
from .wanda_llm_ours import WandaPrunerOurs, SparseGPTPrunerOurs
from .wanda_multi import WandaMulti
from .wanda_eco import WandaEco
from .wanda_etap import WandaEtap
from .varmin_range import VarminRange
from .porta import Porta

try:
    from .smoothflow_dakdak_plot import SmoothFlowDakDakPlot
except ModuleNotFoundError:
    SmoothFlowDakDakPlot = None
try:
    from .smoothflow_dakdak_eco import SmoothFlowDakDakEco
except ModuleNotFoundError:
    SmoothFlowDakDakEco = None
try:
    from .smoothflow_dakdak_grad import SmoothFlowDakDak_Grad
except ModuleNotFoundError:
    SmoothFlowDakDak_Grad = None
try:
    from .smoothflow_dakdak_zeromean import SmoothFlowDakDakZeromean
except ModuleNotFoundError:
    SmoothFlowDakDakZeromean = None
try:
    from .smoothflow_dakdak_diag import SmoothFlowDakDakDiag
except ModuleNotFoundError:
    SmoothFlowDakDakDiag = None
try:
    from .smoothflow_dakdak_multi import SmoothFlowDakDakMulti
except ModuleNotFoundError:
    SmoothFlowDakDakMulti = None

available_pruners = [
    'omp','rand','snip','itersnip','lamp','chita','tamt','l2',
    'multiflow','smoothflow','ecoflap','dlp','varmin',
    'wanda','sparsegpt', 'smoothflow_dakdak', 'varmin_col', 'smoothflow_dakdak_col', 'smoothflow_dakdak_plot',
    'wanda_ours', 'sparsegpt_ours', 'smoothflow_dakdak_eco', 'smoothflow_dakdak_grad', 'smoothflow_dakdak_zeromean',
    'smoothflow_dakdak_diag', 'wanda_multi', 'wanda_eco', 'wanda_etap', 'smoothflow_dakdak_multi', 'varmin_range','porta'
]

def get_pruner_by_name(name, *args, **kwargs):
    pruners = {
        'omp': Mag,
        'porta':Porta,
        'rand': Rand,
        'snip': SNIP,
        'itersnip': IterSNIP,
        'lamp': Lamp,
        'chita': Chita,
        'tamt': TAMT,
        'l2': LayerWiseL2Norm,
        'multiflow': MultiFlow,
        'smoothflow': SmoothFlow,
        'ecoflap': ECoFLaP,
        'dlp': DLP,
        'varmin': Varmin,
        'wanda': WandaPruner,          # ← NEW
        'sparsegpt': SparseGPTPruner,  # ← NEW
        'smoothflow_dakdak' : SmoothFlowDakDak,
        'varmin_col': Varmin_Col,
        'smoothflow_dakdak_col': SmoothFlowDakDakCol,
        'smoothflow_dakdak_plot': SmoothFlowDakDakPlot,
        'wanda_ours': WandaPrunerOurs,          # ← NEW
        'sparsegpt_ours': SparseGPTPrunerOurs,  # ←
        'smoothflow_dakdak_eco' : SmoothFlowDakDakEco,
        'smoothflow_dakdak_grad' : SmoothFlowDakDak_Grad,
        'smoothflow_dakdak_zeromean' : SmoothFlowDakDakZeromean,
        'smoothflow_dakdak_diag' : SmoothFlowDakDakDiag,
        'wanda_multi': WandaMulti,
        'wanda_eco': WandaEco,
        'wanda_etap': WandaEtap,
        'smoothflow_dakdak_multi': SmoothFlowDakDakMulti,
        'varmin_range': VarminRange,
    }
    cls = pruners[name]
    if cls is None:
        raise ModuleNotFoundError(f"Pruner '{name}' is listed but its implementation file is not available.")
    return cls(*args, **kwargs)
