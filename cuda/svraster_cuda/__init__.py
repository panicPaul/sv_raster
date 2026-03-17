from . import meta
from . import utils
from . import renderer
from . import sparse_adam
from . import grid_loss_bw

import os
import importlib

_cuda_tag = os.environ.get("SVRASTER_CUDA_TAG", "")
_module_name = f"svraster_cuda._C_{_cuda_tag}" if _cuda_tag else "svraster_cuda._C"
_C = importlib.import_module(_module_name)
