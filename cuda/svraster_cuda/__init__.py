import importlib
import os

import torch


_cuda_tag = os.environ.get("SVRASTER_CUDA_TAG", "")
_module_name = f"{__name__}._C_{_cuda_tag}" if _cuda_tag else f"{__name__}._C"
_C = importlib.import_module(_module_name)

from . import meta
from . import utils
from . import renderer
from . import sparse_adam
from . import grid_loss_bw
