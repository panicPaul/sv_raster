
import os
import importlib

_cuda_tag = os.environ.get("SVRASTER_CUDA_TAG", "")
_module_name = f"svraster_cuda_spline._C_{_cuda_tag}" if _cuda_tag else "svraster_cuda_spline._C"
_C = importlib.import_module(_module_name)
