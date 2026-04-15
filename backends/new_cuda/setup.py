import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


cuda_tag = os.environ.get("SVRASTER_CUDA_TAG", "")
ext_name = f"new_svraster_cuda._C_{cuda_tag}" if cuda_tag else "new_svraster_cuda._C"
cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
cuda_target_include = cuda_home / "targets" / "x86_64-linux" / "include"
cuda_cccl_include = cuda_target_include / "cccl"

include_dirs = []
for path in (cuda_target_include, cuda_cccl_include):
    if path.exists():
        include_dirs.append(str(path))

setup(
    name="new_svraster_cuda",
    packages=["new_svraster_cuda"],
    ext_modules=[
        CUDAExtension(
            name=ext_name,
            sources=[
                "src/raster_state.cu",
                "src/preprocess.cu",
                "src/forward.cu",
                "src/backward.cu",
                "src/geo_params_gather.cu",
                "src/sh_compute.cu",
                "src/tv_compute.cu",
                "src/utils.cu",
                "src/adam_step.cu",
                "binding.cpp",
            ],
            include_dirs=include_dirs,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
