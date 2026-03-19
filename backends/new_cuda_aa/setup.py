import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cuda_tag = os.environ.get("SVRASTER_CUDA_TAG", "")
ext_name = f"svraster_cuda_aa._C_{cuda_tag}" if cuda_tag else "svraster_cuda_aa._C"

setup(
    name="svraster_cuda_aa",
    packages=["svraster_cuda_aa"],
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
                "binding.cpp"
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
