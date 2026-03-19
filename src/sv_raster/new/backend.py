from functools import lru_cache
from typing import Literal


BackendName = Literal["new_cuda", "new_cuda_cont", "new_cuda_spline"]


@lru_cache(maxsize=None)
def get_backend_module(backend_name: BackendName = "new_cuda"):
    if backend_name == "new_cuda":
        import new_svraster_cuda

        return new_svraster_cuda
    if backend_name == "new_cuda_cont":
        import svraster_cuda_cont

        return svraster_cuda_cont
    if backend_name == "new_cuda_spline":
        import svraster_cuda_spline

        return svraster_cuda_spline
    raise ValueError(f"Unknown backend: {backend_name}")


def get_backend_max_num_levels(backend_name: BackendName = "new_cuda") -> int:
    return get_backend_module(backend_name).meta.MAX_NUM_LEVELS


def get_backend_max_render_tiles(backend_name: BackendName = "new_cuda") -> int:
    return get_backend_module(backend_name).meta.MAX_RENDER_TILES
