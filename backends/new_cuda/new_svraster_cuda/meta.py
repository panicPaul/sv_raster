# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from . import _C


MAX_NUM_LEVELS = _C.MAX_NUM_LEVELS
STEP_SZ_SCALE = _C.STEP_SZ_SCALE
MAX_SORT_KEY_BITS = _C.MAX_SORT_KEY_BITS
MAX_PACKED_IMAGE_DIM = _C.MAX_PACKED_IMAGE_DIM
MAX_RENDER_TILES = _C.MAX_RENDER_TILES
