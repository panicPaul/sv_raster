# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from sv_raster.new.backend import BackendName, get_backend_module
from sv_raster.new.sparse_voxel_gears.constructor import SVConstructor
from sv_raster.new.sparse_voxel_gears.properties import SVProperties
from sv_raster.new.sparse_voxel_gears.renderer import SVRenderer
from sv_raster.new.sparse_voxel_gears.adaptive import SVAdaptive
from sv_raster.new.sparse_voxel_gears.io import SVInOut
from sv_raster.new.sparse_voxel_gears.pooling import SVPooling


class SparseVoxelModel(SVConstructor, SVProperties, SVRenderer, SVAdaptive, SVInOut, SVPooling):

    def __init__(self,
                 backend: BackendName = "new_cuda",
                 n_samp_per_vox=1,       # Number of sampled points per visited voxel
                 sh_degree=3,            # Use 3 * (k+1)^2 params per voxels for view-dependent colors
                 ss=1.5,                 # Super-sampling rates for anti-aliasing
                 white_background=False, # Assum white background
                 black_background=False, # Assum black background
                 max_num_levels=None,    # Runtime octree cap for this model, bounded by the backend maximum
                 ):
        '''
        Setup of the model meta. At this point, no voxel is allocated.
        Use the following methods to allocate voxels and parameters.

        1. `model_load` defined in `src/sparse_voxel_gears/io.py`.
           Load the saved models from a given path.

        2. `model_init` defined in `src/sparse_voxel_gears/constructor.py`.
           Heuristically initial the sparse grid layout and parameters from the training datas.
        '''
        super().__init__()

        self.backend_name = backend
        self.backend = get_backend_module(backend)
        self.n_samp_per_vox = n_samp_per_vox
        self.max_sh_degree = sh_degree
        self.ss = ss
        self.white_background = white_background
        self.black_background = black_background
        self.max_num_levels = max_num_levels if max_num_levels is not None else self.backend.meta.MAX_NUM_LEVELS
        self.color_is_grid = (self.backend_name == "new_cuda_cont")
        self.geo_is_hermite = (self.backend_name == "new_cuda_spline")

        # List the variable names
        self.per_voxel_attr_lst = [
            'octpath', 'octlevel',
            '_subdiv_p',
        ]
        if self.color_is_grid:
            self.per_voxel_param_lst = ['_shs']
            self.grid_pts_param_lst = ['_geo_grid_pts', '_sh0']
        else:
            self.per_voxel_param_lst = ['_sh0', '_shs']
            self.grid_pts_param_lst = ['_geo_grid_pts']

        # To be init from model_init
        self.scene_center = None
        self.scene_extent = None
        self.inside_extent = None
        self.octpath = None
        self.octlevel = None
        self.active_sh_degree = sh_degree

        self._geo_grid_pts = None
        self._sh0 = None
        self._shs = None
        self._subdiv_p = None
