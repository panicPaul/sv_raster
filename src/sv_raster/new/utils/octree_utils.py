# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

from sv_raster.new.backend import BackendName, get_backend_max_num_levels, get_backend_module

DEFAULT_BACKEND: BackendName = "new_cuda"


def get_max_num_levels(backend_name: BackendName = DEFAULT_BACKEND) -> int:
    return get_backend_max_num_levels(backend_name)


'''
Define many low-level functions for the sparse voxels under Octree layout.
Some notes
1. `octpath` (int64):
    Define the path encodings from Octree root to the leave.
2. `octlevel` (int8):
    The Octree level of the voxels.
    Root is level 0, which is not used. The minimum voxel level is 1.
3. When representing 3D coordinate, we follow the (x, y, z) tuple order.
'''

# Binary encoding of the eight octants
subtree_shift_int64 = torch.tensor([
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
], dtype=torch.int64, device="cuda")
subtree_shift = subtree_shift_int64.float()


def octpath_sanity_check(octpath, octlevel, backend_name: BackendName = DEFAULT_BACKEND):
    '''
    Sanity check if the given pvoxel (octpath, octleve) is valid.

    Input:
        @octpath:  [N, 1]
        @octlevel: [N, 1]
    '''
    assert octlevel.min() >= 1, 'Minimum level should be larger than 1.'
    max_num_levels = get_max_num_levels(backend_name)
    assert octlevel.max() <= max_num_levels, 'Maximum level out of bound.'
    assert len(octpath) == len(octlevel), 'Size mismatch.'
    for lv in range(1, 1 + max_num_levels):
        bit_shift = 3 * (max_num_levels - lv)
        lv_mask = 0b111 << bit_shift
        subtree_id = (octpath & lv_mask) >> bit_shift
        assert ((lv <= octlevel) | (subtree_id == 0)).all(), \
            'Inconsist octpath and octlevel encoding.'


def level_2_vox_size(scene_extent, octlevel):
    '''
    The voxel size at the given levels.

    Input:
        @scene_extent:  [1]
        @octlevel:      [N, 1]
    '''
    # Formula of voxel size: scene_extent * pow(2, -L)
    return torch.ldexp(scene_extent, -octlevel)

def vox_size_2_level(scene_extent, vox_size):
    '''
    The Octree levels in floating point of the given voxel sizes.

    Input:
        @scene_extent:  [1]
        @vox_size:      [N, 1]
    '''
    return -torch.log2(vox_size / scene_extent)

def xyz_2_octpath(xyz, octlevel, scene_center, scene_extent, backend_name: BackendName = DEFAULT_BACKEND):
    '''
    Compute the voxels' octpath containing the input xyz.

    Input:
        @xyz:           [N, 3]
        @octlevel:      [N, 1]
        @scene_center:  [3]
        @scene_extent:  [1]
    Output:
        @octpath:       [N, 1] int64 tensor
    '''
    assert len(xyz.shape) == 2 and xyz.shape[-1] == 3, \
        "Expect xyz in [*, 3] shape."

    scene_min_xyz = scene_center - 0.5 * scene_extent
    vox_size = level_2_vox_size(scene_extent, octlevel)

    ijk = ((xyz - scene_min_xyz) * (1 / vox_size)).long()
    if (ijk < 0).any():
        raise Exception("xyz out of scene bound")
    if (ijk >= (1 << octlevel.long())).any():
        raise Exception("xyz out of scene bound")
    octpath = get_backend_module(backend_name).utils.ijk_2_octpath(ijk, octlevel)
    return octpath


def octpath_decoding(octpath, octlevel, scene_center, scene_extent, backend_name: BackendName = DEFAULT_BACKEND):
    '''
    Compute world-space voxel center positions and voxel size.

    Input:
        @octpath        [N, 1]
        @octlevel:      [N, 1]
        @scene_center:  [3]
        @scene_extent:  [1]
    Output:
        @vox_center:    [N, 3] float tensor
        @vox_size:      [N, 1] float tensor
    '''
    # Sanity check
    octpath_sanity_check(octpath, octlevel, backend_name=backend_name)

    # Ensure trailing dim
    octpath = octpath.reshape(-1, 1)
    octlevel = octlevel.reshape(-1, 1)

    # Compute voxel sizes and centers
    scene_min_xyz = scene_center - 0.5 * scene_extent

    vox_size = level_2_vox_size(scene_extent, octlevel)

    vox_ijk = get_backend_module(backend_name).utils.octpath_2_ijk(octpath, octlevel)
    vox_center = scene_min_xyz + (vox_ijk + 0.5) * vox_size
    return vox_center, vox_size


def gen_gridpoints_coordinate(octpath, octlevel, backend_name: BackendName = DEFAULT_BACKEND):
    '''
    Compute the eight grid points integer coordinate of each voxel.
    The grid point coordinate is represtened as (x, y, z) under the finest Octree level.

    Input:
        @octpath        [N, 1]
        @octlevel:      [N, 1]
    Output:
        @gridpts:       [N, 8, 3] int64 tensor
    '''
    max_num_levels = get_max_num_levels(backend_name)
    vox_ijk = get_backend_module(backend_name).utils.octpath_2_ijk(octpath, octlevel)
    lv2max = (max_num_levels - octlevel).long()
    base_grid_ijk = (vox_ijk << lv2max).view(-1, 1, 3)
    gridpts = base_grid_ijk + (subtree_shift_int64 << lv2max.view(-1, 1, 1))
    return gridpts

def compute_gridpoints_xyz(gridpts, scene_center, scene_extent, backend_name: BackendName = DEFAULT_BACKEND):
    '''
    Compute grid points position from the integer coordinates.

    Input:
        @gridpts:       [N, 3] int64 tensor
                        The grid point int coordinate. See `gen_gridpoints_coordinate`.
        @scene_center:  [3]
        @scene_extent:  [1]
    Output:
        @gridxyz:       [N, 3] float tensor
    '''
    scene_min_xyz = scene_center - 0.5 * scene_extent
    finest_vox_size = level_2_vox_size(
        scene_extent,
        torch.tensor(get_max_num_levels(backend_name), dtype=torch.int64, device="cuda"))
    gridxyz = scene_min_xyz + gridpts * finest_vox_size
    return gridxyz


def gen_children(octpath, octlevel, backend_name: BackendName = DEFAULT_BACKEND):
    '''
    Compute the eight subdivided Octree paths and levels.

    Input:
        @octpath        [N, 1]
        @octlevel:      [N, 1]
    Output:
        @octpath        [N*8, 1]
        @octlevel:      [N*8, 1]
    '''
    # Sanity check
    max_num_levels = get_max_num_levels(backend_name)
    octpath_sanity_check(octpath, octlevel, backend_name=backend_name)

    # Ensure trailing dim
    octpath = octpath.reshape(-1, 1)
    octlevel = octlevel.reshape(-1, 1)

    # Next level
    octlevel = octlevel + 1
    assert octlevel.max() <= max_num_levels, \
        'Maximum level out of bound after subdivision.'

    # The eight octans
    # TODO: remove sanity check
    children = torch.arange(8, dtype=torch.int64, device="cuda")
    assert not (octpath & (children << (3 * (max_num_levels - octlevel)))).any()
    octpath = octpath | (children << (3 * (max_num_levels - octlevel)))

    # Reshape
    octpath = octpath.reshape(-1, 1)
    octlevel = octlevel.repeat_interleave(8, dim=0)
    assert torch.stack([octpath, octlevel]).unique(sorted=True, dim=1).shape[1] == len(octpath)

    return octpath, octlevel


def build_grid_pts_link(octpath, octlevel, backend_name: BackendName = DEFAULT_BACKEND):
    '''
    Build link between voxel and grid_pts.

    Input:
        @octpath:      [N, 1]
        @octlevel:     [N, 1]
    Output:
        @grid_pts_key: [M, 3] int64 tensor
                       The integer coordinate of each grid_pts entry.
                       See `gen_gridpoints_coordinate` and `compute_gridpoints_xyz`.
        @vox_key:      [N, 8] int64 tensor
                       The indices to the eight corner grid points of each voxel.
    '''
    assert octpath.shape == octlevel.shape
    gridpts = gen_gridpoints_coordinate(octpath, octlevel, backend_name=backend_name)
    grid_pts_key, vox_key = gridpts.reshape(-1, 3).unique(dim=0, return_inverse=True)
    grid_pts_key = grid_pts_key.contiguous()
    vox_key = vox_key.reshape(-1, 8).contiguous()
    return grid_pts_key, vox_key


def gen_octpath_dense(outside_level, n_level_inside, backend_name: BackendName = DEFAULT_BACKEND):
    '''
    It contructs octpath for dense (2**n_level_inside) ** 3 voxels inside.
    The region covers the inside part of (outside_level+1) level.

    The final node level is "outside_level + n_level_inside".
    '''
    assert n_level_inside > 0
    max_num_levels = get_max_num_levels(backend_name)

    # Compute the path from outside to the eight inside nodes.
    the_eight = torch.arange(8, dtype=torch.int64, device="cuda")
    octpath = the_eight << (3 * (max_num_levels - 1))
    for k in range(outside_level):
        octpath |= (the_eight ^ 0b111) << (3 * (max_num_levels - (k + 2)))

    # Construct dense voxel inside the eight inside nodes.
    if n_level_inside > 1:
        dense_octpath = torch.arange((2**(n_level_inside-1)) ** 3, dtype=torch.int64, device="cuda")
        dense_octpath = dense_octpath << (3 * (max_num_levels - (outside_level + 1) - (n_level_inside - 1)))
        octpath = (octpath.view(8,1) | dense_octpath)

    octpath = octpath.reshape(-1, 1)
    octlevel = torch.full_like(octpath, outside_level + n_level_inside, dtype=torch.int8)
    return octpath, octlevel


def gen_octpath_shell(shell_level, n_level_inside, backend_name: BackendName = DEFAULT_BACKEND):
    '''
    It contructs octpath for a shell at the given levels.
    The region covers the shell part of shell_level level.

    The final node level is "shell_level + n_level_inside".
    '''
    assert shell_level > 0
    assert n_level_inside > 0
    max_num_levels = get_max_num_levels(backend_name)

    # Compute the path from outside to the eight inside nodes.
    the_eight = torch.arange(8, dtype=torch.int64, device="cuda")
    octpath = the_eight << (3 * (max_num_levels - 1))
    for k in range(shell_level-1):
        octpath |= (the_eight ^ 0b111) << (3 * (max_num_levels - (k + 2)))

    # Produce the shell part
    octpath = octpath.view(8,1) | (the_eight << (3 * (max_num_levels - shell_level - 1)))
    octpath = octpath[the_eight != (the_eight ^ 0b111).view(8, 1)]

    # Construct dense voxel inside the eight inside nodes.
    if n_level_inside > 1:
        dense_octpath = torch.arange((2**(n_level_inside-1)) ** 3, dtype=torch.int64, device="cuda")
        dense_octpath = dense_octpath << (3 * (max_num_levels - shell_level - n_level_inside))
        octpath = (octpath.view(56,1) | dense_octpath)

    octpath = octpath.reshape(-1, 1)
    octlevel = torch.full_like(octpath, shell_level + n_level_inside, dtype=torch.int8)
    return octpath, octlevel


def clamp_level(octpath, octlevel, max_lv, backend_name: BackendName = DEFAULT_BACKEND):
    num_bit_to_mask = 3 * max(0, get_max_num_levels(backend_name) - max_lv)
    octpath = (octpath >> num_bit_to_mask) << num_bit_to_mask
    octlevel = octlevel.clamp_max(max_lv)

    # Keep only the unique voxels
    octpath, octlevel = torch.stack([octpath, octlevel]).unique(sorted=True, dim=1)
    octlevel = octlevel.to(torch.int8)

    return octpath, octlevel
