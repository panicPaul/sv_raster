/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#ifndef RASTER_STATE_H_INCLUDED
#define RASTER_STATE_H_INCLUDED

#include "auxiliary.h"

#include <cuda_runtime.h>
#include <torch/extension.h>

namespace RASTER_STATE {

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t);

template <typename T>
static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment);

template<typename T>
size_t required(size_t P);

template<typename T>
size_t required(size_t P, size_t Q);

struct GeometryState
{
    uint32_t* primitive_count;
    uint32_t* resolved_count;
    uint32_t* unresolved_count;
    uint32_t* proxy_count;

    // Voxel duplication related variables.
    // Render primitives are duplicated by the # of touched tile times the # of camera quadrants.
    uint32_t* n_duplicates;
    uint32_t* n_duplicates_scan;
    size_t scan_size;
    char* scanning_temp_space;
    int4* leaf_bboxes;
    uint32_t* leaf_cam_quadrant_bitsets;
    int4* bboxes;

    uint32_t* cam_quadrant_bitsets;
    int64_t* primitive_octree_paths;
    int32_t* primitive_ref_ids;
    uint8_t* primitive_is_proxy;
    uint32_t* primitive_member_starts;
    uint32_t* primitive_member_ends;

    // Temporary leaf and unresolved-member buffers. Sized pessimistically by P.
    uint64_t* unresolved_bucket_keys;
    uint64_t* unresolved_bucket_keys_sorted;
    uint32_t* unresolved_member_ids;
    uint32_t* unresolved_member_ids_sorted;
    uint32_t* unresolved_group_start_flags;
    uint32_t* unresolved_group_ids;
    size_t sort_size;
    char* sorting_temp_space;

    static GeometryState fromChunk(char*& chunk, size_t P);
};

struct ImageState
{
    uint2* ranges;
    uint32_t* tile_last;
    uint32_t* n_contrib;

    static ImageState fromChunk(char*& chunk, size_t N, size_t n_tiles);
};

struct BinningState
{
    size_t sorting_size;
    uint64_t* vox_list_keys_unsorted;
    uint64_t* vox_list_keys;
    SortKey128* vox_list_keys_unsorted_wide;
    SortKey128* vox_list_keys_wide;
    uint32_t* vox_list_unsorted;
    uint32_t* vox_list;
    char* list_sorting_space;

    static BinningState fromChunk(char*& chunk, size_t P, bool use_wide_keys);
};

size_t required_binning_state(size_t P, bool use_wide_keys);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
unpack_ImageState(
    const int image_width, const int image_height,
    const torch::Tensor& imageBuffer);

torch::Tensor filter_geomState(
    const int ori_P,
    const torch::Tensor& indices,
    const torch::Tensor& geomState);

}

#endif
