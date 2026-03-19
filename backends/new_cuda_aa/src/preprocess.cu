/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#include "preprocess.h"
#include "raster_state.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace PREPROCESS {

namespace {

__global__ void init_geom_state_cuda(RASTER_STATE::GeometryState geomState, int P)
{
    if (cg::this_grid().thread_rank() == 0)
    {
        *geomState.primitive_count = 0;
        *geomState.resolved_count = 0;
        *geomState.unresolved_count = 0;
        *geomState.proxy_count = 0;
    }
    for (int idx = cg::this_grid().thread_rank(); idx < P; idx += cg::this_grid().size())
    {
        geomState.n_duplicates[idx] = 0;
        geomState.leaf_bboxes[idx] = make_int4(0, 0, -1, -1);
        geomState.leaf_cam_quadrant_bitsets[idx] = 0;
        geomState.primitive_is_proxy[idx] = 0;
        geomState.primitive_member_starts[idx] = 0;
        geomState.primitive_member_ends[idx] = 0;
    }
}

__global__ void preprocess_leaf_cuda(
    const int P,
    const int W,
    const int H,
    const float tan_fovx,
    const float tan_fovy,
    const float focal_x,
    const float focal_y,
    const float cx,
    const float cy,
    const float* __restrict__ w2c_matrix,
    const float* __restrict__ c2w_matrix,
    const float near,
    const int64_t* __restrict__ octree_paths,
    const int8_t* __restrict__ octlevels,
    const float3* __restrict__ vox_centers,
    const float* __restrict__ vox_lengths,
    int* __restrict__ out_n_duplicates,
    RASTER_STATE::GeometryState geomState,
    const dim3 tile_grid)
{
    const int idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    out_n_duplicates[idx] = 0;

    const float3 vox_c = vox_centers[idx];
    const float vox_r = 0.5f * vox_lengths[idx];
    const float3 ro = last_col_3x4(c2w_matrix);
    float w2c[12];
    for (int i = 0; i < 12; ++i)
        w2c[i] = w2c_matrix[i];

    const float3 rel_pos = vox_c - ro;
    if (dot(rel_pos, rel_pos) < near * near)
        return;

    uint32_t quadrant_bitset = 0;
    float2 coord_min = {1e9f, 1e9f};
    float2 coord_max = {-1e9f, -1e9f};
    for (int i = 0; i < 8; ++i)
    {
        float3 shift = make_float3(
            (float)(((i & 4) >> 2) * 2 - 1),
            (float)(((i & 2) >> 1) * 2 - 1),
            (float)(((i & 1)) * 2 - 1));
        const float3 world_corner = vox_c + vox_r * shift;
        const float3 cam_corner = transform_3x4(w2c, world_corner);
        if (cam_corner.z < near)
            continue;

        const float inv_z = 1.0f / cam_corner.z;
        const float2 corner_coord = make_float2(cam_corner.x * inv_z, cam_corner.y * inv_z);
        const int quadrant_id = compute_corner_quadrant_id(world_corner, ro);

        coord_min = vec_min(coord_min, corner_coord);
        coord_max = vec_max(coord_max, corner_coord);
        quadrant_bitset |= (1u << quadrant_id);
    }

    const float cx_h = cx - 0.5f;
    const float cy_h = cy - 0.5f;
    const float2 bbox_min = {
        max(focal_x * coord_min.x + cx_h, 0.0f),
        max(focal_y * coord_min.y + cy_h, 0.0f)};
    const float2 bbox_max = {
        min(focal_x * coord_max.x + cx_h, (float)W),
        min(focal_y * coord_max.y + cy_h, (float)H)};
    if (bbox_min.x > bbox_max.x || bbox_min.y > bbox_max.y)
        return;

    const int4 bbox = {
        (int)lrintf(bbox_min.x),
        (int)lrintf(bbox_min.y),
        (int)lrintf(bbox_max.x),
        (int)lrintf(bbox_max.y),
    };
    geomState.leaf_bboxes[idx] = bbox;
    geomState.leaf_cam_quadrant_bitsets[idx] = quadrant_bitset;

    uint2 tile_min, tile_max;
    getBboxTileRectXYXY(bbox, tile_min, tile_max, tile_grid);
    const int tiles_touched = (1 + (int)tile_max.y - (int)tile_min.y) * (1 + (int)tile_max.x - (int)tile_min.x);
    if (tiles_touched <= 0)
        return;

    const int quadrant_touched = __popc(quadrant_bitset);
    out_n_duplicates[idx] = tiles_touched * quadrant_touched;

    const float width_px = bbox_max.x - bbox_min.x;
    const float height_px = bbox_max.y - bbox_min.y;
    const bool unresolved = octlevels != nullptr && width_px < 2.f && height_px < 2.f;

    if (!unresolved)
    {
        const uint32_t prim_id = atomicAdd(geomState.resolved_count, 1u);
        geomState.primitive_ref_ids[prim_id] = idx;
        return;
    }

    const int64_t lv = (int64_t)octlevels[idx];
    const int64_t ascend_x = (int64_t)ceilf(log2f(2.f / max(width_px, 1e-6f)));
    const int64_t ascend_y = (int64_t)ceilf(log2f(2.f / max(height_px, 1e-6f)));
    const int64_t ascend = min(max(ascend_x, ascend_y), lv - 1);
    const int64_t ancestor_level = ((lv - ascend) > 1) ? (lv - ascend) : 1;
    const int64_t bits_to_mask = 3LL * (MAX_NUM_LEVELS - ancestor_level);
    const uint64_t ancestor_path = (uint64_t)((octree_paths[idx] >> bits_to_mask) << bits_to_mask);

    const uint32_t unresolved_id = atomicAdd(geomState.unresolved_count, 1u);
    geomState.unresolved_bucket_keys[unresolved_id] = ancestor_path;
    geomState.unresolved_member_ids[unresolved_id] = idx;
}

__global__ void mark_proxy_group_starts_cuda(
    const uint32_t U,
    const uint64_t* __restrict__ sorted_keys,
    uint32_t* __restrict__ start_flags)
{
    const int idx = cg::this_grid().thread_rank();
    if (idx >= (int)U)
        return;
    start_flags[idx] = (idx == 0 || sorted_keys[idx] != sorted_keys[idx - 1]) ? 1u : 0u;
}

__global__ void assemble_resolved_primitives_cuda(
    const uint32_t R,
    const int* __restrict__ out_n_duplicates,
    const int64_t* __restrict__ octree_paths,
    const int4* __restrict__ leaf_bboxes,
    const uint32_t* __restrict__ leaf_quadrant_bitsets,
    RASTER_STATE::GeometryState geomState)
{
    const int idx = cg::this_grid().thread_rank();
    if (idx >= (int)R)
        return;

    const int leaf_id = geomState.primitive_ref_ids[idx];
    geomState.primitive_is_proxy[idx] = 0;
    geomState.primitive_member_starts[idx] = 0;
    geomState.primitive_member_ends[idx] = 0;
    geomState.primitive_octree_paths[idx] = octree_paths[leaf_id];
    geomState.bboxes[idx] = leaf_bboxes[leaf_id];
    geomState.cam_quadrant_bitsets[idx] = leaf_quadrant_bitsets[leaf_id];
    geomState.n_duplicates[idx] = max(out_n_duplicates[leaf_id], 0);
  }

__global__ void init_proxy_primitives_cuda(
    const uint32_t R,
    const uint32_t G,
    RASTER_STATE::GeometryState geomState)
{
    const int idx = cg::this_grid().thread_rank();
    if (idx >= (int)G)
        return;
    const uint32_t prim_id = R + idx;
    geomState.primitive_is_proxy[prim_id] = 1;
    geomState.primitive_member_starts[prim_id] = 0;
    geomState.primitive_member_ends[prim_id] = 0;
    geomState.primitive_ref_ids[prim_id] = -1;
    geomState.bboxes[prim_id] = make_int4(1 << 30, 1 << 30, -(1 << 30), -(1 << 30));
    geomState.cam_quadrant_bitsets[prim_id] = 0;
    geomState.n_duplicates[prim_id] = 0;
}

__global__ void assemble_proxy_primitives_cuda(
    const uint32_t R,
    const uint32_t U,
    const uint32_t* __restrict__ group_start_flags,
    const uint32_t* __restrict__ group_ids,
    const uint64_t* __restrict__ sorted_keys,
    const uint32_t* __restrict__ sorted_member_ids,
    const int4* __restrict__ leaf_bboxes,
    const uint32_t* __restrict__ leaf_quadrant_bitsets,
    RASTER_STATE::GeometryState geomState)
{
    const int idx = cg::this_grid().thread_rank();
    if (idx >= (int)U)
        return;

    const uint32_t group_id = group_ids[idx] - 1u;
    const uint32_t prim_id = R + group_id;
    const uint32_t leaf_id = sorted_member_ids[idx];
    const int4 bbox = leaf_bboxes[leaf_id];

    if (group_start_flags[idx])
    {
        geomState.primitive_octree_paths[prim_id] = (int64_t)sorted_keys[idx];
        geomState.primitive_ref_ids[prim_id] = (int32_t)leaf_id;
        geomState.primitive_member_starts[prim_id] = idx;
    }
    if (idx + 1 == U || group_start_flags[idx + 1])
        geomState.primitive_member_ends[prim_id] = idx + 1;

    atomicMin(&geomState.bboxes[prim_id].x, bbox.x);
    atomicMin(&geomState.bboxes[prim_id].y, bbox.y);
    atomicMax(&geomState.bboxes[prim_id].z, bbox.z);
    atomicMax(&geomState.bboxes[prim_id].w, bbox.w);
    atomicOr(&geomState.cam_quadrant_bitsets[prim_id], leaf_quadrant_bitsets[leaf_id]);
}

__global__ void finalize_proxy_duplicates_cuda(
    const uint32_t R,
    const uint32_t G,
    RASTER_STATE::GeometryState geomState,
    const dim3 tile_grid)
{
    const int idx = cg::this_grid().thread_rank();
    if (idx >= (int)G)
        return;
    const uint32_t prim_id = R + idx;
    uint2 tile_min, tile_max;
    getBboxTileRectXYXY(geomState.bboxes[prim_id], tile_min, tile_max, tile_grid);
    const int tiles_touched = (1 + (int)tile_max.y - (int)tile_min.y) * (1 + (int)tile_max.x - (int)tile_min.x);
    const int quadrant_touched = __popc(geomState.cam_quadrant_bitsets[prim_id]);
    geomState.n_duplicates[prim_id] = max(tiles_touched * quadrant_touched, 0);
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor>
rasterize_preprocess(
    const int image_width, const int image_height,
    const float tan_fovx, const float tan_fovy,
    const float cx, const float cy,
    const torch::Tensor& w2c_matrix,
    const torch::Tensor& c2w_matrix,
    const float near,
    const torch::Tensor& octree_paths,
    const torch::Tensor& octlevels,
    const torch::Tensor& vox_centers,
    const torch::Tensor& vox_lengths,
    const bool debug)
{
    if (vox_centers.ndimension() != 2 || vox_centers.size(1) != 3)
        AT_ERROR("vox_centers must have dimensions (num_points, 3)");

    const int P = vox_centers.size(0);
    auto t_opt_byte = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    auto t_opt_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    torch::Tensor geomBuffer = torch::empty({0}, t_opt_byte);
    torch::Tensor out_n_duplicates = torch::full({P}, 0, t_opt_int32);
    if (P == 0)
        return std::make_tuple(out_n_duplicates, geomBuffer);

    size_t chunk_size = RASTER_STATE::required<RASTER_STATE::GeometryState>(P);
    geomBuffer.resize_({(long long)chunk_size});
    char* chunkptr = reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr());
    RASTER_STATE::GeometryState geomState = RASTER_STATE::GeometryState::fromChunk(chunkptr, P);

    init_geom_state_cuda<<<1, 1>>>(geomState, P);
    CHECK_CUDA(debug);

    dim3 tile_grid((image_width + BLOCK_X - 1) / BLOCK_X, (image_height + BLOCK_Y - 1) / BLOCK_Y, 1);
    const float focal_x = 0.5f * image_width / tan_fovx;
    const float focal_y = 0.5f * image_height / tan_fovy;

    preprocess_leaf_cuda<<<(P + 255) / 256, 256>>>(
        P,
        image_width, image_height,
        tan_fovx, tan_fovy,
        focal_x, focal_y,
        cx, cy,
        w2c_matrix.contiguous().data_ptr<float>(),
        c2w_matrix.contiguous().data_ptr<float>(),
        near,
        octree_paths.contiguous().data_ptr<int64_t>(),
        octlevels.numel() == 0 ? nullptr : octlevels.contiguous().data_ptr<int8_t>(),
        (float3*)(vox_centers.contiguous().data_ptr<float>()),
        vox_lengths.contiguous().data_ptr<float>(),
        out_n_duplicates.contiguous().data_ptr<int>(),
        geomState,
        tile_grid);
    CHECK_CUDA(debug);

    uint32_t resolved_count = 0;
    uint32_t unresolved_count = 0;
    cudaMemcpy(&resolved_count, geomState.resolved_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&unresolved_count, geomState.unresolved_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    CHECK_CUDA(debug);

    assemble_resolved_primitives_cuda<<<(resolved_count + 255) / 256, 256>>>(
        resolved_count,
        out_n_duplicates.contiguous().data_ptr<int>(),
        octree_paths.contiguous().data_ptr<int64_t>(),
        geomState.leaf_bboxes,
        geomState.leaf_cam_quadrant_bitsets,
        geomState);
    CHECK_CUDA(debug);

    uint32_t proxy_count = 0;
    if (unresolved_count > 0)
    {
        cub::DeviceRadixSort::SortPairs(
            geomState.sorting_temp_space,
            geomState.sort_size,
            geomState.unresolved_bucket_keys,
            geomState.unresolved_bucket_keys_sorted,
            geomState.unresolved_member_ids,
            geomState.unresolved_member_ids_sorted,
            unresolved_count);
        CHECK_CUDA(debug);

        mark_proxy_group_starts_cuda<<<(unresolved_count + 255) / 256, 256>>>(
            unresolved_count,
            geomState.unresolved_bucket_keys_sorted,
            geomState.unresolved_group_start_flags);
        CHECK_CUDA(debug);

        cub::DeviceScan::InclusiveSum(
            geomState.scanning_temp_space,
            geomState.scan_size,
            geomState.unresolved_group_start_flags,
            geomState.unresolved_group_ids,
            unresolved_count);
        CHECK_CUDA(debug);

        cudaMemcpy(
            &proxy_count,
            geomState.unresolved_group_ids + unresolved_count - 1,
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost);
        CHECK_CUDA(debug);

        init_proxy_primitives_cuda<<<(proxy_count + 255) / 256, 256>>>(
            resolved_count,
            proxy_count,
            geomState);
        CHECK_CUDA(debug);

        assemble_proxy_primitives_cuda<<<(unresolved_count + 255) / 256, 256>>>(
            resolved_count,
            unresolved_count,
            geomState.unresolved_group_start_flags,
            geomState.unresolved_group_ids,
            geomState.unresolved_bucket_keys_sorted,
            geomState.unresolved_member_ids_sorted,
            geomState.leaf_bboxes,
            geomState.leaf_cam_quadrant_bitsets,
            geomState);
        CHECK_CUDA(debug);

        finalize_proxy_duplicates_cuda<<<(proxy_count + 255) / 256, 256>>>(
            resolved_count,
            proxy_count,
            geomState,
            tile_grid);
        CHECK_CUDA(debug);
    }

    const uint32_t primitive_count = resolved_count + proxy_count;
    cudaMemcpy(geomState.primitive_count, &primitive_count, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(geomState.proxy_count, &proxy_count, sizeof(uint32_t), cudaMemcpyHostToDevice);
    CHECK_CUDA(debug);

    return std::make_tuple(out_n_duplicates, geomBuffer);
}

} // namespace PREPROCESS
