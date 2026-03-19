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

#include "backward.h"
#include "raster_state.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace BACKWARD {

// CUDA backward pass of sparse voxel rendering.
template <bool need_depth, bool need_distortion, bool need_normal,
          int n_samp>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ vox_list,
    const int W, const int H,
    const float tan_fovx, const float tan_fovy,
    const float cx, const float cy,
    const float* __restrict__ c2w_matrix,
    const float bg_color,

    const int4* __restrict__ bboxes,
    const int32_t* __restrict__ primitive_ref_ids,
    const uint8_t* __restrict__ primitive_is_proxy,
    const uint32_t* __restrict__ primitive_member_starts,
    const uint32_t* __restrict__ primitive_member_ends,
    const uint32_t* __restrict__ unresolved_member_ids,
    const float3* __restrict__ vox_centers,
    const float* __restrict__ vox_lengths,
    const float* __restrict__ geos,
    const float3* __restrict__ rgbs,

    const float* __restrict__ out_T,
    const uint32_t* __restrict__ tile_last,
    const uint32_t* __restrict__ n_contrib,

    const float* __restrict__ dL_dout_color,
    const float* __restrict__ dL_dout_depth,
    const float* __restrict__ dL_dout_normal,
    const float* __restrict__ dL_dout_T,

    const float lambda_R_concen,
    const float* gt_color,
    const float lambda_ascending,
    const float lambda_dist,
    const float* out_D,
    const float* out_N,

    float* dL_dvox)
{
    // We rasterize again. Compute necessary block info.
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    int thread_id = block.thread_rank();
    int tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };

    uint2 pix;
    uint32_t pix_id;
    float2 pixf;
    if (BLOCK_X % 8 == 0 && BLOCK_Y % 4 == 0)
    {
        // Pack the warp threads into a 4x8 macro blocks.
        // It could reduce idle warp threads as the voxels to render
        // are more coherent in 4x8 than 2x16 rectangle.
        int macro_x_num = BLOCK_X / 8;
        int macro_id = thread_id / 32;
        int macro_xid = macro_id % macro_x_num;
        int macro_yid = macro_id / macro_x_num;
        int micro_id = thread_id % 32;
        int micro_xid = micro_id % 8;
        int micro_yid = micro_id / 8;
        pix = { pix_min.x + macro_xid * 8 + micro_xid, pix_min.y + macro_yid * 4 + micro_yid};
        pix_id = W * pix.y + pix.x;
        pixf = { (float)pix.x, (float)pix.y };
    }
    else
    {
        pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
        pix_id = W * pix.y + pix.x;
        pixf = { (float)pix.x, (float)pix.y };
    }

    // Compute camera info.
    const float3 cam_rd = compute_ray_d(pixf, W, H, tan_fovx, tan_fovy, cx, cy);
    const float rd_norm = sqrtf(dot(cam_rd, cam_rd));
    const float rd_norm_inv = 1.f / rd_norm;
    const float3 ro = last_col_3x4(c2w_matrix);
    const float3 cam_forward = third_col_3x4(c2w_matrix);
    const float3 rd_raw = rotate_3x4(c2w_matrix, cam_rd);
    const float3 rd = rd_raw * rd_norm_inv;
    const float3 rd_inv = {1.f/ rd.x, 1.f / rd.y, 1.f / rd.z};
    uint32_t pix_quad_id = compute_ray_quadrant_id(rd);

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = (pix.x < W) && (pix.y < H);
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    const uint2 range_raw = ranges[tile_id];
    const uint2 range = {range_raw.x, tile_last[tile_id]};
    if (range.y > range_raw.y)
    {
        // TODO: remove sanity check.
        printf("range.y > range_raw.y !???");
        __trap();
    }
    if (range.x > range.y)
    {
        // TODO: remove sanity check.
        printf("range.x > range.y !???");
        __trap();
    }

    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int toDo = range.y - range.x;

    // Allocate storage for batches of collectively fetched data.
    // 3090Ti shared memory per-block statistic:
    //   total shared memory      = 49152 bytes
    //   shared memory per-thread = 49152/BLOCK_SIZE = 192 bytes
    //                            = 48 int or float
    __shared__ int collected_prim_id[BLOCK_SIZE];
    __shared__ int collected_quad_id[BLOCK_SIZE];
    __shared__ int4 collected_bbox[BLOCK_SIZE];
    __shared__ int collected_ref_id[BLOCK_SIZE];
    __shared__ uint8_t collected_is_proxy[BLOCK_SIZE];
    __shared__ uint32_t collected_member_start[BLOCK_SIZE];
    __shared__ uint32_t collected_member_end[BLOCK_SIZE];

    // In the forward, we stored the final value for T, the
    // product of all (1 - alpha) factors.
    const float T_final = inside ? out_T[pix_id] : 0.f;
    float T = T_final;

    // We start from the back.
    // The last contributing voxel ID of each pixel is known from the forward.
    uint32_t contributor = toDo;
    const int last_contributor = inside ? n_contrib[pix_id] : 0;

    // Init gradient from the last computation node.
    float3 dL_dpix;
    float dL_dD;
    float3 dL_dN;
    float last_dL_dT;
    if (inside)
    {
        dL_dpix.x = dL_dout_color[0 * H * W + pix_id];
        dL_dpix.y = dL_dout_color[1 * H * W + pix_id];
        dL_dpix.z = dL_dout_color[2 * H * W + pix_id];
        const float dL_dpix_T = dL_dout_T[pix_id];
        last_dL_dT = dL_dpix_T + bg_color * (dL_dpix.x + dL_dpix.y + dL_dpix.z);

        dL_dD = dL_dout_depth[pix_id] * rd_norm_inv;
        dL_dN.x = dL_dout_normal[0 * H * W + pix_id];
        dL_dN.y = dL_dout_normal[1 * H * W + pix_id];
        dL_dN.z = dL_dout_normal[2 * H * W + pix_id];
    }

    // Compute regularization weights.
    const float WH_inv = 1.f / ((float)(W * H));
    const float weight_R_concen = lambda_R_concen * WH_inv;
    const float weight_ascending = lambda_ascending * WH_inv;
    float3 gt_pix;
    if (lambda_R_concen > 0 && inside)
    {
        gt_pix.x = gt_color[0 * H * W + pix_id];
        gt_pix.y = gt_color[1 * H * W + pix_id];
        gt_pix.z = gt_color[2 * H * W + pix_id];
    }

    float3 pix_n;
    if (need_normal && inside)
    {
        pix_n.x = out_N[0 * H * W + pix_id];
        pix_n.y = out_N[1 * H * W + pix_id];
        pix_n.z = out_N[2 * H * W + pix_id];
        pix_n = safe_rnorm(pix_n) * pix_n;
    }

    const float weight_dist = lambda_dist * WH_inv;
    float prefix_wm, suffix_wm, prefix_w, suffix_w;
    if (lambda_dist > 0 && inside)
    {
        // See DVGOv2 for formula.
        prefix_wm = out_D[H * W + pix_id];
        suffix_wm = 0.f;
        prefix_w = 1.f - T_final;
        suffix_w = 0.f;
    }

    // For seam regularizaiton.
    int j_lst[BLOCK_SIZE];

    // Traverse all voxels.
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        // Load auxiliary data into shared memory, start in the BACK
        // and load them in revers order.
        block.sync();
        const int progress = i * BLOCK_SIZE + thread_id;
        if (range.x + progress < range.y)
        {
            uint32_t order_val = vox_list[range.y - progress - 1];
            uint32_t prim_id = decode_order_val_4_vox_id(order_val);
            uint32_t quad_id = decode_order_val_4_quadrant_id(order_val);
            collected_prim_id[thread_id] = prim_id;
            collected_quad_id[thread_id] = quad_id;
            collected_bbox[thread_id] = bboxes[prim_id];
            collected_ref_id[thread_id] = primitive_ref_ids[prim_id];
            collected_is_proxy[thread_id] = primitive_is_proxy[prim_id];
            collected_member_start[thread_id] = primitive_member_starts[prim_id];
            collected_member_end[thread_id] = primitive_member_ends[prim_id];
        }
        block.sync();

        // Iterate over voxels.
        const int end_j = min(BLOCK_SIZE, toDo);
        int j_lst_top = -1;
        for (int j = 0; !done && j < end_j; j++)
        {
            // Keep track of current voxel ID. Skip, if this one
            // is behind the last contributor for this pixel.
            contributor--;
            if (contributor >= last_contributor)
                continue;

            /**************************
            Below, we first compute blending values, as in the forward.
            **************************/

            // Check if the pixel in the projected bbox region.
            // Check if the quadrant id match the pixel.
            if (!pix_in_bbox_xyxy(pix, collected_bbox[j]) || pix_quad_id != collected_quad_id[j])
                continue;

            // Compute ray aabb intersection.
            const bool unresolved = collected_is_proxy[j] != 0;
            if (!unresolved)
            {
                const int leaf_id = collected_ref_id[j];
                const float3 vox_c = vox_centers[leaf_id];
                const float vox_l = vox_lengths[leaf_id];
                const float2 ab = ray_aabb(vox_c, vox_l, ro, rd_inv);
                const float a = ab.x;
                const float b = ab.y;
                if (a > b)
                    continue;  // Skip if no intersection.
            }

            j_lst_top += 1;
            j_lst[j_lst_top] = j;
        }

        for (int jj = 0; !done && jj <= j_lst_top; jj++)
        {
            int j = j_lst[jj];
            const bool unresolved = collected_is_proxy[j] != 0;
            const int leaf_id = collected_ref_id[j];
            float3 vox_c = {0.f, 0.f, 0.f};
            float vox_l = 0.f;
            float a = 0.f, b = 0.f;
            float geo_params[8] = {0.f};
            float dL_dgeo_params[8] = {0.f};
            float dI_dgeo_params[8] = {0.f};
            float each_dI_dgeo_params[n_samp][8];
            float local_alphas[n_samp];
            float interp_w[8];
            float vox_l_inv = 1.f;
            float3 c = {0.f, 0.f, 0.f};
            float3 surf_n = {0.f, 0.f, 0.f};
            float z_proxy = 0.f;
            float vol_int = 0.f;
            float tau_sum = 0.f;

            if (!unresolved)
            {
                vox_c = vox_centers[leaf_id];
                vox_l = vox_lengths[leaf_id];
                vox_l_inv = 1.f / vox_l;
                const float2 ab = ray_aabb(vox_c, vox_l, ro, rd_inv);
                a = ab.x;
                b = ab.y;
                for (int k = 0; k < 8; ++k)
                    geo_params[k] = geos[leaf_id * 8 + k];
                c = rgbs[leaf_id];

                const float step_sz = (b - a) * (1.f / n_samp);
                const float3 step = step_sz * rd;
                float3 pt = ro + (a + 0.5f * step_sz) * rd;
                float3 qt = (pt - (vox_c - 0.5f * vox_l)) * vox_l_inv;
                const float3 qt_step = step * vox_l_inv;
                #pragma unroll
                for (int k = 0; k < n_samp; ++k, qt = qt + qt_step)
                {
                    tri_interp_weight(qt, interp_w);
                    float d = 0.f;
                    for (int iii = 0; iii < 8; ++iii)
                        d += geo_params[iii] * interp_w[iii];
                    const float local_vol_int = STEP_SZ_SCALE * step_sz * exp_linear_11(d);
                    vol_int += local_vol_int;
                    if (need_depth && n_samp > 1)
                        local_alphas[k] = min(MAX_ALPHA, 1.f - expf(-local_vol_int));
                    const float dd_dd = STEP_SZ_SCALE * step_sz * exp_linear_11_bw(d);
                    for (int iii = 0; iii < 8; ++iii)
                    {
                        float tmp = dd_dd * interp_w[iii];
                        dI_dgeo_params[iii] += tmp;
                        if (need_depth && n_samp > 1)
                            each_dI_dgeo_params[k][iii] = tmp;
                    }
                }

                const float lin_nx = (
                    (geo_params[0b100] + geo_params[0b101] + geo_params[0b110] + geo_params[0b111]) -
                    (geo_params[0b000] + geo_params[0b001] + geo_params[0b010] + geo_params[0b011]));
                const float lin_ny = (
                    (geo_params[0b010] + geo_params[0b011] + geo_params[0b110] + geo_params[0b111]) -
                    (geo_params[0b000] + geo_params[0b001] + geo_params[0b100] + geo_params[0b101]));
                const float lin_nz = (
                    (geo_params[0b001] + geo_params[0b011] + geo_params[0b101] + geo_params[0b111]) -
                    (geo_params[0b000] + geo_params[0b010] + geo_params[0b100] + geo_params[0b110]));
                const float3 lin_n = make_float3(lin_nx, lin_ny, lin_nz);
                surf_n = safe_rnorm(lin_n) * lin_n;
            }
            else
            {
                const uint32_t member_start = collected_member_start[j];
                const uint32_t member_end = collected_member_end[j];
                float depth_weighted = 0.f;
                float3 rgb_acc = {0.f, 0.f, 0.f};
                float3 nsum = {0.f, 0.f, 0.f};
                for (uint32_t mid = member_start; mid < member_end; ++mid)
                {
                    const uint32_t mid_leaf = unresolved_member_ids[mid];
                    float mid_geo[8];
                    for (int k = 0; k < 8; ++k)
                        mid_geo[k] = geos[mid_leaf * 8 + k];
                    const float z_i = dot(vox_centers[mid_leaf] - ro, cam_forward);
                    const float pix_area = pixel_footprint_area_at_depth(z_i, W, H, tan_fovx, tan_fovy);
                    const float tau_i = unresolved_tau_proxy(mid_geo, vox_lengths[mid_leaf], pix_area);
                    tau_sum += tau_i;
                    depth_weighted += tau_i * z_i;
                    rgb_acc = rgb_acc + tau_i * rgbs[mid_leaf];

                    const float lin_nx = (
                        (mid_geo[0b100] + mid_geo[0b101] + mid_geo[0b110] + mid_geo[0b111]) -
                        (mid_geo[0b000] + mid_geo[0b001] + mid_geo[0b010] + mid_geo[0b011]));
                    const float lin_ny = (
                        (mid_geo[0b010] + mid_geo[0b011] + mid_geo[0b110] + mid_geo[0b111]) -
                        (mid_geo[0b000] + mid_geo[0b001] + mid_geo[0b100] + mid_geo[0b101]));
                    const float lin_nz = (
                        (mid_geo[0b001] + mid_geo[0b011] + mid_geo[0b101] + mid_geo[0b111]) -
                        (mid_geo[0b000] + mid_geo[0b010] + mid_geo[0b100] + mid_geo[0b110]));
                    const float3 lin_n = make_float3(lin_nx, lin_ny, lin_nz);
                    nsum = nsum + tau_i * safe_rnorm(lin_n) * lin_n;
                }
                vol_int = tau_sum;
                if (tau_sum > 0.f)
                {
                    c = (1.f / tau_sum) * rgb_acc;
                    z_proxy = depth_weighted / tau_sum;
                    surf_n = safe_rnorm(nsum) * nsum;
                }
            }

            const float exp_neg_vol_int = expf(-vol_int);
            const float alpha = min(MAX_ALPHA, 1.f - exp_neg_vol_int);
            if (alpha < MIN_ALPHA)
                continue;

            T = T / (1.f - alpha);
            const float pt_w = alpha * T;
            float dL_dpt_w = dot(dL_dpix, c);

            if (need_distortion)
            {
                const float now_m = unresolved ? depth_contracted(z_proxy) : 0.5f * (depth_contracted(a) + depth_contracted(b));
                const float now_wm = now_m * pt_w;
                prefix_wm -= now_wm;
                prefix_w -= pt_w;
                const float dist_grad_uni = unresolved ? 0.f : 0.6666666f * pt_w * (depth_contracted(b) - depth_contracted(a));
                const float dist_grad_bi = 2.f * (now_m * (prefix_w - suffix_w) - (prefix_wm - suffix_wm));
                dL_dpt_w += weight_dist * (dist_grad_uni + dist_grad_bi);
                suffix_wm += now_wm;
                suffix_w += pt_w;
            }

            float3 dL_dsurf_n = {0.f, 0.f, 0.f};
            if (need_normal)
            {
                dL_dpt_w += dot(dL_dN, surf_n);
                dL_dsurf_n = pt_w * dL_dN;
            }

            const float dL_dalpha = T * (dL_dpt_w - last_dL_dT);
            last_dL_dT += alpha * (dL_dpt_w - last_dL_dT);
            float dL_dI = dL_dalpha * exp_neg_vol_int;

            float dL_ddepthbar = 0.f;
            if (need_depth)
            {
                const float dbar = unresolved ? z_proxy :
                    ((n_samp == 3) ? (local_alphas[0]*(a + 0.5f*(b-a)/3.f) + (1.f-local_alphas[0])*local_alphas[1]*(a + 1.5f*(b-a)/3.f) + (1.f-local_alphas[0])*(1.f-local_alphas[1])*local_alphas[2]*(a + 2.5f*(b-a)/3.f)) :
                    ((n_samp == 2) ? (local_alphas[0]*(a + 0.25f*(b-a)*2.f) + (1.f-local_alphas[0])*local_alphas[1]*(a + 0.75f*(b-a)*2.f)) :
                    (unresolved ? z_proxy : 0.5f * (a + b))));
                last_dL_dT += dL_dD * alpha * dbar;
                dL_dI += dL_dD * T * dbar * (1.f - alpha);
                dL_ddepthbar = dL_dD * T * alpha;
            }

            if (!unresolved)
            {
                for (int iii = 0; iii < 8; ++iii)
                    dL_dgeo_params[iii] += dL_dI * dI_dgeo_params[iii];

                if (need_normal)
                {
                    const float lin_nx = (
                        (geo_params[0b100] + geo_params[0b101] + geo_params[0b110] + geo_params[0b111]) -
                        (geo_params[0b000] + geo_params[0b001] + geo_params[0b010] + geo_params[0b011]));
                    const float lin_ny = (
                        (geo_params[0b010] + geo_params[0b011] + geo_params[0b110] + geo_params[0b111]) -
                        (geo_params[0b000] + geo_params[0b001] + geo_params[0b100] + geo_params[0b101]));
                    const float lin_nz = (
                        (geo_params[0b001] + geo_params[0b011] + geo_params[0b101] + geo_params[0b111]) -
                        (geo_params[0b000] + geo_params[0b010] + geo_params[0b100] + geo_params[0b110]));
                    const float3 lin_n = make_float3(lin_nx, lin_ny, lin_nz);
                    const float r_lin = safe_rnorm(lin_n);
                    float3 dL_dlin_n = r_lin * (dL_dsurf_n - dot(dL_dsurf_n, lin_n) * r_lin * surf_n);
                    dL_dgeo_params[0b000] += -dL_dlin_n.x - dL_dlin_n.y - dL_dlin_n.z;
                    dL_dgeo_params[0b001] += -dL_dlin_n.x - dL_dlin_n.y + dL_dlin_n.z;
                    dL_dgeo_params[0b010] += -dL_dlin_n.x + dL_dlin_n.y - dL_dlin_n.z;
                    dL_dgeo_params[0b011] += -dL_dlin_n.x + dL_dlin_n.y + dL_dlin_n.z;
                    dL_dgeo_params[0b100] += +dL_dlin_n.x - dL_dlin_n.y - dL_dlin_n.z;
                    dL_dgeo_params[0b101] += +dL_dlin_n.x - dL_dlin_n.y + dL_dlin_n.z;
                    dL_dgeo_params[0b110] += +dL_dlin_n.x + dL_dlin_n.y - dL_dlin_n.z;
                    dL_dgeo_params[0b111] += +dL_dlin_n.x + dL_dlin_n.y + dL_dlin_n.z;
                }

                float dL_drgb[3] = {pt_w * dL_dpix.x, pt_w * dL_dpix.y, pt_w * dL_dpix.z};
                if (lambda_R_concen > 0)
                {
                    const float3 grad_R_concen = weight_R_concen * pt_w * 2.0f * (c - gt_pix);
                    dL_drgb[0] += grad_R_concen.x;
                    dL_drgb[1] += grad_R_concen.y;
                    dL_drgb[2] += grad_R_concen.z;
                }

                for (int iii = 0; iii < 8; ++iii)
                    atomicAdd(dL_dvox + leaf_id * 12 + iii, dL_dgeo_params[iii]);
                atomicAdd(dL_dvox + leaf_id * 12 + 8, dL_drgb[0]);
                atomicAdd(dL_dvox + leaf_id * 12 + 9, dL_drgb[1]);
                atomicAdd(dL_dvox + leaf_id * 12 + 10, dL_drgb[2]);
                atomicAdd(dL_dvox + leaf_id * 12 + 11, fabs(dL_dalpha * alpha));
            }
            else
            {
                const uint32_t member_start = collected_member_start[j];
                const uint32_t member_end = collected_member_end[j];
                const float3 dL_dcolorbar =
                    make_float3(pt_w * dL_dpix.x, pt_w * dL_dpix.y, pt_w * dL_dpix.z) +
                    (lambda_R_concen > 0 ? weight_R_concen * pt_w * 2.0f * (c - gt_pix) : make_float3(0.f, 0.f, 0.f));

                float3 nsum = {0.f, 0.f, 0.f};
                if (need_normal)
                {
                    for (uint32_t mid = member_start; mid < member_end; ++mid)
                    {
                        const uint32_t mid_leaf = unresolved_member_ids[mid];
                        float mid_geo[8];
                        for (int k = 0; k < 8; ++k)
                            mid_geo[k] = geos[mid_leaf * 8 + k];
                        const float z_i = dot(vox_centers[mid_leaf] - ro, cam_forward);
                        const float pix_area = pixel_footprint_area_at_depth(z_i, W, H, tan_fovx, tan_fovy);
                        const float tau_i = unresolved_tau_proxy(mid_geo, vox_lengths[mid_leaf], pix_area);
                        const float lin_nx = (
                            (mid_geo[0b100] + mid_geo[0b101] + mid_geo[0b110] + mid_geo[0b111]) -
                            (mid_geo[0b000] + mid_geo[0b001] + mid_geo[0b010] + mid_geo[0b011]));
                        const float lin_ny = (
                            (mid_geo[0b010] + mid_geo[0b011] + mid_geo[0b110] + mid_geo[0b111]) -
                            (mid_geo[0b000] + mid_geo[0b001] + mid_geo[0b100] + mid_geo[0b101]));
                        const float lin_nz = (
                            (mid_geo[0b001] + mid_geo[0b011] + mid_geo[0b101] + mid_geo[0b111]) -
                            (mid_geo[0b000] + mid_geo[0b010] + mid_geo[0b100] + mid_geo[0b110]));
                        const float3 lin_n = make_float3(lin_nx, lin_ny, lin_nz);
                        nsum = nsum + tau_i * safe_rnorm(lin_n) * lin_n;
                    }
                }
                const float r_nsum = safe_rnorm(nsum);
                const float3 dL_dnsum = need_normal ? r_nsum * (dL_dsurf_n - dot(dL_dsurf_n, nsum) * r_nsum * surf_n) : make_float3(0.f, 0.f, 0.f);

                for (uint32_t mid = member_start; mid < member_end; ++mid)
                {
                    const uint32_t mid_leaf = unresolved_member_ids[mid];
                    float mid_geo[8];
                    for (int k = 0; k < 8; ++k)
                        mid_geo[k] = geos[mid_leaf * 8 + k];
                    const float z_i = dot(vox_centers[mid_leaf] - ro, cam_forward);
                    const float pix_area = pixel_footprint_area_at_depth(z_i, W, H, tan_fovx, tan_fovy);
                    const float tau_i = unresolved_tau_proxy(mid_geo, vox_lengths[mid_leaf], pix_area);
                    float dL_dtau_i = dL_dI;
                    dL_dtau_i += dot(dL_dcolorbar, rgbs[mid_leaf] - c) / max(tau_sum, 1e-12f);
                    dL_dtau_i += dL_ddepthbar * (z_i - z_proxy) / max(tau_sum, 1e-12f);

                    const float lin_nx = (
                        (mid_geo[0b100] + mid_geo[0b101] + mid_geo[0b110] + mid_geo[0b111]) -
                        (mid_geo[0b000] + mid_geo[0b001] + mid_geo[0b010] + mid_geo[0b011]));
                    const float lin_ny = (
                        (mid_geo[0b010] + mid_geo[0b011] + mid_geo[0b110] + mid_geo[0b111]) -
                        (mid_geo[0b000] + mid_geo[0b001] + mid_geo[0b100] + mid_geo[0b101]));
                    const float lin_nz = (
                        (mid_geo[0b001] + mid_geo[0b011] + mid_geo[0b101] + mid_geo[0b111]) -
                        (mid_geo[0b000] + mid_geo[0b010] + mid_geo[0b100] + mid_geo[0b110]));
                    const float3 lin_n = make_float3(lin_nx, lin_ny, lin_nz);
                    const float r_lin = safe_rnorm(lin_n);
                    const float3 surf_mid_n = r_lin * lin_n;
                    dL_dtau_i += dot(dL_dnsum, surf_mid_n);
                    const float3 dL_dsurf_mid_n = tau_i * dL_dnsum;
                    const float3 dL_dlin_n = r_lin * (dL_dsurf_mid_n - dot(dL_dsurf_mid_n, lin_n) * r_lin * surf_mid_n);

                    const float dI_dgeo = unresolved_tau_proxy_bw(mid_geo, vox_lengths[mid_leaf], pix_area);
                    float local_grad[8];
                    for (int iii = 0; iii < 8; ++iii)
                        local_grad[iii] = dL_dtau_i * dI_dgeo;
                    local_grad[0b000] += -dL_dlin_n.x - dL_dlin_n.y - dL_dlin_n.z;
                    local_grad[0b001] += -dL_dlin_n.x - dL_dlin_n.y + dL_dlin_n.z;
                    local_grad[0b010] += -dL_dlin_n.x + dL_dlin_n.y - dL_dlin_n.z;
                    local_grad[0b011] += -dL_dlin_n.x + dL_dlin_n.y + dL_dlin_n.z;
                    local_grad[0b100] += +dL_dlin_n.x - dL_dlin_n.y - dL_dlin_n.z;
                    local_grad[0b101] += +dL_dlin_n.x - dL_dlin_n.y + dL_dlin_n.z;
                    local_grad[0b110] += +dL_dlin_n.x + dL_dlin_n.y - dL_dlin_n.z;
                    local_grad[0b111] += +dL_dlin_n.x + dL_dlin_n.y + dL_dlin_n.z;

                    for (int iii = 0; iii < 8; ++iii)
                        atomicAdd(dL_dvox + mid_leaf * 12 + iii, local_grad[iii]);
                    atomicAdd(dL_dvox + mid_leaf * 12 + 8, (tau_i / max(tau_sum, 1e-12f)) * dL_dcolorbar.x);
                    atomicAdd(dL_dvox + mid_leaf * 12 + 9, (tau_i / max(tau_sum, 1e-12f)) * dL_dcolorbar.y);
                    atomicAdd(dL_dvox + mid_leaf * 12 + 10, (tau_i / max(tau_sum, 1e-12f)) * dL_dcolorbar.z);
                    atomicAdd(dL_dvox + mid_leaf * 12 + 11, (tau_i / max(tau_sum, 1e-12f)) * fabs(dL_dalpha * alpha));
                }
            }
        }
    }
}

#ifndef BwRendFunc
// Dirty trick. The argument name must be aligned with BACKWARD::render.
#define BwRendFunc(...) \
    ( \
        (need_depth && need_distortion && need_normal) ?\
            renderCUDA<true, true, true, __VA_ARGS__> :\
        (need_depth && need_distortion && !need_normal) ?\
            renderCUDA<true, true, false, __VA_ARGS__> :\
        (need_depth && !need_distortion && need_normal) ?\
            renderCUDA<true, false, true, __VA_ARGS__> :\
        (need_depth && !need_distortion && !need_normal) ?\
            renderCUDA<true, false, false, __VA_ARGS__> :\
        (!need_depth && need_distortion && need_normal) ?\
            renderCUDA<false, true, true, __VA_ARGS__> :\
        (!need_depth && need_distortion && !need_normal) ?\
            renderCUDA<false, true, false, __VA_ARGS__> :\
        (!need_depth && !need_distortion && need_normal) ?\
            renderCUDA<false, false, true, __VA_ARGS__> :\
        (!need_depth && !need_distortion && !need_normal) ?\
            renderCUDA<false, false, false, __VA_ARGS__> :\
        (need_depth && need_distortion && need_normal) ?\
            renderCUDA<true, true, true, __VA_ARGS__> :\
        (need_depth && need_distortion && !need_normal) ?\
            renderCUDA<true, true, false, __VA_ARGS__> :\
        (need_depth && !need_distortion && need_normal) ?\
            renderCUDA<true, false, true, __VA_ARGS__> :\
        (need_depth && !need_distortion && !need_normal) ?\
            renderCUDA<true, false, false, __VA_ARGS__> :\
        (!need_depth && need_distortion && need_normal) ?\
            renderCUDA<false, true, true, __VA_ARGS__> :\
        (!need_depth && need_distortion && !need_normal) ?\
            renderCUDA<false, true, false, __VA_ARGS__> :\
        (!need_depth && !need_distortion && need_normal) ?\
            renderCUDA<false, false, true, __VA_ARGS__> :\
            renderCUDA<false, false, false, __VA_ARGS__> \
    )
#endif

// Lowest-level C interface for launching the CUDA.
void render(
    const dim3 tile_grid, const dim3 block,
    const uint2* ranges,
    const uint32_t* vox_list,
    const int n_samp_per_vox,
    const int W, const int H,
    const float tan_fovx, const float tan_fovy,
    const float cx, const float cy,
    const float* c2w_matrix,
    const float bg_color,

    const int4* bboxes,
    const int32_t* primitive_ref_ids,
    const uint8_t* primitive_is_proxy,
    const uint32_t* primitive_member_starts,
    const uint32_t* primitive_member_ends,
    const uint32_t* unresolved_member_ids,
    const float3* vox_centers,
    const float* vox_lengths,
    const float* geos,
    const float3* rgbs,

    const float* out_T,
    const uint32_t* tile_last,
    const uint32_t* n_contrib,

    const float* dL_dout_color,
    const float* dL_dout_depth,
    const float* dL_dout_normal,
    const float* dL_dout_T,

    const float lambda_R_concen,
    const float* gt_color,
    const float lambda_ascending,
    const float lambda_dist,
    const bool need_depth,
    const bool need_normal,
    const float* out_D,
    const float* out_N,

    float* dL_dvox)
{
    const bool need_distortion = (lambda_dist > 0);

    // The density_mode now is always EXP_LINEAR_11_MODE
    const auto kernel_func =
        (n_samp_per_vox == 3) ?
            BwRendFunc(3) :
        (n_samp_per_vox == 2) ?
            BwRendFunc(2) :
            BwRendFunc(1) ;

    kernel_func <<<tile_grid, block>>> (
        ranges,
        vox_list,
        W, H,
        tan_fovx, tan_fovy,
        cx, cy,
        c2w_matrix,
        bg_color,

        bboxes,
        primitive_ref_ids,
        primitive_is_proxy,
        primitive_member_starts,
        primitive_member_ends,
        unresolved_member_ids,
        vox_centers,
        vox_lengths,
        geos,
        rgbs,

        out_T,
        tile_last,
        n_contrib,

        dL_dout_color,
        dL_dout_depth,
        dL_dout_normal,
        dL_dout_T,

        lambda_R_concen,
        gt_color,
        lambda_ascending,
        lambda_dist,
        out_D,
        out_N,

        dL_dvox);
}


// Interface for python to run backward pass of voxel rasterization.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_voxels_backward(
    const int R,
    const int n_samp_per_vox,
    const int image_width, const int image_height,
    const float tan_fovx, const float tan_fovy,
    const float cx, const float cy,
    const torch::Tensor& w2c_matrix,
    const torch::Tensor& c2w_matrix,
    const float bg_color,

    const torch::Tensor& octree_paths,
    const torch::Tensor& vox_centers,
    const torch::Tensor& vox_lengths,
    const torch::Tensor& geos,
    const torch::Tensor& rgbs,

    const torch::Tensor& geomBuffer,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const torch::Tensor& out_T,

    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_depth,
    const torch::Tensor& dL_dout_normal,
    const torch::Tensor& dL_dout_T,

    const float lambda_R_concen,
    const torch::Tensor& gt_color,
    const float lambda_ascending,
    const float lambda_dist,
    const bool need_depth,
    const bool need_normal,
    const torch::Tensor& out_D,
    const torch::Tensor& out_N,

    const bool debug)
{
    if (vox_centers.ndimension() != 2 || vox_centers.size(1) != 3)
        AT_ERROR("vox_centers must have dimensions (num_points, 3)");

    const int P = vox_centers.size(0);

    if (P == 0)
    {
        torch::Tensor dL_dgeos = torch::empty({0});
        torch::Tensor dL_drgbs = torch::empty({0});
        torch::Tensor subdiv_p_bw = torch::empty({0});
        return std::make_tuple(dL_dgeos, dL_drgbs, subdiv_p_bw);
    }

    torch::Tensor dL_dvox = torch::zeros({P, geos.size(1)+3+1}, vox_centers.options());
    dim3 tile_grid((image_width + BLOCK_X - 1) / BLOCK_X, (image_height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Retrive raster state from pytorch tensor
    char* geomB_ptr = reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr());
    char* binningB_ptr = reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr());
    char* imageB_ptr = reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr());
    RASTER_STATE::GeometryState geomState = RASTER_STATE::GeometryState::fromChunk(
        geomB_ptr,
        P);
    const int total_key_bits = ORDER_RANK_BITS + required_bits_u32(tile_grid.x * tile_grid.y);
    const bool use_wide_sort_key = total_key_bits > SINGLE_SORT_KEY_BITS;
    RASTER_STATE::BinningState binningState = RASTER_STATE::BinningState::fromChunk(
        binningB_ptr,
        R,
        use_wide_sort_key);
    RASTER_STATE::ImageState imgState = RASTER_STATE::ImageState::fromChunk(
        imageB_ptr,
        image_width * image_height,
        tile_grid.x * tile_grid.y);

    // Compute loss gradients w.r.t. surface property and voxel color.
    render(
        tile_grid, block,
        imgState.ranges,
        binningState.vox_list,
        n_samp_per_vox,
        image_width, image_height,
        tan_fovx, tan_fovy,
        cx, cy,
        c2w_matrix.contiguous().data_ptr<float>(),
        bg_color,

        geomState.bboxes,
        geomState.primitive_ref_ids,
        geomState.primitive_is_proxy,
        geomState.primitive_member_starts,
        geomState.primitive_member_ends,
        geomState.unresolved_member_ids_sorted,
        (float3*)(vox_centers.contiguous().data_ptr<float>()),
        vox_lengths.contiguous().data_ptr<float>(),
        geos.contiguous().data_ptr<float>(),
        (float3*)(rgbs.contiguous().data_ptr<float>()),

        out_T.contiguous().data_ptr<float>(),
        imgState.tile_last,
        imgState.n_contrib,

        dL_dout_color.contiguous().data_ptr<float>(),
        dL_dout_depth.contiguous().data_ptr<float>(),
        dL_dout_normal.contiguous().data_ptr<float>(),
        dL_dout_T.contiguous().data_ptr<float>(),

        lambda_R_concen,
        gt_color.contiguous().data_ptr<float>(),
        lambda_ascending,
        lambda_dist,
        need_depth,
        need_normal,
        out_D.contiguous().data_ptr<float>(),
        out_N.contiguous().data_ptr<float>(),

        dL_dvox.contiguous().data_ptr<float>());
    CHECK_CUDA(debug);

    std::vector<torch::Tensor> gradient_lst = dL_dvox.split({geos.size(1), 3, 1}, 1);
    torch::Tensor dL_dgeos = gradient_lst[0].contiguous();
    torch::Tensor dL_drgbs = gradient_lst[1].contiguous();
    torch::Tensor subdiv_p_bw = gradient_lst[2].contiguous();

    return std::make_tuple(dL_dgeos, dL_drgbs, subdiv_p_bw);
}

}
