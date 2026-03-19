/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#ifndef RASTERIZER_AUXILIARY_H_INCLUDED
#define RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#ifdef __CUDACC__
#include <cuda/std/tuple>
#else
#include <tuple>
#endif
#include <cuda_runtime.h>
#include <cstdint>

struct SortKey128
{
    uint64_t hi;
    uint64_t lo;
};

struct SortKey128Decomposer
{
    __host__ __device__ auto operator()(SortKey128& key) const
    {
#ifdef __CUDACC__
        return cuda::std::tuple<uint64_t&, uint64_t&>(key.hi, key.lo);
#else
        return std::tuple<uint64_t&, uint64_t&>(key.hi, key.lo);
#endif
    }
};

static_assert(ORDER_RANK_BITS < 64, "Wide sort keys require the order-rank to fit in 64 bits.");

// Octant ordering tables
template<uint64_t id, int n>
struct repeat_3bits
{
    static constexpr uint64_t value = (repeat_3bits<id, n-1>::value << 3) | id;
};

template<uint64_t id>
struct repeat_3bits<id, 1>
{
    static constexpr uint64_t value = id;
};

__constant__ uint64_t order_tables[8] = {
    repeat_3bits<0ULL, MAX_NUM_LEVELS>::value,
    repeat_3bits<1ULL, MAX_NUM_LEVELS>::value,
    repeat_3bits<2ULL, MAX_NUM_LEVELS>::value,
    repeat_3bits<3ULL, MAX_NUM_LEVELS>::value,
    repeat_3bits<4ULL, MAX_NUM_LEVELS>::value,
    repeat_3bits<5ULL, MAX_NUM_LEVELS>::value,
    repeat_3bits<6ULL, MAX_NUM_LEVELS>::value,
    repeat_3bits<7ULL, MAX_NUM_LEVELS>::value
};

__forceinline__ __device__ uint64_t compute_order_rank(uint64_t octree_path, int quadrant_id)
{
    return octree_path ^ order_tables[quadrant_id];
}

__forceinline__ __device__ uint64_t encode_order_key(uint64_t tile_id, uint64_t order_rank)
{
    return (tile_id << ORDER_RANK_BITS) | order_rank;
}

__forceinline__ __host__ __device__ SortKey128 encode_order_key_wide(
    uint64_t tile_id,
    uint64_t order_rank)
{
    constexpr int tile_lo_bits = SINGLE_SORT_KEY_BITS - ORDER_RANK_BITS;
    SortKey128 key;
    key.lo = (tile_id << ORDER_RANK_BITS) | order_rank;
    key.hi = tile_id >> tile_lo_bits;
    return key;
}

__forceinline__ __host__ __device__ uint64_t decode_tile_id_from_key(uint64_t key)
{
    return key >> ORDER_RANK_BITS;
}

__forceinline__ __host__ __device__ uint64_t decode_tile_id_from_key(SortKey128 key)
{
    constexpr int tile_lo_bits = SINGLE_SORT_KEY_BITS - ORDER_RANK_BITS;
    return (key.hi << tile_lo_bits) | (key.lo >> ORDER_RANK_BITS);
}

__forceinline__ __host__ __device__ uint32_t required_bits_u32(uint32_t n)
{
    if (n <= 1)
        return 0;

    uint32_t bits = 0;
    n -= 1;
    while (n > 0)
    {
        n >>= 1;
        bits += 1;
    }
    return bits;
}

__forceinline__ __host__ __device__ float scalar_min(float a, float b)
{
    return (a < b) ? a : b;
}

__forceinline__ __host__ __device__ float scalar_max(float a, float b)
{
    return (a > b) ? a : b;
}

__forceinline__ __device__ uint32_t encode_order_val(uint32_t vox_id, uint32_t quadrant_id)
{
    return (((uint32_t)quadrant_id) << 29) | vox_id;
}

__forceinline__ __device__ uint32_t decode_order_val_4_vox_id(uint32_t val)
{
    return (val << 3) >> 3;
}

__forceinline__ __device__ uint32_t decode_order_val_4_quadrant_id(uint32_t val)
{
    return val >> 29;
}

__forceinline__ __device__ uint32_t compute_ray_quadrant_id(float3 rd)
{
    return ((rd.x < 0) << 2) | ((rd.y < 0) << 1) | (rd.z < 0);
}

__forceinline__ __device__ uint32_t compute_corner_quadrant_id(float3 corner, float3 ro)
{
    return ((corner.x < ro.x) << 2) | ((corner.y < ro.y) << 1) | (corner.z < ro.z);
}

// Utility functions.
__forceinline__ __device__ float3 compute_ray_d(float2 pixf, int W, int H, float tan_fovx, float tan_fovy, float cx, float cy)
{
    const float3 rd = {
        (pixf.x + 0.5f - cx) * 2.f * tan_fovx / (float)W,
        (pixf.y + 0.5f - cy) * 2.f * tan_fovy / (float)H,
        1.f
    };
    return rd;
}

__forceinline__ __device__ void getBboxTileRect(const uint2& bbox, uint2& tile_min, uint2& tile_max, dim3 grid)
{
    uint32_t xmin = (bbox.x >> 16);
    uint32_t ymin = (bbox.x << 16 >> 16);
    uint32_t xmax = (bbox.y >> 16);
    uint32_t ymax = (bbox.y << 16 >> 16);
    int max_grid_x = ((int)grid.x) - 1;
    int max_grid_y = ((int)grid.y) - 1;
    int tile_xmin = (int)(xmin / BLOCK_X);
    int tile_ymin = (int)(ymin / BLOCK_Y);
    int tile_xmax = (int)(xmax / BLOCK_X);
    int tile_ymax = (int)(ymax / BLOCK_Y);
    tile_min = {
        (uint32_t)(tile_xmin > max_grid_x ? max_grid_x : tile_xmin),
        (uint32_t)(tile_ymin > max_grid_y ? max_grid_y : tile_ymin)
    };
    tile_max = {
        (uint32_t)(tile_xmax > max_grid_x ? max_grid_x : tile_xmax),
        (uint32_t)(tile_ymax > max_grid_y ? max_grid_y : tile_ymax)
    };
}

__forceinline__ __device__ bool pix_in_bbox(const uint2& pix, const uint2& bbox)
{
    bool valid_xmin = pix.x >= (bbox.x >> 16);
    bool valid_ymin = pix.y >= (bbox.x << 16 >> 16);
    bool valid_xmax = pix.x <= (bbox.y >> 16);
    bool valid_ymax = pix.y <= (bbox.y << 16 >> 16);
    return valid_xmin && valid_ymin && valid_xmax && valid_ymax;
}

__forceinline__ __device__ void getBboxTileRectXYXY(const int4& bbox, uint2& tile_min, uint2& tile_max, dim3 grid)
{
    int max_grid_x = ((int)grid.x) - 1;
    int max_grid_y = ((int)grid.y) - 1;
    int tile_xmin = bbox.x / BLOCK_X;
    int tile_ymin = bbox.y / BLOCK_Y;
    int tile_xmax = bbox.z / BLOCK_X;
    int tile_ymax = bbox.w / BLOCK_Y;
    tile_min = {
        (uint32_t)(tile_xmin > max_grid_x ? max_grid_x : tile_xmin),
        (uint32_t)(tile_ymin > max_grid_y ? max_grid_y : tile_ymin)
    };
    tile_max = {
        (uint32_t)(tile_xmax > max_grid_x ? max_grid_x : tile_xmax),
        (uint32_t)(tile_ymax > max_grid_y ? max_grid_y : tile_ymax)
    };
}

__forceinline__ __device__ bool pix_in_bbox_xyxy(const uint2& pix, const int4& bbox)
{
    return pix.x >= (uint32_t)bbox.x &&
           pix.y >= (uint32_t)bbox.y &&
           pix.x <= (uint32_t)bbox.z &&
           pix.y <= (uint32_t)bbox.w;
}

__forceinline__ __device__ float projected_voxel_width_px(
    const float vox_l,
    const float z,
    const int W,
    const float tan_fovx)
{
    return (z <= 1e-8f) ? 1e9f : ((float)W) * vox_l / (2.f * tan_fovx * z);
}

__forceinline__ __device__ float projected_voxel_height_px(
    const float vox_l,
    const float z,
    const int H,
    const float tan_fovy)
{
    return (z <= 1e-8f) ? 1e9f : ((float)H) * vox_l / (2.f * tan_fovy * z);
}

__forceinline__ __device__ float exp_linear_11(float x);
__forceinline__ __device__ float exp_linear_11_bw(float x);

__forceinline__ __device__ bool use_unresolved_proxy(
    const float3& vox_c,
    const float vox_l,
    const float3& ro,
    const float3& cam_forward,
    const int W,
    const int H,
    const float tan_fovx,
    const float tan_fovy,
    float& z)
{
    const float3 rel = make_float3(vox_c.x - ro.x, vox_c.y - ro.y, vox_c.z - ro.z);
    z = rel.x * cam_forward.x + rel.y * cam_forward.y + rel.z * cam_forward.z;
    if (z <= 1e-8f)
        return false;

    const float width_px = projected_voxel_width_px(vox_l, z, W, tan_fovx);
    const float height_px = projected_voxel_height_px(vox_l, z, H, tan_fovy);
    return width_px < 2.f && height_px < 2.f;
}

__forceinline__ __device__ float pixel_footprint_area_at_depth(
    const float z,
    const int W,
    const int H,
    const float tan_fovx,
    const float tan_fovy)
{
    const float pix_w = 2.f * z * tan_fovx / (float)W;
    const float pix_h = 2.f * z * tan_fovy / (float)H;
    return scalar_max(1e-12f, pix_w * pix_h);
}

__forceinline__ __device__ float unresolved_tau_proxy(
    const float* geo_params,
    const float vox_l,
    const float pix_area)
{
    float d_mean = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        d_mean += geo_params[i];
    d_mean *= 0.125f;

    const float sigma_bar = STEP_SZ_SCALE * exp_linear_11(d_mean);
    const float vox_vol = vox_l * vox_l * vox_l;
    return sigma_bar * vox_vol / pix_area;
}

__forceinline__ __device__ float unresolved_tau_proxy_bw(
    const float* geo_params,
    const float vox_l,
    const float pix_area)
{
    float d_mean = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        d_mean += geo_params[i];
    d_mean *= 0.125f;

    const float vox_vol = vox_l * vox_l * vox_l;
    return STEP_SZ_SCALE * exp_linear_11_bw(d_mean) * vox_vol / pix_area * 0.125f;
}

__forceinline__ __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __host__ __device__ float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}

__forceinline__ __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}

__forceinline__ __host__ __device__ float3 operator-(float3 a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

__forceinline__ __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__forceinline__ __host__ __device__ float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}

__forceinline__ __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__forceinline__ __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __host__ __device__ float3 operator+(float b, float3 a)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__forceinline__ __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__forceinline__ __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

__forceinline__ __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

__forceinline__ __host__ __device__ float2 operator*(float b, float2 a)
{
    return make_float2(a.x * b, a.y * b);
}

__forceinline__ __host__ __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}

__forceinline__ __host__ __device__ float2 vec_min(const float2& a, const float2& b)
{
    return make_float2(scalar_min(a.x, b.x), scalar_min(a.y, b.y));
}

__forceinline__ __host__ __device__ float2 vec_max(const float2& a, const float2& b)
{
    return make_float2(scalar_max(a.x, b.x), scalar_max(a.y, b.y));
}

__forceinline__ __host__ __device__ float3 clamp0(const float3& a)
{
    return make_float3(scalar_max(a.x, 0.f), scalar_max(a.y, 0.f), scalar_max(a.z, 0.f));
}

__forceinline__ __device__ float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__ float3 last_col_3x4(const float* matrix)
{
    float3 last_col = {matrix[3], matrix[7], matrix[11]};
    return last_col;
}

__forceinline__ __device__ float3 third_col_3x4(const float* matrix)
{
    float3 third_col = {matrix[2], matrix[6], matrix[10]};
    return third_col;
}


__forceinline__ __device__ float3 transform_3x4(const float* matrix, const float3& p)
{
    float3 transformed = {
        matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3],
        matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7],
        matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11]
    };
    return transformed;
}

__forceinline__ __device__ float3 rotate_3x4(const float* matrix, const float3& p)
{
    float3 transformed = {
        matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
        matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
        matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z
    };
    return transformed;
}


__forceinline__ __device__ float depth_contracted(float x)
{
    return (x < 1.f) ? x : 2.f - 1.f / x;
}

__forceinline__ __device__ float2 ray_aabb(float3 vox_c, float vox_l, float3 ro, float3 rd_inv)
{
    float vox_r = 0.5f * vox_l;
    float3 dir = vox_c - ro;
    float3 c0_ = (dir - vox_r) * rd_inv;
    float3 c1_ = (dir + vox_r) * rd_inv;
    float3 c0 = make_float3(scalar_min(c0_.x, c1_.x), scalar_min(c0_.y, c1_.y), scalar_min(c0_.z, c1_.z));
    float3 c1 = make_float3(scalar_max(c0_.x, c1_.x), scalar_max(c0_.y, c1_.y), scalar_max(c0_.z, c1_.z));
    float2 ab = make_float2(
        scalar_max(scalar_max(c0.x, c0.y), c0.z),
        scalar_min(scalar_min(c1.x, c1.y), c1.z)
    );
    return ab;
}

__forceinline__ __device__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ float softplus(float x)
{
    return (x > 10.f) ? x : log1pf(expf(x));
}

__forceinline__ __device__ float2 softplus2(float base, float x)
{
    const float total = base + x;
    return (total > 10.f)
                ? make_float2(base, x)
                : make_float2(log1pf(expf(total)), 0.0f);
}

__forceinline__ __device__ float softplus_bw(float x)
{
    return (x > 10.f) ? 1.f : sigmoid(x);
}

__forceinline__ __device__ float exp_linear_11(float x)
{
    return (x > 1.1f) ? x : expf(0.909090909091f * x - 0.904689820196f);
}

__forceinline__ __device__ float exp_linear_11_bw(float x)
{
    return (x > 1.1f) ? 1.0f : 0.909090909091f * expf(0.909090909091f * x - 0.904689820196f);
}

__forceinline__ __device__ float relu(float x)
{
    return scalar_max(x, 0.f);
}

__forceinline__ __device__ float relu_bw(float x)
{
    return float(x > 0.f);
}

__forceinline__ __host__ __device__ float safe_rnorm(const float3& v)
{
    return 1.0f / sqrtf(v.x*v.x + v.y*v.y + v.z*v.z + 1e-15f);
}

__forceinline__ __host__ __device__ float safe_rnorm(const float x, const float y, const float z)
{
    return 1.0f / sqrtf(x*x + y*y + z*z + 1e-15f);
}

__forceinline__ __device__ void tri_interp_weight(const float3 qt, float interp_w[8])
{
    float wx[2] = {1.f - qt.x, qt.x};
    float wy[2] = {1.f - qt.y, qt.y};
    float wz[2] = {1.f - qt.z, qt.z};
    interp_w[0] = wx[0] * wy[0] * wz[0];
    interp_w[1] = wx[0] * wy[0] * wz[1];
    interp_w[2] = wx[0] * wy[1] * wz[0];
    interp_w[3] = wx[0] * wy[1] * wz[1];
    interp_w[4] = wx[1] * wy[0] * wz[0];
    interp_w[5] = wx[1] * wy[0] * wz[1];
    interp_w[6] = wx[1] * wy[1] * wz[0];
    interp_w[7] = wx[1] * wy[1] * wz[1];
}

// Debugging helper.
#define CHECK_CUDA(debug) \
if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
