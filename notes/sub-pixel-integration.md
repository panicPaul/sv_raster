# Sub-Pixel Integration in `new_cuda`

## Problem

`new_cuda` currently renders each pixel using a single ray through the pixel center.

At a high level:

1. Preprocess projects each voxel corner and builds a screen-space bounding box.
2. Rasterization considers a voxel for a pixel if the pixel index lies inside that bounding box.
3. The pixel still only receives contribution if the pixel-center ray intersects the voxel AABB.

This creates a failure mode for unresolved voxels:

- a voxel can be fully enclosed by a pixel footprint in image space,
- yet the center ray can miss the voxel entirely,
- so the voxel contributes zero.

This is a point-sampling artifact, not a principled pixel-footprint integral.

## When does this become a problem?

In principle, once a voxel projects below about `2x2` pixels, we are in an undersampled regime.

That should be the actual threshold for this method:

- projected voxel size `< 2x2` pixels

The `1x1` case is still the simplest motivating example, because it makes the center-ray miss problem obvious, but it is too strict as the real trigger. Above `2x2`, ordinary supersampling can still recover the footprint more faithfully. Below `2x2`, center-ray hit testing is already underresolved in the Nyquist sense.

## Current `new_cuda` behavior

The current renderer:

- uses the exact voxel cube as the support region,
- computes `a, b` as the ray entry/exit distances through that cube,
- integrates density only along that single ray segment.

If the center ray misses, there is no contribution at all, even if the voxel lies inside the pixel footprint.

## Desired behavior

For unresolved voxels, we want to stop relying on a single ray-AABB intersection and instead use a proxy contribution that approximates the average contribution over the full pixel footprint.

This means:

- detect unresolved voxels during preprocessing,
- replace their direct center-ray hit/miss behavior with a filtered proxy,
- keep the whole construction differentiable so gradients propagate back to the original leaf parameters.

## Why merging to a coarser proxy helps

For unresolved voxels, the leaf support is too fine to resolve. A better approach is to aggregate the unresolved fine voxels into a coarser proxy representation.

Important detail:

- in a sparse adaptive octree, we cannot rely on complete sibling sets,
- so the proxy must work for partial descendant sets as well.

This rules out a strict "only collapse complete 8-sibling groups" design.

## What should the proxy preserve?

For color / SH:

- a weighted average is a reasonable starting point.

For density:

- the correct target is not the maximum child density,
- and not the center-ray density,
- but an unresolved footprint-average extinction proxy.

The key physical quantity is not raw density by itself, but integrated extinction:

```text
tau = ∫ sigma(x) dx
alpha = 1 - exp(-tau)
```

For unresolved voxels, the proxy should approximate the average `tau` across rays inside the pixel footprint.

## Footprint-based proxy idea

Instead of asking whether the center ray hits, define the unresolved contribution using the pixel footprint.

Conceptually:

```text
tau_proxy = (1 / pixel_area) ∫pixel tau(ray(u,v)) du dv
```

This avoids the center-ray miss problem entirely.

## Practical approximation

We still need a differentiable and cheap approximation.

The useful quantity is density-weighted volume (mass / extinction content), not plain geometric voxel volume.

If `d(x)` is the trilinear pre-activation and the renderer uses:

```text
sigma(x) = STEP_SZ_SCALE * exp_linear_11(d(x))
```

then the unresolved proxy should ideally preserve:

```text
M = ∫voxel sigma(x) dV
```

An inexpensive approximation is:

1. Average the corner pre-activations:

```text
d_bar = mean(d_corner)
```

2. Convert to an effective density:

```text
sigma_bar = STEP_SZ_SCALE * exp_linear_11(d_bar)
```

3. Multiply by the occupied volume represented by the unresolved voxel set:

```text
M_proxy = sigma_bar * occupied_volume
```

This is differentiable with respect to the original corner values.

## From mass to pixel-footprint extinction

Once we have `M_proxy`, we still need an effective line integral through the pixel footprint.

The intended approximation is:

- unresolved rays through one pixel are nearly parallel,
- so mean chord-length / beam arguments can be used,
- giving an expected extinction from volume divided by projected footprint area at depth.

Conceptually:

```text
tau_proxy ≈ M_proxy / footprint_area_at_depth
alpha_proxy = 1 - exp(-tau_proxy)
```

This is the key step that removes dependence on whether the pixel-center ray intersects the voxel.

## Backpropagation requirement

The proxy construction must remain differentiable.

That means:

- do not destroy leaf parameters irreversibly,
- build render-time proxy values as differentiable reductions of the original leaf parameters,
- use the proxy in forward,
- scatter proxy gradients back to leaves through the same weights in backward.

Linear reductions such as averaging are especially attractive because backward is straightforward.

## Current proposed scope

Focus on the `< 2x2` unresolved case.

For now:

- rely on supersampling for voxels above that threshold,
- add a preprocess-time unresolved proxy path for voxels below it,
- use a footprint-based extinction proxy rather than center-ray hit testing.

## Open questions

1. What exact projected-size test should be used in preprocess?
   Current direction: use `< 2x2` as the unresolved threshold, and keep the `1x1` case only as the clearest illustrative example.

2. How should sparse partial descendant sets be grouped into proxies?
   This likely needs to be based on coarser ancestors, but without requiring complete sibling occupancy.

3. How should color aggregation be weighted?
   Volume-weighted and extinction-weighted averages are both plausible.

4. Is averaging corner pre-activations sufficient, or do we need a better local approximation of
   `∫ exp_linear_11(d(x)) dV`?

5. How should depth / normal outputs be defined for unresolved proxies?
   The immediate concern is opacity / color stability, but geometry outputs will also need a principled fallback.
