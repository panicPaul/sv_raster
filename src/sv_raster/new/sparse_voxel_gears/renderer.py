# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

from sv_raster.new.utils.image_utils import resize_rendering


def level2rgb(octlevel: torch.Tensor, level_range: tuple[int, int] | None = None, *, max_num_levels: int) -> torch.Tensor:
    level = octlevel.float().squeeze(1)
    if level_range is None:
        level_min = 1
        level_max = max_num_levels
    else:
        level_min, level_max = level_range
    denom = max(1, level_max - level_min)
    t = (level - level_min).clamp(0, denom) / denom

    # A compact false-color ramp that separates coarse and fine voxels clearly.
    r = (1.5 - (4.0 * t - 3.0).abs()).clamp(0.0, 1.0)
    g = (1.5 - (4.0 * t - 2.0).abs()).clamp(0.0, 1.0)
    b = (1.5 - (4.0 * t - 1.0).abs()).clamp(0.0, 1.0)
    return torch.stack([r, g, b], dim=1)


class SVRenderer:

    def freeze_vox_geo(self):
        '''
        Freeze grid points parameter and pre-gather them to each voxel.
        '''
        with torch.no_grad():
            self.frozen_vox_geo = self.backend.renderer.GatherGeoParams.apply(
                self.vox_key,
                torch.arange(self.num_voxels, device="cuda"),
                self._geo_grid_pts
            )
        self._geo_grid_pts.requires_grad = False

    def unfreeze_vox_geo(self):
        '''
        Unfreeze grid points parameter.
        '''
        del self.frozen_vox_geo
        self._geo_grid_pts.requires_grad = True

    def _gather_continuous_sh(self, idx):
        sh0 = self.backend.renderer.GatherFeatParams.apply(
            self.vox_key,
            idx,
            self._sh0,
        )
        return sh0

    def vox_fn(self, idx, cam_pos, color_mode=None, viewdir=None):
        '''
        Per-frame voxel property processing. Two important operations:
        1. Gather grid points parameter into each voxel.
        2. Compute view-dependent color of each voxel.

        Input:
            @idx        Indices for active voxel for current frame.
            @cam_pos    Camera position.
        Output:
            @vox_params A dictionary of the pre-process voxel properties.
        '''

        # Gather the density values at the eight corners of each voxel.
        # It defined a trilinear density field.
        # The final tensor are in shape [#vox, 8]
        if hasattr(self, 'frozen_vox_geo'):
            geos = self.frozen_vox_geo
        else:
            geos = self.backend.renderer.GatherGeoParams.apply(
                self.vox_key,
                idx,
                self._geo_grid_pts
            )

        if hasattr(self, "level_render_filter") and self.level_render_filter is not None:
            visible_mask = self.octlevel.squeeze(1) == int(self.level_render_filter)
            geos = geos.clone()
            if geos.ndim == 3:
                geos[~visible_mask] = 0.0
                geos[~visible_mask, :, 0] = -100.0
            else:
                geos[~visible_mask] = -100.0

        # Compute voxel colors
        if color_mode is None or color_mode == "sh":
            active_sh_degree = self.active_sh_degree
            color_mode = "sh"
        elif color_mode.startswith("sh"):
            active_sh_degree = int(color_mode[2])
            color_mode = "sh"

        if color_mode == "sh":
            if self.color_is_grid:
                sh0 = self._gather_continuous_sh(idx)
                residual_rgbs = self.backend.renderer.SH_eval_residual.apply(
                    active_sh_degree,
                    idx,
                    self.vox_center,
                    cam_pos,
                    viewdir,
                    self._shs,
                )
                vox_params = {
                    'geos': geos,
                    'sh0': sh0,
                    'rgbs': residual_rgbs,
                    'subdiv_p': self._subdiv_p,
                }
                if vox_params['subdiv_p'] is None:
                    vox_params['subdiv_p'] = torch.ones([self.num_voxels, 1], device="cuda")
                return vox_params
            rgbs = self.backend.renderer.SH_eval.apply(
                active_sh_degree,
                idx,
                self.vox_center,
                cam_pos,
                viewdir, # Ignore above two when viewdir is not None
                self.sh0,
                self.shs,
            )
        elif color_mode == "rand":
            rgbs = torch.rand([self.num_voxels, 3], dtype=torch.float32, device="cuda")
        elif color_mode == "level":
            rgbs = level2rgb(
                self.octlevel,
                getattr(self, "level_color_range", None),
                max_num_levels=self.backend.meta.MAX_NUM_LEVELS,
            )
        elif color_mode == "dontcare":
            rgbs = torch.empty([self.num_voxels, 3], dtype=torch.float32, device="cuda")
        else:
            raise NotImplementedError

        # Pack everything
        vox_params = {
            'geos': geos,
            'rgbs': rgbs,
            'subdiv_p': self._subdiv_p, # Dummy param to record subdivision priority
        }
        if vox_params['subdiv_p'] is None:
            vox_params['subdiv_p'] = torch.ones([self.num_voxels, 1], device="cuda")

        return vox_params

    def render(
            self,
            camera,
            color_mode=None,
            track_max_w=False,
            ss=None,
            output_depth=False,
            output_normal=False,
            output_T=False,
            rand_bg=False,
            use_auto_exposure=False,
            **other_opt):

        ###################################
        # Pre-processing
        ###################################
        if ss is None:
            ss = self.ss
        w_src, h_src = camera.image_width, camera.image_height
        w, h = round(w_src * ss), round(h_src * ss)
        w_ss, h_ss = w / w_src, h / h_src
        if 'gt_color' in other_opt and other_opt['gt_color'].shape[-2:] != (h, w):
            other_opt['gt_color'] = resize_rendering(other_opt['gt_color'], size=(h, w))

        n_samp_per_vox = other_opt.pop('n_samp_per_vox', self.n_samp_per_vox)

        ###################################
        # Call low-level rasterization API
        ###################################
        raster_settings = self.backend.renderer.RasterSettings(
            color_mode=color_mode,
            n_samp_per_vox=n_samp_per_vox,
            image_width=w,
            image_height=h,
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            cx=camera.cx * w_ss,
            cy=camera.cy * h_ss,
            w2c_matrix=camera.w2c,
            c2w_matrix=camera.c2w,
            bg_color=float(self.white_background),
            near=camera.near,
            need_depth=output_depth,
            need_normal=output_normal,
            track_max_w=track_max_w,
            **other_opt)
        rasterize_kwargs = {}
        if self.backend_name == "new_cuda_aa":
            rasterize_kwargs["octlevels"] = self.octlevel

        color, depth, normal, T, max_w = self.backend.renderer.rasterize_voxels(
            raster_settings,
            self.octpath,
            self.vox_center,
            self.vox_size,
            self.vox_fn,
            **rasterize_kwargs)

        ###################################
        # Post-processing and pack output
        ###################################
        if rand_bg:
            color = color + T * torch.rand_like(color, requires_grad=False)
        elif not self.white_background and not self.black_background:
            color = color + T * color.mean((1,2), keepdim=True)

        if use_auto_exposure:
            color = camera.auto_exposure_apply(color)

        render_pkg = {
            'color': color,
            'depth': depth if output_depth else None,
            'normal': normal if output_normal else None,
            'T': T if output_T else None,
            'max_w': max_w,
        }

        for k in ['color', 'depth', 'normal', 'T']:
            render_pkg[f'raw_{k}'] = render_pkg[k]

            # Post process super-sampling
            if render_pkg[k] is not None and render_pkg[k].shape[-2:] != (h_src, w_src):
                # print(f"Resizing {k} from {render_pkg[k].shape[-2:]} to {(h_src, w_src)}")
                render_pkg[k] = resize_rendering(render_pkg[k], size=(h_src, w_src))

        # Clip intensity
        render_pkg['color'] = render_pkg['color'].clamp(0, 1)

        return render_pkg
