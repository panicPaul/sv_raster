# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from tqdm import tqdm
from os import makedirs
import imageio

import torch
import tyro

from sv_raster.new.config import load_config

from sv_raster.new.dataloader.data_pack import DataPack
from sv_raster.new.sparse_voxel_model import SparseVoxelModel
from sv_raster.new.cameras import MiniCam
from sv_raster.new.utils.image_utils import im_tensor2np, viz_tensordepth
from sv_raster.new.utils.camera_utils import interpolate_poses


if __name__ == "__main__":
    @dataclass
    class FlyThroughArgs:
        model_path: Path
        iteration: int = -1
        n_frames: int = 300
        save_scale: float = 1.0
        ids: list[int] = field(default_factory=list)
        starting_id: int = 0
        step_forward: float = 0.0

    args = tyro.cli(FlyThroughArgs)
    print(f"Rendering {args.model_path}")

    # Load config
    cfg = load_config(args.model_path / "config.yaml")

    # Load data
    data_pack = DataPack(
        source_path=cfg.data.source_path,
        image_dir_name=cfg.data.image_dir_name,
        res_downscale=cfg.data.res_downscale,
        res_width=cfg.data.res_width,
        max_render_ss=cfg.model.ss,
        skip_blend_alpha=cfg.data.skip_blend_alpha,
        alpha_is_white=cfg.model.white_background,
        data_device=cfg.data.data_device,
        use_test=cfg.data.eval,
        test_every=cfg.data.test_every,
        camera_params_only=True,
    )

    # Interpolate cameras
    interp_cams = data_pack.interpolate_cameras(
        n_frames=args.n_frames,
        starting_id=args.starting_id,
        ids=args.ids,
        step_forward=args.step_forward,
    )

    # Load model
    voxel_model = SparseVoxelModel(
        n_samp_per_vox=cfg.model.n_samp_per_vox,
        sh_degree=cfg.model.sh_degree,
        ss=cfg.model.ss,
        white_background=cfg.model.white_background,
        black_background=cfg.model.black_background,
    )
    loaded_iter = voxel_model.load_iteration(args.model_path, args.iteration)
    voxel_model.freeze_vox_geo()

    # Rendering
    video = []
    for cam in tqdm(interp_cams, desc="Rendering progress"):

        with torch.no_grad():
            render_pkg = voxel_model.render(cam)
            rendering = render_pkg['color']

        if args.save_scale != 0:
            rendering = torch.nn.functional.interpolate(
                rendering[None],
                scale_factor=args.save_scale,
                mode="bilinear",
                antialias=True)[0]

        video.append(im_tensor2np(rendering))

    outpath = os.path.join(args.model_path, "render_fly_through.mp4")
    imageio.mimwrite(outpath, video, fps=30)
    print("Save to", outpath)
