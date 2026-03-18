# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import json
import sys
import time
import uuid
import imageio
import datetime
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import tyro

from sv_raster.new.config import (
    AutoExposureConfig,
    BoundingConfig,
    Config,
    DataConfig,
    InitConfig,
    ModelConfig,
    OptimizerConfig,
    ProcedureConfig,
    RegularizerConfig,
    dump_config,
    load_config,
    load_config_override,
)

from sv_raster.new.utils.system_utils import seed_everything
from sv_raster.new.utils.image_utils import im_tensor2np, viz_tensordepth
from sv_raster.new.utils.bounding_utils import decide_main_bounding
from sv_raster.new.utils import mono_utils
from sv_raster.new.utils import loss_utils

from sv_raster.new.dataloader.data_pack import DataPack, compute_iter_idx
from sv_raster.new.sparse_voxel_model import SparseVoxelModel

import new_svraster_cuda


def training(args, cfg: Config):
    # Init and load data pack
    data_pack = DataPack(
        source_path=cfg.data.source_path,
        image_dir_name=cfg.data.image_dir_name,
        mask_dir_name=cfg.data.mask_dir_name,
        res_downscale=cfg.data.res_downscale,
        res_width=cfg.data.res_width,
        max_render_ss=max(cfg.model.ss, cfg.regularizer.ss_aug_max),
        skip_blend_alpha=cfg.data.skip_blend_alpha,
        alpha_is_white=cfg.model.white_background,
        data_device=cfg.data.data_device,
        use_test=cfg.data.eval,
        test_every=cfg.data.test_every,
    )

    # Instantiate data loader
    tr_cams = data_pack.get_train_cameras()
    tr_cam_indices = compute_iter_idx(len(tr_cams), cfg.procedure.n_iter)

    if cfg.auto_exposure.enable:
        for cam in tr_cams:
            cam.auto_exposure_init()

    # Prepare monocular depth priors if instructed
    if cfg.regularizer.lambda_depthanythingv2:
        mono_utils.prepare_depthanythingv2(
            cameras=tr_cams,
            source_path=cfg.data.source_path,
            force_rerun=False)

    if cfg.regularizer.lambda_mast3r_metric_depth:
        mono_utils.prepare_mast3r_metric_depth(
            cameras=tr_cams,
            source_path=cfg.data.source_path,
            mast3r_repo_path=cfg.regularizer.mast3r_repo_path)

    # Decide main (inside) region bounding box
    bounding = decide_main_bounding(
        bound_mode=cfg.bounding.bound_mode,
        forward_dist_scale=cfg.bounding.forward_dist_scale,
        pcd_density_rate=cfg.bounding.pcd_density_rate,
        bound_scale=cfg.bounding.bound_scale,
        tr_cams=tr_cams,
        pcd=data_pack.point_cloud,
        suggested_bounding=data_pack.suggested_bounding)

    # Init voxel model
    voxel_model = SparseVoxelModel(
        n_samp_per_vox=cfg.model.n_samp_per_vox,
        sh_degree=cfg.model.sh_degree,
        ss=cfg.model.ss,
        white_background=cfg.model.white_background,
        black_background=cfg.model.black_background,
        max_num_levels=cfg.model.max_num_levels,
    )

    if args.load_iteration:
        loaded_iter = voxel_model.load_iteration(
            args.model_path, args.load_iteration)
    else:
        loaded_iter = None
        voxel_model.model_init(
            bounding=bounding,
            outside_level=cfg.bounding.outside_level,
            init_n_level=cfg.init.init_n_level,
            init_out_ratio=cfg.init.init_out_ratio,
            sh_degree_init=cfg.init.sh_degree_init,
            geo_init=cfg.init.geo_init,
            sh0_init=cfg.init.sh0_init,
            shs_init=cfg.init.shs_init,
            cameras=tr_cams,
        )

    first_iter = loaded_iter if loaded_iter else 1
    print(f"Start optmization from iters={first_iter}.")

    # Init optimizer
    def create_trainer():
        # The pytorch built-in `torch.optim.Adam` also works
        optimizer = new_svraster_cuda.sparse_adam.SparseAdam(
            [
                {'params': [voxel_model._geo_grid_pts], 'lr': cfg.optimizer.geo_lr},
                {'params': [voxel_model._sh0], 'lr': cfg.optimizer.sh0_lr},
                {'params': [voxel_model._shs], 'lr': cfg.optimizer.shs_lr},
            ],
            betas=(cfg.optimizer.optim_beta1, cfg.optimizer.optim_beta2),
            eps=cfg.optimizer.optim_eps)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.optimizer.lr_decay_ckpt,
            gamma=cfg.optimizer.lr_decay_mult)
        return optimizer, scheduler

    optimizer, scheduler = create_trainer()
    if loaded_iter and args.load_optimizer:
        optim_ckpt = torch.load(os.path.join(args.model_path, "optim.pt"))
        optimizer.load_state_dict(optim_ckpt['optim'])
        scheduler.load_state_dict(optim_ckpt['sched'])
        del optim_ckpt

    # Some other initialization
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    elapsed = 0

    tr_render_opt = {
        'track_max_w': False,
        'lambda_R_concen': cfg.regularizer.lambda_R_concen,
        'output_T': False,
        'output_depth': False,
        'ss': 1.0,  # disable supersampling at first
        'rand_bg': cfg.regularizer.rand_bg,
        'use_auto_exposure': cfg.auto_exposure.enable,
    }

    sparse_depth_loss = loss_utils.SparseDepthLoss(
        iter_end=cfg.regularizer.sparse_depth_until)
    depthanythingv2_loss = loss_utils.DepthAnythingv2Loss(
        iter_from=cfg.regularizer.depthanythingv2_from,
        iter_end=cfg.regularizer.depthanythingv2_end,
        end_mult=cfg.regularizer.depthanythingv2_end_mult)
    mast3r_metric_depth_loss = loss_utils.Mast3rMetricDepthLoss(
        iter_from=cfg.regularizer.mast3r_metric_depth_from,
        iter_end=cfg.regularizer.mast3r_metric_depth_end,
        end_mult=cfg.regularizer.mast3r_metric_depth_end_mult)
    nd_loss = loss_utils.NormalDepthConsistencyLoss(
        iter_from=cfg.regularizer.n_dmean_from,
        iter_end=cfg.regularizer.n_dmean_end,
        ks=cfg.regularizer.n_dmean_ks,
        tol_deg=cfg.regularizer.n_dmean_tol_deg)
    nmed_loss = loss_utils.NormalMedianConsistencyLoss(
        iter_from=cfg.regularizer.n_dmed_from,
        iter_end=cfg.regularizer.n_dmed_end)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    iter_rng = range(first_iter, cfg.procedure.n_iter+1)
    progress_bar = tqdm(iter_rng, desc="Training")
    for iteration in iter_rng:

        # Start processing time tracking of this iteration
        iter_start.record()

        # Increase the degree of SH by one up to a maximum degree
        if iteration % 1000 == 0:
            voxel_model.sh_degree_add1()

        # Recompute sh from cameras
        if iteration in cfg.procedure.reset_sh_ckpt:
            print("Reset sh0 from cameras.")
            print("Reset shs to zero.")
            voxel_model.reset_sh_from_cameras(tr_cams)
            torch.cuda.empty_cache()

        # Use default super-sampling option
        if iteration > 1000:
            if cfg.regularizer.ss_aug_max > 1:
                tr_render_opt['ss'] = np.random.uniform(1, cfg.regularizer.ss_aug_max)
            elif 'ss' in tr_render_opt:
                tr_render_opt.pop('ss')  # Use default ss

        need_sparse_depth = cfg.regularizer.lambda_sparse_depth > 0 and sparse_depth_loss.is_active(iteration)
        need_depthanythingv2 = cfg.regularizer.lambda_depthanythingv2 > 0 and depthanythingv2_loss.is_active(iteration)
        need_mast3r_metric_depth = cfg.regularizer.lambda_mast3r_metric_depth > 0 and mast3r_metric_depth_loss.is_active(iteration)
        need_nd_loss = cfg.regularizer.lambda_normal_dmean > 0 and nd_loss.is_active(iteration)
        need_nmed_loss = cfg.regularizer.lambda_normal_dmed > 0 and nmed_loss.is_active(iteration)
        tr_render_opt['output_T'] = cfg.regularizer.lambda_T_concen > 0 or cfg.regularizer.lambda_T_inside > 0 or cfg.regularizer.lambda_mask > 0 or need_sparse_depth or need_nd_loss or need_depthanythingv2 or need_mast3r_metric_depth
        tr_render_opt['output_normal'] = need_nd_loss or need_nmed_loss
        tr_render_opt['output_depth'] = need_sparse_depth or need_nd_loss or need_nmed_loss or need_depthanythingv2 or need_mast3r_metric_depth

        if iteration >= cfg.regularizer.dist_from and cfg.regularizer.lambda_dist:
            tr_render_opt['lambda_dist'] = cfg.regularizer.lambda_dist

        if iteration >= cfg.regularizer.ascending_from and cfg.regularizer.lambda_ascending:
            tr_render_opt['lambda_ascending'] = cfg.regularizer.lambda_ascending

        # Update auto exposure
        if cfg.auto_exposure.enable and iteration in cfg.auto_exposure.auto_exposure_upd_ckpt:
            for cam in tr_cams:
                with torch.no_grad():
                    ref = voxel_model.render(cam, ss=1.0)['color']
                cam.auto_exposure_update(ref, cam.image.cuda())

        # Pick a Camera
        cam = tr_cams[tr_cam_indices[iteration-1]]

        # Get gt image
        gt_image = cam.image.cuda()
        if cfg.regularizer.lambda_R_concen > 0:
            tr_render_opt['gt_color'] = gt_image

        # Render
        render_pkg = voxel_model.render(cam, **tr_render_opt)
        render_image = render_pkg['color']

        # Loss
        mse = loss_utils.l2_loss(render_image, gt_image)

        if cfg.regularizer.use_l1:
            photo_loss = loss_utils.l1_loss(render_image, gt_image)
        elif cfg.regularizer.use_huber:
            photo_loss = loss_utils.huber_loss(render_image, gt_image, cfg.regularizer.huber_thres)
        else:
            photo_loss = mse
        loss = cfg.regularizer.lambda_photo * photo_loss

        if need_sparse_depth:
            loss += cfg.regularizer.lambda_sparse_depth * sparse_depth_loss(cam, render_pkg)

        if cfg.regularizer.lambda_mask:
            gt_T = 1 - cam.mask.cuda()
            loss += cfg.regularizer.lambda_mask * loss_utils.l2_loss(render_pkg['T'], gt_T)

        if need_depthanythingv2:
            loss += cfg.regularizer.lambda_depthanythingv2 * depthanythingv2_loss(cam, render_pkg, iteration)

        if need_mast3r_metric_depth:
            loss += cfg.regularizer.lambda_mast3r_metric_depth * mast3r_metric_depth_loss(cam, render_pkg, iteration)

        if cfg.regularizer.lambda_ssim:
            loss += cfg.regularizer.lambda_ssim * loss_utils.fast_ssim_loss(render_image, gt_image)
        if cfg.regularizer.lambda_T_concen:
            loss += cfg.regularizer.lambda_T_concen * loss_utils.prob_concen_loss(render_pkg[f'raw_T'])
        if cfg.regularizer.lambda_T_inside:
            loss += cfg.regularizer.lambda_T_inside * render_pkg[f'raw_T'].square().mean()
        if need_nd_loss:
            loss += cfg.regularizer.lambda_normal_dmean * nd_loss(cam, render_pkg, iteration)
        if need_nmed_loss:
            loss += cfg.regularizer.lambda_normal_dmed * nmed_loss(cam, render_pkg, iteration)

        # Backward to get gradient of current iteration
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Total variation regularization
        if cfg.regularizer.lambda_tv_density and \
                iteration >= cfg.regularizer.tv_from and \
                iteration <= cfg.regularizer.tv_until:
            voxel_model.apply_tv_on_density_field(cfg.regularizer.lambda_tv_density)

        # Optimizer step
        optimizer.step()

        ######################################################
        # Start adaptive voxels pruning and subdividing
        ######################################################

        meet_adapt_period = (
            iteration % cfg.procedure.adapt_every == 0 and \
            iteration >= cfg.procedure.adapt_from and \
            iteration <= cfg.procedure.n_iter-500)
        need_pruning = (
            meet_adapt_period and \
            iteration <= cfg.procedure.prune_until)
        need_subdividing = (
            meet_adapt_period and \
            iteration <= cfg.procedure.subdivide_until and \
            voxel_model.num_voxels < cfg.procedure.subdivide_max_num)

        if need_pruning or need_subdividing:
            # Track voxel statistic
            stat_pkg = voxel_model.compute_training_stat(camera_lst=tr_cams)
            # Cache scheduler state
            scheduler_state = scheduler.state_dict()

        if need_pruning:
            ori_n = voxel_model.num_voxels

            # Compute pruning threshold
            prune_thres = np.interp(
                iteration,
                xp=[cfg.procedure.adapt_from, cfg.procedure.prune_until],
                fp=[cfg.procedure.prune_thres_init, cfg.procedure.prune_thres_final])

            # Prune voxels
            prune_mask = (stat_pkg['max_w'] < prune_thres).squeeze(1)

            # Pruning
            voxel_model.pruning(prune_mask)

            # Show statistic
            new_n = voxel_model.num_voxels
            print(f'[PRUNING]     {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f};  thres={prune_thres:.4f})')

        if need_subdividing:
            ori_n = voxel_model.num_voxels

            # Exclude some voxels
            min_samp_interval = stat_pkg['min_samp_interval']
            if need_pruning:
                min_samp_interval = min_samp_interval[~prune_mask]
            size_thres = min_samp_interval * cfg.procedure.subdivide_samp_thres
            large_enough = (voxel_model.vox_size * 0.5 > size_thres).squeeze(1)
            non_finest = voxel_model.octlevel.squeeze(1) < voxel_model.max_num_levels
            valid_mask = large_enough & non_finest

            # Compute subdivision threshold
            priority = voxel_model.subdivision_priority.squeeze(1) * valid_mask

            if iteration <= cfg.procedure.subdivide_all_until:
                thres = -1
            else:
                thres = priority.quantile(1 - cfg.procedure.subdivide_prop)

            subdivide_mask = (priority > thres) & valid_mask

            # In case the number of voxels over the threshold
            max_n_subdiv = round((cfg.procedure.subdivide_max_num - voxel_model.num_voxels) / 7)
            if subdivide_mask.sum() > max_n_subdiv:
                n_removed = subdivide_mask.sum() - max_n_subdiv
                subdivide_mask &= (priority > priority[subdivide_mask].sort().values[n_removed-1])

            # Subdivision
            voxel_model.subdividing(subdivide_mask)

            # Show statistic
            new_n = voxel_model.num_voxels
            in_p = voxel_model.inside_mask.float().mean().item()
            print(f'[SUBDIVIDING] {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f}; inside={in_p*100:.1f}%)')

            # Reset priority for the next round
            voxel_model.reset_subdivision_priority()

        if need_pruning or need_subdividing:
            # Re-create trainer for the updated parameters
            optimizer, scheduler = create_trainer()
            scheduler.load_state_dict(scheduler_state)
            del scheduler_state

            torch.cuda.empty_cache()

        ######################################################
        # End of adaptive voxels procedure
        ######################################################

        # Update learning rate
        scheduler.step()

        # End processing time tracking of this iteration
        iter_end.record()
        torch.cuda.synchronize()
        elapsed += iter_start.elapsed_time(iter_end)

        # Logging
        with torch.no_grad():
            # Metric
            loss = loss.item()
            psnr = -10 * np.log10(mse.item())

            # Progress bar
            ema_p = max(0.01, 1 / (iteration - first_iter + 1))
            ema_loss_for_log += ema_p * (loss - ema_loss_for_log)
            ema_psnr_for_log += ema_p * (psnr - ema_psnr_for_log)
            if iteration % 10 == 0:
                current_level = int(voxel_model.octlevel.max().item())
                pb_text = {
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "psnr": f"{ema_psnr_for_log:.2f}",
                    "level": f"{current_level}/{voxel_model.max_num_levels}",
                    "vox": f"{voxel_model.num_voxels:.2e}",
                }
                progress_bar.set_postfix(pb_text)
                progress_bar.update(10)
            if iteration == cfg.procedure.n_iter:
                progress_bar.close()

            # Log and save
            training_report(
                args=args,
                cfg=cfg,
                data_pack=data_pack,
                voxel_model=voxel_model,
                iteration=iteration,
                elapsed=elapsed,
                ema_psnr=ema_psnr_for_log)

            if iteration in args.checkpoint_iterations or iteration == cfg.procedure.n_iter:
                voxel_model.save_iteration(args.model_path, iteration, quantize=args.save_quantized)
                if args.save_optimizer:
                    torch.save(
                        {'optim': optimizer.state_dict(), 'sched': scheduler.state_dict()},
                        os.path.join(args.model_path, "optim.pt"))
                print(f"[SAVE] path={voxel_model.latest_save_path}")


def training_report(args, cfg: Config, data_pack, voxel_model, iteration, elapsed, ema_psnr):

    voxel_model.freeze_vox_geo()

    # Progress view
    if args.pg_view_every > 0 and (iteration % args.pg_view_every == 0 or iteration == 1):
        torch.cuda.empty_cache()
        test_cameras = data_pack.get_test_cameras()
        if len(test_cameras) == 0:
            test_cameras = data_pack.get_train_cameras()
        pg_idx = 0
        view = test_cameras[pg_idx]
        render_pkg = voxel_model.render(view, output_depth=True, output_normal=True, output_T=True)
        render_image = render_pkg['color']
        render_depth = render_pkg['depth'][0]
        render_depth_med = render_pkg['depth'][2]
        render_normal = render_pkg['normal']
        render_alpha = 1 - render_pkg['T'][0]

        im = np.concatenate([
            np.concatenate([
                im_tensor2np(render_image),
                im_tensor2np(render_alpha)[...,None].repeat(3, axis=-1),
            ], axis=1),
            np.concatenate([
                viz_tensordepth(render_depth, render_alpha),
                im_tensor2np(render_normal * 0.5 + 0.5),
            ], axis=1),
            np.concatenate([
                im_tensor2np(view.depth2normal(render_depth) * 0.5 + 0.5),
                im_tensor2np(view.depth2normal(render_depth_med) * 0.5 + 0.5),
            ], axis=1),
        ], axis=0)
        torch.cuda.empty_cache()

        outdir = os.path.join(args.model_path, "pg_view")
        outpath = os.path.join(outdir, f"iter{iteration:06d}.jpg")
        os.makedirs(outdir, exist_ok=True)

        imageio.imwrite(outpath, im)

        eps_file = os.path.join(args.model_path, "pg_view", "eps.txt")
        with open(eps_file, 'a') as f:
            f.write(f"{iteration},{elapsed/1000:.1f}\n")

    # Report test and samples of training set
    if iteration in args.test_iterations:
        print(f"[EVAL] running...")
        torch.cuda.empty_cache()
        test_cameras = data_pack.get_test_cameras()
        save_every = max(1, len(test_cameras) // 8)
        outdir = os.path.join(args.model_path, "test_view")
        os.makedirs(outdir, exist_ok=True)
        psnr_lst = []
        video = []
        max_w = torch.zeros([voxel_model.num_voxels, 1], dtype=torch.float32, device="cuda")
        for idx, camera in enumerate(test_cameras):
            render_pkg = voxel_model.render(camera, output_normal=True, track_max_w=True)
            render_image = render_pkg['color']
            im = im_tensor2np(render_image)
            gt = im_tensor2np(camera.image)
            video.append(im)
            if idx % save_every == 0:
                outpath = os.path.join(outdir, f"idx{idx:04d}_iter{iteration:06d}.jpg")
                cat = np.concatenate([gt, im], axis=1)
                imageio.imwrite(outpath, cat)

                outpath = os.path.join(outdir, f"idx{idx:04d}_iter{iteration:06d}_normal.jpg")
                render_normal = render_pkg['normal']
                render_normal = im_tensor2np(render_normal * 0.5 + 0.5)
                imageio.imwrite(outpath, render_normal)
            mse = np.square(im/255 - gt/255).mean()
            psnr_lst.append(-10 * np.log10(mse))
            max_w = torch.maximum(max_w, render_pkg['max_w'])
        avg_psnr = np.mean(psnr_lst)
        imageio.mimwrite(
            os.path.join(outdir, f"video_iter{iteration:06d}.mp4"),
            video, fps=30)
        torch.cuda.empty_cache()

        fps = time.time()
        for idx, camera in enumerate(test_cameras):
            voxel_model.render(camera, track_max_w=False)
        torch.cuda.synchronize()
        fps = len(test_cameras) / (time.time() - fps)
        torch.cuda.empty_cache()

        # Sample training views to render
        train_cameras = data_pack.get_train_cameras()
        for idx in range(0, len(train_cameras), max(1, len(train_cameras)//8)):
            camera = train_cameras[idx]
            render_pkg = voxel_model.render(
                camera, output_normal=True, track_max_w=True,
                use_auto_exposure=cfg.auto_exposure.enable)
            render_image = render_pkg['color']
            im = im_tensor2np(render_image)
            gt = im_tensor2np(camera.image)
            outpath = os.path.join(outdir, f"train_idx{idx:04d}_iter{iteration:06d}.jpg")
            cat = np.concatenate([gt, im], axis=1)
            imageio.imwrite(outpath, cat)

            outpath = os.path.join(outdir, f"train_idx{idx:04d}_iter{iteration:06d}_normal.jpg")
            render_normal = render_pkg['normal']
            render_normal = im_tensor2np(render_normal * 0.5 + 0.5)
            imageio.imwrite(outpath, render_normal)

        print(f"[EVAL] iter={iteration:6d}  psnr={avg_psnr:.2f}  fps={fps:.0f}")

        outdir = os.path.join(args.model_path, "test_stat")
        outpath = os.path.join(outdir, f"iter{iteration:06d}.json")
        os.makedirs(outdir, exist_ok=True)
        with open(outpath, 'w') as f:
            q = torch.linspace(0,1,5, device="cuda")
            max_w_q = max_w.quantile(q).tolist()
            peak_mem = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024 ** 3
            stat = {
                'psnr': avg_psnr,
                'ema_psnr': ema_psnr,
                'elapsed': elapsed,
                'fps': fps,
                'n_voxels': voxel_model.num_voxels,
                'max_w_q': max_w_q,
                'peak_mem': peak_mem,
            }
            json.dump(stat, f, indent=4)

    voxel_model.unfreeze_vox_geo()



@dataclass
class TrainCliArgs:
    model_path: Path | None = None
    cfg_file: Path | None = None
    overwrite: bool = False
    detect_anomaly: bool = False
    test_iterations: list[int] = field(default_factory=lambda: [-1])
    pg_view_every: int = 200
    checkpoint_iterations: list[int] = field(default_factory=list)
    load_iteration: int | None = None
    load_optimizer: bool = False
    save_optimizer: bool = False
    save_quantized: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig.model_construct)
    bounding: BoundingConfig = field(default_factory=BoundingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    regularizer: RegularizerConfig = field(default_factory=RegularizerConfig)
    init: InitConfig = field(default_factory=InitConfig)
    procedure: ProcedureConfig = field(default_factory=ProcedureConfig)
    auto_exposure: AutoExposureConfig = field(default_factory=AutoExposureConfig)


@dataclass
class TrainArgs:
    model_path: Path
    overwrite: bool
    detect_anomaly: bool
    test_iterations: list[int]
    pg_view_every: int
    checkpoint_iterations: list[int]
    load_iteration: int | None
    load_optimizer: bool
    save_optimizer: bool
    save_quantized: bool


def parse_cfg_file_arg(argv: list[str]) -> Path | None:
    for i, token in enumerate(argv):
        if token in {"--cfg-file", "--cfg_file"} and i + 1 < len(argv):
            return Path(argv[i + 1])
        if token.startswith("--cfg-file=") or token.startswith("--cfg_file="):
            return Path(token.split("=", 1)[1])
    return None


def build_cli_defaults(cfg_file: Path | None) -> TrainCliArgs:
    defaults = TrainCliArgs()
    if cfg_file is None:
        return defaults

    override = load_config_override(cfg_file)
    section_defaults = {
        "model": defaults.model,
        "data": defaults.data,
        "bounding": defaults.bounding,
        "optimizer": defaults.optimizer,
        "regularizer": defaults.regularizer,
        "init": defaults.init,
        "procedure": defaults.procedure,
        "auto_exposure": defaults.auto_exposure,
    }
    for section_name, section_default in section_defaults.items():
        if section_name not in override:
            continue
        section_override = override.pop(section_name)
        if not isinstance(section_override, dict):
            raise TypeError(f"Config override section '{section_name}' must be a mapping.")
        section_data = section_default.model_dump(mode="python", exclude_unset=True)
        section_data.update(section_override)
        if section_name == "data" and "source_path" not in section_data:
            updated_section = type(section_default).model_construct(**section_data)
        else:
            updated_section = type(section_default).model_validate(section_data)
        setattr(defaults, section_name, updated_section)

    if override:
        unknown_keys = ", ".join(sorted(override.keys()))
        raise KeyError(f"Unknown config override section(s): {unknown_keys}")

    defaults.cfg_file = cfg_file
    return defaults


def build_config(args: TrainCliArgs) -> Config:
    return Config.model_validate({
        "model": args.model.model_dump(mode="python"),
        "data": args.data.model_dump(mode="python"),
        "bounding": args.bounding.model_dump(mode="python"),
        "optimizer": args.optimizer.model_dump(mode="python"),
        "regularizer": args.regularizer.model_dump(mode="python"),
        "init": args.init.model_dump(mode="python"),
        "procedure": args.procedure.model_dump(mode="python"),
        "auto_exposure": args.auto_exposure.model_dump(mode="python"),
    })


def apply_schedule_multiplier(cfg: Config) -> Config:
    schedule_multiplier = cfg.procedure.schedule_multiplier
    if schedule_multiplier == 1.0:
        return cfg

    scaled_cfg = cfg.model_copy(deep=True)

    scaled_cfg.optimizer.geo_lr /= schedule_multiplier
    scaled_cfg.optimizer.sh0_lr /= schedule_multiplier
    scaled_cfg.optimizer.shs_lr /= schedule_multiplier
    scaled_cfg.optimizer.lr_decay_ckpt = [
        round(v * schedule_multiplier) if v > 0 else v
        for v in scaled_cfg.optimizer.lr_decay_ckpt
    ]

    scaled_cfg.regularizer.sparse_depth_until = round(scaled_cfg.regularizer.sparse_depth_until * schedule_multiplier)
    scaled_cfg.regularizer.ascending_from = round(scaled_cfg.regularizer.ascending_from * schedule_multiplier)
    scaled_cfg.regularizer.dist_from = round(scaled_cfg.regularizer.dist_from * schedule_multiplier)
    scaled_cfg.regularizer.tv_from = round(scaled_cfg.regularizer.tv_from * schedule_multiplier)
    scaled_cfg.regularizer.tv_until = round(scaled_cfg.regularizer.tv_until * schedule_multiplier)
    scaled_cfg.regularizer.n_dmean_from = round(scaled_cfg.regularizer.n_dmean_from * schedule_multiplier)
    scaled_cfg.regularizer.n_dmean_end = round(scaled_cfg.regularizer.n_dmean_end * schedule_multiplier)
    scaled_cfg.regularizer.n_dmed_from = round(scaled_cfg.regularizer.n_dmed_from * schedule_multiplier)
    scaled_cfg.regularizer.n_dmed_end = round(scaled_cfg.regularizer.n_dmed_end * schedule_multiplier)
    scaled_cfg.regularizer.depthanythingv2_from = round(scaled_cfg.regularizer.depthanythingv2_from * schedule_multiplier)
    scaled_cfg.regularizer.depthanythingv2_end = round(scaled_cfg.regularizer.depthanythingv2_end * schedule_multiplier)
    scaled_cfg.regularizer.mast3r_metric_depth_from = round(scaled_cfg.regularizer.mast3r_metric_depth_from * schedule_multiplier)
    scaled_cfg.regularizer.mast3r_metric_depth_end = round(scaled_cfg.regularizer.mast3r_metric_depth_end * schedule_multiplier)

    scaled_cfg.procedure.n_iter = round(scaled_cfg.procedure.n_iter * schedule_multiplier)
    scaled_cfg.procedure.adapt_from = round(scaled_cfg.procedure.adapt_from * schedule_multiplier)
    scaled_cfg.procedure.adapt_every = round(scaled_cfg.procedure.adapt_every * schedule_multiplier)
    scaled_cfg.procedure.prune_until = round(scaled_cfg.procedure.prune_until * schedule_multiplier)
    scaled_cfg.procedure.subdivide_until = round(scaled_cfg.procedure.subdivide_until * schedule_multiplier)
    scaled_cfg.procedure.subdivide_all_until = round(scaled_cfg.procedure.subdivide_all_until * schedule_multiplier)
    scaled_cfg.procedure.reset_sh_ckpt = [
        round(v * schedule_multiplier) if v > 0 else v
        for v in scaled_cfg.procedure.reset_sh_ckpt
    ]
    scaled_cfg.procedure.schedule_multiplier = 1.0
    scaled_cfg.auto_exposure.auto_exposure_upd_ckpt = [
        round(v * schedule_multiplier) if v > 0 else v
        for v in scaled_cfg.auto_exposure.auto_exposure_upd_ckpt
    ]

    return scaled_cfg


def normalize_iterations(iterations: list[int], n_iter: int) -> list[int]:
    normalized = list(iterations)
    for i, value in enumerate(normalized):
        if value < 0:
            normalized[i] += n_iter + 1
    return normalized


def resolve_model_path(model_path: Path | None) -> Path:
    if model_path is not None:
        return model_path
    datetime_str = datetime.datetime.now().strftime("%Y-%m%d-%H%M")
    unique_str = str(uuid.uuid4())[:6]
    folder_name = f"{datetime_str}-{unique_str}"
    return Path("./output") / folder_name


def resolve_train_args(args: TrainCliArgs, cfg: Config) -> TrainArgs:
    return TrainArgs(
        model_path=resolve_model_path(args.model_path),
        overwrite=args.overwrite,
        detect_anomaly=args.detect_anomaly,
        test_iterations=normalize_iterations(args.test_iterations, cfg.procedure.n_iter),
        pg_view_every=args.pg_view_every,
        checkpoint_iterations=normalize_iterations(args.checkpoint_iterations, cfg.procedure.n_iter),
        load_iteration=args.load_iteration,
        load_optimizer=args.load_optimizer,
        save_optimizer=args.save_optimizer,
        save_quantized=args.save_quantized,
    )


def main() -> None:
    cfg_file = parse_cfg_file_arg(sys.argv[1:])
    cli_defaults = build_cli_defaults(cfg_file)
    cli_args = tyro.cli(TrainCliArgs, default=cli_defaults)
    cfg = apply_schedule_multiplier(build_config(cli_args))
    args = resolve_train_args(cli_args, cfg)

    # Global init
    seed_everything(cfg.procedure.seed)
    torch.cuda.set_device(torch.device("cuda:0"))
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Setup output folder and dump config
    is_resume = args.load_iteration is not None
    if args.model_path.exists() and any(args.model_path.iterdir()) and not (args.overwrite or is_resume):
        raise FileExistsError(
            f"Refusing to train into non-empty output directory: {args.model_path}. "
            "Pass --overwrite to allow this, or use --load-iteration to resume."
        )

    args.model_path.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, args.model_path / "config.yaml", overwrite=args.overwrite or is_resume)
    print(f"Output folder: {args.model_path}")

    # Launch training loop
    training(args, cfg)
    print("Everything done.")


if __name__ == "__main__":
    main()
