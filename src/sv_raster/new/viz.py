# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
from pydantic.dataclasses import dataclass
from pathlib import Path
import numpy as np
import imageio.v3 as iio
from scipy.spatial.transform import Rotation

import torch
import tyro
import new_svraster_cuda

from sv_raster.new.config import load_config

from sv_raster.new.dataloader.data_pack import DataPack
from sv_raster.new.sparse_voxel_model import SparseVoxelModel
from sv_raster.new.utils.image_utils import im_tensor2np, viz_tensordepth
from sv_raster.new.cameras import MiniCam

import viser
import viser.transforms as tf


def matrix2wxyz(R):
    return Rotation.from_matrix(R).as_quat()[[3,0,1,2]]

def wxyz2matrix(wxyz):
    return Rotation.from_quat(wxyz[[1,2,3,0]]).as_matrix()


def desaturate_image(im: np.ndarray, amount: float = 0.8) -> np.ndarray:
    im_float = im.astype(np.float32) / 255.0
    gray = im_float.mean(axis=2, keepdims=True)
    mixed = im_float * (1.0 - amount) + gray * amount
    return np.clip(mixed * 255.0, 0, 255).astype(np.uint8)


class SVRasterViewer:
    def __init__(self, cfg, model_path: Path, iteration: int, port: int):

        # Load cameras
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
        self.tr_cam_lst = data_pack.get_train_cameras()
        self.te_cam_lst = data_pack.get_test_cameras()

        # Load model
        self.voxel_model = SparseVoxelModel(
            n_samp_per_vox=cfg.model.n_samp_per_vox,
            sh_degree=cfg.model.sh_degree,
            ss=cfg.model.ss,
            white_background=cfg.model.white_background,
            black_background=cfg.model.black_background,
            max_num_levels=cfg.model.max_num_levels,
        )
        self.voxel_model.load_iteration(model_path, iteration)
        self.voxel_model.freeze_vox_geo()

        # Create viser server
        self.server = viser.ViserServer(port=port)
        self.is_connected = False

        self.server.gui.set_panel_label("SVRaster viser")
        self.server.gui.add_markdown('''
        View control:
        - Mouse drag + scroll
        - WASD + QE keys
        ''')
        self.fps = self.server.gui.add_text("Rending FPS", initial_value="-1", disabled=True)

        # Create gui for setup viewer
        self.active_sh_degree_slider = self.server.gui.add_slider(
            "active_sh_degree",
            min=0,
            max=self.voxel_model.max_sh_degree,
            step=1,
            initial_value=self.voxel_model.active_sh_degree,
        )

        self.ss_slider = self.server.gui.add_slider(
            "ss",
            min=0.5,
            max=2.0,
            step=0.05,
            initial_value=self.voxel_model.ss,
        )

        self.width_slider = self.server.gui.add_slider(
            "width",
            min=64,
            max=4096,
            step=8,
            initial_value=2048,
        )

        self.fovx_slider = self.server.gui.add_slider(
            "fovx",
            min=10,
            max=150,
            step=1,
            initial_value=70,
        )

        self.near_slider = self.server.gui.add_slider(
            "near",
            min=0.02,
            max=10,
            step=0.01,
            initial_value=0.02,
        )

        self.render_dropdown = self.server.gui.add_dropdown(
            "render mod",
            options=["all", "rgb only", "depth only", "normal only"],
            initial_value="all",
        )

        self.output_dropdown = self.server.gui.add_dropdown(
            "output",
            options=["rgb", "level", "alpha", "dmean", "dmed", "dmean2n", "dmed2n", "n"],
            initial_value="rgb",
        )
        self.level_range_slider = self.server.gui.add_multi_slider(
            "level range",
            min=1,
            max=new_svraster_cuda.meta.MAX_NUM_LEVELS,
            step=1,
            initial_value=(3, new_svraster_cuda.meta.MAX_NUM_LEVELS),
            min_range=0,
        )
        self.highlight_level_dropdown = self.server.gui.add_dropdown(
            "highlight level",
            options=["None"] + [str(v) for v in range(1, new_svraster_cuda.meta.MAX_NUM_LEVELS + 1)],
            initial_value="None",
        )

        # Add camera frustrum
        self.tr_frust = []
        self.te_frust = []

        def add_frustum(name, cam, color):
            c2w = cam.c2w.cpu().numpy()
            frame = self.server.scene.add_camera_frustum(
                name,
                fov=cam.fovy,
                aspect=cam.image_width / cam.image_height,
                scale=0.10,
                wxyz=matrix2wxyz(c2w[:3, :3]),
                position=c2w[:3, 3],
                color=color,
                visible=False,
            )
            @frame.on_click
            def _(event: viser.SceneNodePointerEvent):
                print('Select', name)
                target = event.target
                client = event.client
                with client.atomic():
                    client.camera.wxyz = target.wxyz
                    client.camera.position = target.position
            return frame

        for i, cam in enumerate(self.tr_cam_lst):
            self.tr_frust.append(add_frustum(f"/frustum/train/{i:04d}", cam, [0.,1.,0.]))
        for i, cam in enumerate(self.te_cam_lst):
            self.te_frust.append(add_frustum(f"/frustum/test/{i:04d}", cam, [1.,0.,0.]))

        self.show_cam_dropdown = self.server.gui.add_dropdown(
            "show cameras",
            options=["none", "train", "test", "all"],
            initial_value="none",
        )
        @self.show_cam_dropdown.on_update
        def _(_):
            for frame in self.tr_frust:
                frame.visible = self.show_cam_dropdown.value in ["train", "all"]
            for frame in self.te_frust:
                frame.visible = self.show_cam_dropdown.value in ["test", "all"]

        # Server listening
        @self.server.on_client_connect
        def _(client: viser.ClientHandle):

            # Init camera
            with client.atomic():
                init_c2w = self.tr_cam_lst[0].c2w.cpu().numpy()
                client.camera.wxyz = matrix2wxyz(init_c2w[:3, :3])
                client.camera.position = init_c2w[:3, 3]

            @client.camera.on_update
            def _(_):
                pass

            # Everyting ready to go
            self.is_connected = True

        # Download current view
        self.download_button = self.server.gui.add_button("Download view")
        @self.download_button.on_click
        def _(event: viser.GuiEvent):
            client = event.client
            assert client is not None

            im, eps = self.render_viser_camera(client.camera)

            client.send_file_download(
                "svraster_viser.png",
                iio.imwrite("<bytes>", im, extension=".png"),
            )

    @torch.no_grad()
    def render_viser_camera(self, camera: viser.CameraHandle):
        width = self.width_slider.value
        height = round(width / camera.aspect)
        fovx_deg = self.fovx_slider.value
        fovy_deg = fovx_deg * height / width
        near = self.near_slider.value

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = wxyz2matrix(camera.wxyz)
        c2w[:3, 3] = camera.position

        minicam = MiniCam(
            c2w,
            fovx=np.deg2rad(fovx_deg),
            fovy=np.deg2rad(fovy_deg),
            width=width,
            height=height,
            near=near,
        )

        self.voxel_model.active_sh_degree = int(self.active_sh_degree_slider.value)
        self.voxel_model.level_color_range = tuple(int(v) for v in self.level_range_slider.value)

        render_opt = {
            'ss': self.ss_slider.value,
            'output_T': True,
            'output_depth': True,
            'output_normal': True,
        }
        if self.output_dropdown.value == "level":
            render_opt['color_mode'] = "level"
        if self.render_dropdown.value == "rgb only":
            render_opt['output_depth'] = False
            render_opt['output_normal'] = False
        elif self.render_dropdown.value == "depth only":
            render_opt['color_mode'] = "dontcare"
            render_opt['output_normal'] = False
        elif self.render_dropdown.value == "normal only":
            render_opt['color_mode'] = "dontcare"
            render_opt['output_depth'] = False

        start = time.time()
        try:
            self.voxel_model.level_render_filter = None
            render_pkg = self.voxel_model.render(minicam, **render_opt)
        except RuntimeError as e:
            print(e)
        torch.cuda.synchronize()
        end = time.time()
        eps = end - start

        if self.output_dropdown.value == "dmean":
            im = viz_tensordepth(render_pkg['depth'][0])
        elif self.output_dropdown.value == "dmed":
            im = viz_tensordepth(render_pkg['depth'][2])
        elif self.output_dropdown.value == "dmean2n":
            depth2normal = minicam.depth2normal(render_pkg['depth'][0])
            im = im_tensor2np(depth2normal * 0.5 + 0.5)
        elif self.output_dropdown.value == "dmed2n":
            depth_med2normal = minicam.depth2normal(render_pkg['depth'][2])
            im = im_tensor2np(depth_med2normal * 0.5 + 0.5)
        elif self.output_dropdown.value == "n":
            im = im_tensor2np(render_pkg['normal'] * 0.5 + 0.5)
        elif self.output_dropdown.value == "alpha":
            im = im_tensor2np(1 - render_pkg["T"].repeat(3, 1, 1))
        else:
            im = im_tensor2np(render_pkg["color"])

        highlight_level = self.highlight_level_dropdown.value
        if highlight_level != "None":
            self.voxel_model.level_render_filter = int(highlight_level)
            highlight_pkg = self.voxel_model.render(minicam, **render_opt)
            self.voxel_model.level_render_filter = None

            highlight_alpha = 1 - highlight_pkg["T"]
            if highlight_alpha.shape[0] == 1:
                highlight_alpha = highlight_alpha.repeat(3, 1, 1)
            highlight_alpha = im_tensor2np(highlight_alpha).astype(np.float32) / 255.0
            highlight_alpha = np.clip(highlight_alpha, 0.0, 1.0)

            desaturated = desaturate_image(im)
            im = np.clip(
                desaturated.astype(np.float32) * (1.0 - highlight_alpha)
                + im.astype(np.float32) * highlight_alpha,
                0,
                255,
            ).astype(np.uint8)
            del highlight_pkg

        self.voxel_model.level_render_filter = None
        del render_pkg

        return im, eps

    def update(self):
        if not self.is_connected:
            return

        times = []
        for client in self.server.get_clients().values():
            im, eps = self.render_viser_camera(client.camera)
            times.append(eps)
            client.scene.set_background_image(im, format="jpeg")

        if len(times):
            fps = 1 / np.mean(times)
            self.fps.value = f"{round(fps):4d}"



if __name__ == "__main__":
    @dataclass
    class VizArgs:
        model_path: Path
        iteration: int = -1
        port: int = 8080

    args = tyro.cli(VizArgs)
    print(f"Rendering {args.model_path}")

    # Load config
    cfg = load_config(args.model_path / "config.yaml")

    # Create and run viewer
    svraster_viewer = SVRasterViewer(cfg, model_path=args.model_path, iteration=args.iteration, port=args.port)

    while True:
        svraster_viewer.update()
        time.sleep(0.003)
