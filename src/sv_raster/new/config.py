from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_yaml import parse_yaml_file_as, to_yaml_str
import yaml

from sv_raster.new.backend import BackendName, get_backend_max_num_levels


class ModelConfig(BaseModel):
    # Raster backend used by this run and checkpoint.
    backend: BackendName = "new_cuda"
    # Number of samples taken along each visited voxel segment during rasterization.
    n_samp_per_vox: int = 1
    # Maximum spherical-harmonics degree allocated for view-dependent color.
    sh_degree: int = 3
    # Default supersampling factor used when a render path does not pass `ss` explicitly.
    ss: float = 1.5
    # If true, alpha compositing and RGBA blending use white as the background.
    white_background: bool = False
    # If true, keeps black-background handling enabled in renderer/model code paths.
    black_background: bool = False
    # Per-run octree detail cap; the default `16` uses `3 * 16 = 48` order bits, so many common renders still
    # fit the composite `[tile_id | order_rank]` key into one 64-bit word before the rasterizer needs 128-bit keys.
    max_num_levels: int = 16

    @model_validator(mode="after")
    def validate_backend_limits(self):
        backend_max_num_levels = get_backend_max_num_levels(self.backend)
        if self.max_num_levels > backend_max_num_levels:
            raise ValueError(
                f"model.max_num_levels={self.max_num_levels} exceeds the compiled {self.backend} backend limit "
                f"of {backend_max_num_levels}"
            )
        return self


class DataConfig(BaseModel):
    # Root scene directory used by dataset readers and mono-prior caching.
    source_path: Path
    # Relative image folder name inside `source_path`.
    image_dir_name: str = "images"
    # Relative mask folder name inside `source_path`.
    mask_dir_name: str = "masks"
    # Uniform image downscale factor applied when cameras are created.
    res_downscale: float = 0.0
    # Alternative target image width used to derive downscaling.
    res_width: int = 0
    # If false, RGBA inputs are blended against the configured background color.
    skip_blend_alpha: bool = False
    # Frame tensors are currently staged on CPU and moved to CUDA at use sites.
    data_device: str = "cpu"
    # If true, dataset readers split views into train/test according to `test_every`.
    eval: bool = False
    # Every Nth view goes to the test split when `eval` is enabled.
    test_every: int = 8


class BoundingConfig(BaseModel):
    # Heuristic used to choose the main inside-scene bounding box.
    bound_mode: str = "default"
    # Global multiplier applied to the chosen inside-scene bound.
    bound_scale: float = 1.0
    # Forward-distance multiplier used by forward-looking bound heuristics.
    forward_dist_scale: float = 1.0
    # Point-cloud density factor used by point-cloud based bound selection.
    pcd_density_rate: float = 0.1
    # Number of octree levels reserved for outside/background space.
    outside_level: int = 5


class OptimizerConfig(BaseModel):
    # Learning rate for geometry grid-point parameters.
    geo_lr: float = 0.025
    # Learning rate for SH degree-0 color coefficients.
    sh0_lr: float = 0.010
    # Learning rate for higher-order SH coefficients.
    shs_lr: float = 0.00025
    # Beta1 parameter for sparse Adam.
    optim_beta1: float = 0.1
    # Beta2 parameter for sparse Adam.
    optim_beta2: float = 0.99
    # Numerical epsilon for sparse Adam.
    optim_eps: float = 1e-15
    # Iterations where the learning-rate scheduler decays.
    lr_decay_ckpt: list[int] = [19000]
    # Multiplicative decay factor applied at each scheduler milestone.
    lr_decay_mult: float = 0.1


class RegularizerConfig(BaseModel):
    # Weight of the main photometric reconstruction term.
    lambda_photo: float = 1.0
    # If true, switches the photometric loss from MSE to L1.
    use_l1: bool = False
    # If true, switches the photometric loss from MSE to Huber.
    use_huber: bool = False
    # Delta threshold used by the Huber photometric loss.
    huber_thres: float = 0.03
    # Weight of the SSIM loss term added to the photometric objective.
    lambda_ssim: float = 0.02
    # Weight of sparse-depth supervision from camera sparse points.
    lambda_sparse_depth: float = 0.0
    # Last iteration where sparse-depth supervision remains active.
    sparse_depth_until: int = 10_000
    # Weight of mask supervision on rendered transmittance.
    lambda_mask: float = 0.0

    # Weight of DepthAnythingV2 monocular depth supervision.
    lambda_depthanythingv2: float = 0.0
    # First iteration where DepthAnythingV2 supervision becomes active.
    depthanythingv2_from: int = 3000
    # Last iteration where DepthAnythingV2 supervision is evaluated at full strength.
    depthanythingv2_end: int = 20000
    # End-of-schedule multiplier applied to the DepthAnythingV2 loss weight.
    depthanythingv2_end_mult: float = 0.1

    # Weight of MASt3R metric-depth supervision.
    lambda_mast3r_metric_depth: float = 0.0
    # Filesystem path to the external MASt3R repository.
    mast3r_repo_path: Path | None = None
    # First iteration where MASt3R metric-depth supervision becomes active.
    mast3r_metric_depth_from: int = 0
    # Last iteration where MASt3R metric-depth supervision is evaluated at full strength.
    mast3r_metric_depth_end: int = 20000
    # End-of-schedule multiplier applied to the MASt3R metric-depth loss weight.
    mast3r_metric_depth_end_mult: float = 0.01

    # Weight encouraging final transmittance to concentrate near 0 or 1.
    lambda_T_concen: float = 0.0
    # Weight encouraging final transmittance to approach 0.
    lambda_T_inside: float = 0.0
    # Weight of the per-point RGB concentration loss in the rasterizer backward pass.
    lambda_R_concen: float = 0.01

    # Weight of the ascending regularizer in the rasterizer backward pass.
    lambda_ascending: float = 0.0
    # First iteration where ascending regularization is enabled.
    ascending_from: int = 0

    # Weight of the distortion regularizer in the rasterizer.
    lambda_dist: float = 0.1
    # First iteration where distortion regularization is enabled.
    dist_from: int = 10000

    # Weight of expected-depth vs normal consistency.
    lambda_normal_dmean: float = 0.0
    # First iteration where expected-depth normal consistency is enabled.
    n_dmean_from: int = 10_000
    # Last iteration where expected-depth normal consistency remains active.
    n_dmean_end: int = 20_000
    # Kernel size used by expected-depth normal consistency.
    n_dmean_ks: int = 3
    # Angular tolerance in degrees for expected-depth normal consistency.
    n_dmean_tol_deg: float = 90.0

    # Weight of median-depth vs normal consistency.
    lambda_normal_dmed: float = 0.0
    # First iteration where median-depth normal consistency is enabled.
    n_dmed_from: int = 3000
    # Last iteration where median-depth normal consistency remains active.
    n_dmed_end: int = 20_000

    # Weight of total-variation regularization on the density field.
    lambda_tv_density: float = 1e-10
    # First iteration where TV density regularization is enabled.
    tv_from: int = 0
    # Last iteration where TV density regularization is enabled.
    tv_until: int = 10000

    # Upper bound for random supersampling augmentation after iteration 1000.
    ss_aug_max: float = 1.5
    # If true, training renders use a random background color.
    rand_bg: bool = False

    @model_validator(mode="after")
    def validate_mast3r_repo_path(self):
        if self.lambda_mast3r_metric_depth > 0 and self.mast3r_repo_path is None:
            raise ValueError(
                "regularizer.mast3r_repo_path is required when "
                "regularizer.lambda_mast3r_metric_depth > 0"
            )
        return self


class InitConfig(BaseModel):
    # Initial density pre-activation value for geometry grid points.
    geo_init: float = -10.0
    # Initial SH degree-0 color value before rgb-to-SH conversion.
    sh0_init: float = 0.5
    # Initial value for higher-order SH coefficients.
    shs_init: float = 0.0
    # Initially active SH degree after model initialization.
    sh_degree_init: int = 3
    # Inside-region initialization uses `(2 ** init_n_level) ** 3` voxels.
    init_n_level: int = 6
    # Target outside-voxel count relative to the inside voxel count.
    init_out_ratio: float = 2.0


class ProcedureConfig(BaseModel):
    # Effective iteration budget used by training after schedule normalization.
    n_iter: int = 20_000
    # Multiplies schedule-related fields before training starts; normalized saved configs reset this to `1.0`.
    schedule_multiplier: float = 1.0
    # Global seed used by `seed_everything`.
    seed: int = 3721
    # Iterations where SH colors are recomputed from cameras.
    reset_sh_ckpt: list[int] = [-1]
    # First iteration where adaptive prune/subdivide passes may start.
    adapt_from: int = 1000
    # Frequency in iterations for adaptive prune/subdivide passes.
    adapt_every: int = 1000
    # Last iteration where pruning is allowed.
    prune_until: int = 18000
    # Initial pruning threshold on tracked voxel contribution.
    prune_thres_init: float = 0.0001
    # Final pruning threshold on tracked voxel contribution.
    prune_thres_final: float = 0.05
    # Last iteration where subdivision is allowed.
    subdivide_until: int = 15000
    # Iteration until which all valid voxels are subdivided unconditionally.
    subdivide_all_until: int = 0
    # Multiplier on sampling interval used to decide whether a voxel is large enough to subdivide.
    subdivide_samp_thres: float = 1.0
    # Fraction of valid voxels targeted for subdivision after the all-until stage.
    subdivide_prop: float = 0.05
    # Hard cap on voxel count during adaptive subdivision.
    subdivide_max_num: int = 10_000_000


class AutoExposureConfig(BaseModel):
    # Enables per-camera auto-exposure fitting against rendered references.
    enable: bool = False
    # Iterations where auto-exposure parameters are updated.
    auto_exposure_upd_ckpt: list[int] = [5000, 10000, 15000]


class CoarseToFineScheduleLevelConfig(BaseModel):
    # First max occupied octree level where this temporary image downscale becomes active.
    min_level: int
    # Shared image-supervision downscale applied to all image-based losses at and above `min_level`.
    downscale: float


class CoarseToFineScheduleConfig(BaseModel):
    # Enables progressive temporary downscaling for all image-based supervision.
    enabled: bool = False
    # Ordered level-to-downscale schedule shared by RGB, SSIM, mask, and raster-side GT color terms.
    levels: list[CoarseToFineScheduleLevelConfig] = Field(default_factory=lambda: [
        CoarseToFineScheduleLevelConfig(min_level=11, downscale=8.0),
        CoarseToFineScheduleLevelConfig(min_level=12, downscale=4.0),
        CoarseToFineScheduleLevelConfig(min_level=13, downscale=2.0),
        CoarseToFineScheduleLevelConfig(min_level=14, downscale=1.0),
    ])

    @model_validator(mode="after")
    def validate_levels(self):
        prev_min_level = None
        for level_cfg in self.levels:
            if level_cfg.downscale < 1.0:
                raise ValueError("coarse_to_fine_schedule downscale values must be >= 1.0")
            if prev_min_level is not None and level_cfg.min_level <= prev_min_level:
                raise ValueError("coarse_to_fine_schedule.levels must be strictly increasing in min_level")
            prev_min_level = level_cfg.min_level
        return self


class Config(BaseModel):
    # Model and renderer hyperparameters.
    model: ModelConfig = Field(default_factory=ModelConfig)
    # Dataset loading and image-resolution settings.
    data: DataConfig
    # Scene-bound estimation settings.
    bounding: BoundingConfig = Field(default_factory=BoundingConfig)
    # Optimizer and scheduler settings.
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    # Loss weights and regularization schedules.
    regularizer: RegularizerConfig = Field(default_factory=RegularizerConfig)
    # Sparse-voxel initialization settings.
    init: InitConfig = Field(default_factory=InitConfig)
    # Training schedule and adaptation settings.
    procedure: ProcedureConfig = Field(default_factory=ProcedureConfig)
    # Per-camera exposure fitting settings.
    auto_exposure: AutoExposureConfig = Field(default_factory=AutoExposureConfig)
    # Shared temporary image resolution used by all image-based losses during training.
    coarse_to_fine_schedule: CoarseToFineScheduleConfig = Field(default_factory=CoarseToFineScheduleConfig)


def load_config(cfg_file: str | Path) -> Config:
    """Load the config from a YAML file."""
    return parse_yaml_file_as(Config, cfg_file)


def load_config_override(cfg_file: str | Path) -> dict:
    """Load a partial config override from YAML without validating it as a full Config."""
    cfg_file = Path(cfg_file)
    raw_data = yaml.safe_load(cfg_file.read_text())
    if raw_data is None:
        return {}
    if not isinstance(raw_data, dict):
        raise TypeError(f"Config override must be a mapping at the top level: {cfg_file}")
    return raw_data


def dump_config(
    config: Config,
    path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> str:
    """Dump the config to YAML and optionally write it to disk."""
    yaml_str = to_yaml_str(config)
    if path is None:
        return yaml_str

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing config file: {path}")
    path.write_text(yaml_str)
    return yaml_str
