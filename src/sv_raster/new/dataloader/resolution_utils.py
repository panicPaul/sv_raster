TILE_SIZE = 16
MAX_NUM_TILES = 1 << 16


def compute_target_image_size(width, height, res_downscale=0.0, res_width=0):
    if res_downscale > 0:
        downscale = res_downscale
    elif res_width > 0:
        downscale = width / res_width
    else:
        downscale = 1

    target_width = round(width / downscale)
    target_height = round(height / downscale)
    return target_width, target_height


def validate_camera_resolution(width, height, res_downscale=0.0, res_width=0, max_render_ss=1.0, image_name=""):
    target_width, target_height = compute_target_image_size(
        width=width,
        height=height,
        res_downscale=res_downscale,
        res_width=res_width,
    )
    render_width = round(target_width * max_render_ss)
    render_height = round(target_height * max_render_ss)
    n_tiles_x = (render_width + TILE_SIZE - 1) // TILE_SIZE
    n_tiles_y = (render_height + TILE_SIZE - 1) // TILE_SIZE
    n_tiles = n_tiles_x * n_tiles_y
    if n_tiles > MAX_NUM_TILES:
        image_desc = f" for '{image_name}'" if image_name else ""
        raise ValueError(
            f"Target render resolution {render_height}x{render_width}{image_desc} requires "
            f"{n_tiles} tiles of size {TILE_SIZE}x{TILE_SIZE}, exceeding the renderer "
            f"limit of {MAX_NUM_TILES}. Increase --res_downscale, lower --res_width, "
            f"or reduce supersampling."
        )
    return target_width, target_height
