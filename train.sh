#!/bin/bash
DATA_PATH=/home/schlack/Documents/3DGS_scenes/360/counter
OUTPUT_PATH=checkpoints/debug/full_res_no_ss_l18/counter
# CONFIG=cfg/mipnerf360.yaml
CONFIG=cfg/mipnerf360_fast_train.yaml

python -m sv_raster.new.train \
    --data.eval \
    --data.source_path $DATA_PATH \
    --model_path $OUTPUT_PATH \
    --data.res_downscale 1 \
    --cfg_file $CONFIG \
    --model.max_num_levels 18 \
    --model.backend new_cuda \
    --model.ss  1.0 \
    --regularizer.ss_aug_max  1.0
