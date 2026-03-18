#!/bin/bash
DATA_PATH=/home/schlack/Documents/3DGS_scenes/360/counter
OUTPUT_PATH=checkpoints/debug/half_res_continous/counter_v4
# CONFIG=cfg/mipnerf360.yaml
CONFIG=cfg/mipnerf360_fast_train.yaml

python -m sv_raster.new.train \
    --data.eval \
    --data.source_path $DATA_PATH \
    --model_path $OUTPUT_PATH \
    --data.res_downscale 2 \
    --cfg_file $CONFIG \
    --model.max_num_levels 16 \
    --model.backend new_cuda_cont \
    --model.sh_degree 0 \
    --regularizer.use_l1
