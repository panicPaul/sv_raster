#!/bin/bash
DATA_PATH=/home/schlack/Documents/3DGS_scenes/360/counter
OUTPUT_PATH=checkpoints/debug/half_res_ss/counter
# CONFIG=cfg/mipnerf360.yaml
CONFIG=cfg/mipnerf360_fast_train.yaml

python -m sv_raster.new.train \
    --model_path $OUTPUT_PATH \
    --cfg.data.eval \
    --cfg.data.source-path $DATA_PATH \
    --cfg.data.res-downscale 1 \
    --cfg-file $CONFIG \
    --cfg.model.max-num-levels 18 \
    --cfg.model.backend new_cuda \
    --cfg.regularizer.lambda-normal-dmean 0.001 \
    --cfg.regularizer.lambda-normal-dmed 0.001 \
    --cfg.coarse-to-fine-schedule.enabled
