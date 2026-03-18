#!/bin/bash
CHECKPOINT_PATH=checkpoints/debug/half_res_level_18/counter

python -m sv_raster.new.viz --model_path $CHECKPOINT_PATH
