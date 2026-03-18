#!/bin/bash
CHECKPOINT_PATH=checkpoints/debug/half_res/counter

python -m sv_raster.new.viz --model_path $CHECKPOINT_PATH
