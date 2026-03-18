#!/bin/bash
CHECKPOINT_PATH=checkpoints/debug/full_res/counter

python -m sv_raster.new.viz --model_path $CHECKPOINT_PATH
