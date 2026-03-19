#!/bin/bash
CHECKPOINT_PATH=checkpoints/debug/half_res_depth_supervision_full_training/counter
python -m sv_raster.new.viz --model_path $CHECKPOINT_PATH
