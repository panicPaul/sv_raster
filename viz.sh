#!/bin/bash
CHECKPOINT_PATH=checkpoints/debug/full_res_no_ss_l18/counter

python -m sv_raster.new.viz --model_path $CHECKPOINT_PATH
