- Make the raster sort-key order-rank width runtime-aware instead of always using the compiled backend `MAX_NUM_LEVELS`.
  Runs capped at `model.max_num_levels=16` should still use `48` order bits and stay on the 64-bit fast path when the tile count allows it, even if the backend is compiled with support up to `21` levels.
- Add proper type hints everywhere
- add progressive training: low res -> mid res -> full res
