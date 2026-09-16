# relation_failure_decomp_v1 launch record

- Registered immutable decomposition contract before carrier training.
- Carrier: Fair PSG DSFormer, pinned source revision
  `2497be40bd8bd0ca6c2ad25be1522ebf42e6623b`.
- Training recipe: `configs/table2/masks-loc-sem.json`, 40 epochs, seed 0,
  CUDA device 7, 4 data-loader workers. GPUs 2, 5 and 6 became occupied by
  unrelated jobs while the dataset was downloading; no unrelated process was
  touched.
- SHA256: config `a02f873...94058aa2`, OpenPSG annotation
  `d4a8c6d...a06817c1`, Mask2Former carrier weight
  `2bdb3d3...ffe5d3b`.
- Dataset annotation and panoptic GT paths were verified. The required
  `train2017` image directory was absent at launch preparation time; the
  experiment is pending lawful COCO train2017 retrieval.
- No existing DSFormer process or checkpoint was overwritten.
- At 2026-09-15 17:48 CST, resumable COCO retrieval followed by validation,
  extraction, and automatic DSFormer launch was started in tmux session
  `psg_dsformer_relation_decomp_v1_seed0`.
- The first one-image smoke correctly exposed undefined inverse-frequency
  weights on a class-incomplete toy subset. A second engineering-only smoke
  greedily covered all 133 node and 56 predicate classes in train and all 56
  predicates in validation; one full epoch completed and wrote checkpoints.
- Formal training fixes Python, NumPy, CPU and CUDA seeds to 0 and enables
  deterministic-algorithm warnings through the local wrapper.
- The same launch chain will export dense full-pair logits on the frozen
  Mask2Former SGDet carrier, run corrected official evaluation with
  `--dedup fail`, and execute the registered four-stage decomposition.
- Resume check at 2026-09-15 21:58 CST: tmux session and downloader are alive;
  `train2017.zip` is ~73% complete. GPUs 2, 5 and 6 are occupied externally;
  GPU 7 remains free and is reserved by the launch script for the carrier.
