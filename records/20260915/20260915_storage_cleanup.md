# Storage cleanup before endpoint_survival_v2

The `/data2` volume had only 39 GB free. Generated raw-query exports from
closed experiment lines were removed before launching the new fit-500 matrix.
This cleanup does not remove source data, model weights, frozen populations,
result summaries, logs, split manifests, or the active fit-500 raw evidence.

## Scientific state retained

- `results/gssr_p1_v2/` retains the failed dev actionability result, all eight
  evaluation-shard summaries/logs, and the grouped train split manifest.
- `records/20260913/20260913_p1_v2_contract.md` retains the export profile and
  contract; `models/gssr_p0c_v1/export_raw_mask_queries.py` is the regeneration
  entry point. Use `/data2/liuhaoran/venvs/fair_psg_p0/bin/python` as documented
  in the repository README.
- The OpenPSG annotation/images/panoptic GT, Mask2Former checkpoint and HF
  cache are retained under `/data2/liuhaoran/psg_data/`.
- The active `psg-pregraph-fit500-seed0-p0c-full-{shard0of8..shard7of8,merged}`
  exports are retained because endpoint_survival_v2 consumes them directly.
- Git-tracked summaries retain the earlier P0C test evidence. Its raw exports
  can be regenerated from the frozen exporter, dataset, and checkpoint.

## Removed generated directories

The following non-Git artifact directories were deleted. They are not directly
recoverable from trash; they are reproducible by rerunning the frozen exporter.

- partial, superseded full-mask P1 exports:
  `p1_v2_train_shard0of4` through `p1_v2_train_shard3of4`
- closed P1 compact export:
  `p1_v2_train_compact_shard0of4` through
  `p1_v2_train_compact_shard3of4` and
  `p1_v2_train_compact_merged_v1`
- historical P0C full-test export:
  `fulltest_mask2former_swin_large_raw_queries_shard0of4_v1` through
  `fulltest_mask2former_swin_large_raw_queries_shard3of4_v1` and
  `fulltest_mask2former_swin_large_raw_queries_merged_v1`

All paths were children of:

```text
/data2/liuhaoran/psg_data/raw_queries/mask2former_swin_large/
```

The targets were resolved with `find`/`du`, and the process table was checked
before deletion. No running exporter or evaluator referenced them.
