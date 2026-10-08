"""Checkpoint-only validation for the ACRD profiling/promotion runs."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import torch

from models.ambiguity_conditioned_relation_v1.acrd import (
    AmbiguityConditionedRelationDecoder,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--anno", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=("uniform", "mask", "predicate"), required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=112)
    ap.add_argument("--rel-chunk", type=int, default=224)
    ap.add_argument("--image-ids", default=None)
    args = ap.parse_args()

    os.environ["ACRD_MODE"] = args.mode
    os.environ["ACRD_TOP_K"] = "16"
    os.environ["ACRD_NUM_CANDIDATES"] = "3"
    os.environ["FAIR_PSG_AMP"] = "1"
    os.environ["FAIR_PSG_FUSED"] = "1"

    from fair_psgg.config import Config
    import fair_psgg.from_config as factory
    import fair_psgg.metrics as fair_metrics
    import fair_psgg.trainer as trainer_module
    from fair_psgg.trainer import Trainer

    # The fixed development subset does not contain every predicate class.
    # Fair-PSG's ordinary evaluator asserts on the resulting 0/0 per-class
    # recalls, whereas the agreed dev-only protocol excludes those classes
    # from the macro average (the official split remains untouched).
    def _per_class_dev(gts: torch.Tensor, hits: torch.Tensor) -> torch.Tensor:
        recalls = torch.full(
            gts.shape, float("nan"), dtype=torch.float32, device=hits.device
        )
        observed = gts > 0
        recalls[observed] = hits[observed].float() / gts[observed].float()
        return recalls

    def build_rel_metrics_dict_dev(rel_names, gt_list, output_list):
        r20_g, r20_o = fair_metrics._recall_k_counts(
            k=20, gt_list=gt_list, output_list=output_list
        )
        r50_g, r50_o = fair_metrics._recall_k_counts(
            k=50, gt_list=gt_list, output_list=output_list
        )
        n20_g, n20_o = fair_metrics._nogc_recall_k_counts(
            k=20, gt_list=gt_list, output_list=output_list
        )
        n50_g, n50_o = fair_metrics._nogc_recall_k_counts(
            k=50, gt_list=gt_list, output_list=output_list
        )
        recall20_classes = _per_class_dev(r20_g, r20_o)
        recall50_classes = _per_class_dev(r50_g, r50_o)
        nogc20_classes = _per_class_dev(n20_g, n20_o)
        nogc50_classes = _per_class_dev(n50_g, n50_o)
        assert len(rel_names) == len(recall20_classes)
        metrics = {
            "rel_recall/20": r20_o.sum() / r20_g.sum(),
            "rel_recall/50": r50_o.sum() / r50_g.sum(),
            "rel_mean_recall/20": torch.nanmean(recall20_classes),
            "rel_mean_recall/50": torch.nanmean(recall50_classes),
            "rel_nogc_recall/20": n20_o.sum() / n20_g.sum(),
            "rel_nogc_recall/50": n50_o.sum() / n50_g.sum(),
            "rel_mean_nogc_recall/20": torch.nanmean(nogc20_classes),
            "rel_mean_nogc_recall/50": torch.nanmean(nogc50_classes),
        }
        for name, value in zip(rel_names, recall50_classes):
            metrics[f"rel_class_recall/50-{name}"] = value
        for name, value in zip(rel_names, nogc50_classes):
            metrics[f"rel_class_nogc_recall/50-{name}"] = value
        return {key: value.item() for key, value in metrics.items()}

    trainer_module.build_rel_metrics_dict = build_rel_metrics_dict_dev

    config = Config.from_file(args.config).with_cli_overrides(
        [f"batch_size={args.batch_size}", f"rels_per_batch={args.rel_chunk}"]
    )

    if args.image_ids:
        id_path = Path(args.image_ids)
        if id_path.suffix == ".pkl":
            with id_path.open("rb") as stream:
                id_doc = pickle.load(stream)
            selected_ids = {str(x["image_id"]) for x in id_doc["images"]}
        else:
            selected_ids = {str(x) for x in json.loads(id_path.read_text())}
        original_get_data_entries = factory.get_data_entries

        def get_data_entries(config, anno_path, split, img_dir=None):
            if split != "val":
                return original_get_data_entries(config, anno_path, split, img_dir)
            entries, node_names, rel_names = original_get_data_entries(
                config, anno_path, "train", img_dir
            )
            entries = [e for e in entries if str(e["image_id"]) in selected_ids]
            if not entries:
                raise RuntimeError("development image-id list did not match train entries")
            return entries, node_names, rel_names

        factory.get_data_entries = get_data_entries

    def get_model(config, num_node_outputs, num_rel_outputs):
        extractor = factory.get_extractor(config.extractor)
        with torch.inference_mode():
            sample = extractor(torch.rand(1, 3, 640, 640))
        arch = config.architecture
        return AmbiguityConditionedRelationDecoder(
            num_node_outputs=num_node_outputs,
            num_rel_outputs=num_rel_outputs,
            extractor=extractor,
            transformer_depth=1,
            embed_dim=arch.embed_dim,
            patch_size=arch.patch_size,
            feature_shape=sample.shape[1:],
            use_semantics=arch.use_semantics,
            use_masks=arch.use_masks,
            bg_ratio_strategy=arch.bg_ratio_strategy,
            encode_coords=arch.encode_coords,
            retrieval_mode=args.mode,
            top_k=16,
            num_candidates=3,
        )

    factory.get_model = get_model
    trainer = Trainer(
        anno_path=args.anno,
        img_dir=args.img,
        seg_dir=args.seg,
        config=config,
        out_dir=None,
        num_workers=args.workers,
        hide_batch_progress=True,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trainer.model.load_state_dict(state["model"])
    metrics, _ = trainer.evaluate(0)
    def _json_metric(value):
        value = float(value)
        return value if torch.isfinite(torch.tensor(value)) else None

    result = {
        "schema_version": 1,
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "split": "val",
        "development_image_ids": args.image_ids,
        "metrics": {
            k: _json_metric(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
