#!/usr/bin/env python3
"""Run a stage-wise raw->graph endpoint attrition audit.

This runner consumes the lossless P0C profile (binary query RLE plus logits) and
does all stages per image, so it never materializes a split-sized tensor cache.
It is intentionally an audit/oracle tool: it does not train a selector and its
``--split test`` mode is suitable only for implementation smoke tests.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.gssr_p0_v1.audit import aggregate_counts, image_counts
from models.gssr_p0b_v1.balanced_oracle import (
    gt_set_oracle, inverse_predicate_weights, predicate_counts,
)
from models.gssr_p0b_v1.mask_matcher import load_index_mask, single_mpo_binary_mask_mapping
from models.gssr_p0c_v1.native_admission import pre_admission_state, upsampled_query_probabilities
from models.gssr_p0c_v1.raw_query_schema import decode_binary_mask, validate_image_record
from models.pregraph_attrition_v1.endpoint_lifecycle import (
    build_endpoint_lifecycle, stage_transition_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--manifest", type=Path,
                        help="P0C full artifact manifest; omit with --images/--model for one-pass mode")
    parser.add_argument("--images", type=Path, help="image root for one-pass model forward")
    parser.add_argument("--model", help="Mask2Former model path for one-pass mode")
    parser.add_argument("--gt-seg-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("all", "train", "test"), default="train")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--partition", choices=("fit", "dev", "confirm"))
    parser.add_argument("--population-manifest", type=Path,
                        help="frozen image population manifest (supersedes split-manifest/partition)")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--overlap-mask-area-threshold", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _select_items(psg: dict, args: argparse.Namespace) -> list[dict]:
    test_ids = {str(value) for value in psg.get("test_image_ids", [])}
    items = []
    for item in psg.get("data", []):
        image_id = str(item["image_id"])
        if args.split == "test" and image_id not in test_ids:
            continue
        if args.split == "train" and image_id in test_ids:
            continue
        if item.get("relations"):
            items.append(item)
    if args.population_manifest and (args.split_manifest or args.partition):
        raise ValueError("--population-manifest cannot be combined with --split-manifest/--partition")
    if bool(args.split_manifest) != bool(args.partition):
        raise ValueError("--split-manifest and --partition must be supplied together")
    if args.population_manifest:
        document = json.loads(args.population_manifest.read_text())
        allowed_ids = {str(value) for value in document.get("image_ids", [])}
        allowed_files = {str(value) for value in document.get("file_names", [])}
        if not allowed_ids and not allowed_files:
            raise ValueError("population manifest has no image_ids or file_names")
        items = [item for item in items if str(item["image_id"]) in allowed_ids or str(item["file_name"]) in allowed_files]
    elif args.partition:
        document = json.loads(args.split_manifest.read_text())
        allowed = {str(value) for value in document[args.partition]}
        items = [item for item in items if str(item["image_id"]) in allowed]
    items.sort(key=lambda item: str(item["image_id"]))
    if args.max_images is not None:
        items = items[:args.max_images]
    return items


def _bootstrap_deltas(stage_rows: dict[str, list[dict]], num_predicates: int,
                      seed: int = 0, replicates: int = 2000) -> dict:
    """Paired bootstrap over physical files for adjacent-stage losses."""
    grouped: dict[str, dict[str, list[dict]]] = {}
    for stage, rows in stage_rows.items():
        for row in rows:
            grouped.setdefault(str(row["bootstrap_group"]), {}).setdefault(stage, []).append(row)
    groups = [grouped[key] for key in sorted(grouped)]
    if not groups:
        return {"unit": "physical_file_name", "replicates": 0, "deltas": {}}

    def metric(sample: list[dict], stage: str) -> float:
        rows = [row for group in sample for row in group.get(stage, [])]
        return float(aggregate_counts(rows, num_predicates)["predicate_balanced_endpoint_support"])

    full = {stage: metric(groups, stage) for stage in ("raw", "semantic", "competition", "admission")}
    rng = np.random.default_rng(seed)
    draws = {"semantic_loss": [], "competition_loss": [], "admission_loss": []}
    for _ in range(int(replicates)):
        sample = [groups[int(index)] for index in rng.integers(0, len(groups), size=len(groups))]
        values = {stage: metric(sample, stage) for stage in ("raw", "semantic", "competition", "admission")}
        draws["semantic_loss"].append(values["raw"] - values["semantic"])
        draws["competition_loss"].append(values["semantic"] - values["competition"])
        draws["admission_loss"].append(values["competition"] - values["admission"])
    result = {"unit": "physical_file_name", "groups": len(groups), "replicates": int(replicates),
              "point": {"raw": full["raw"], "semantic": full["semantic"],
                        "competition": full["competition"], "admission": full["admission"]},
              "deltas": {}}
    point_deltas = {
        "semantic_loss": full["raw"] - full["semantic"],
        "competition_loss": full["semantic"] - full["competition"],
        "admission_loss": full["competition"] - full["admission"],
    }
    for name, values in draws.items():
        array = np.asarray(values, dtype=float)
        result["deltas"][name] = {
            "estimate": float(point_deltas[name]),
            "bootstrap_mean": float(np.mean(array)),
            "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
        }
    return result


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _mapping(gt_mask, masks, gt_labels, labels, threshold):
    return single_mpo_binary_mask_mapping(
        gt_mask, np.asarray(masks, dtype=bool), gt_labels,
        np.asarray(labels, dtype=int), threshold,
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    psg = json.loads(args.psg.read_text())
    records = {}
    if args.manifest:
        with args.manifest.open() as stream:
            for line in stream:
                record = json.loads(line)
                validate_image_record(record)
                records[str(record["file_name"])] = record
    if not args.manifest and not (args.images and args.model):
        raise ValueError("provide --manifest or both --images and --model")
    processor = model = None
    if args.images and args.model:
        from transformers import AutoModelForUniversalSegmentation, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=Path(args.model).exists())
        model = AutoModelForUniversalSegmentation.from_pretrained(
            args.model, local_files_only=Path(args.model).exists()
        ).to(args.device).eval()
    items = _select_items(psg, args)
    if not items:
        raise ValueError("no relation-bearing images selected")
    num_predicates = len(psg["predicate_classes"])
    weights = inverse_predicate_weights(
        predicate_counts([np.asarray(item["relations"]) for item in items], num_predicates)
    )
    stage_rows = {stage: [] for stage in ("raw", "semantic", "competition", "admission")}
    oracle_rows = {stage: [] for stage in ("raw", "semantic", "competition")}
    transitions = {key: 0 for key in ("raw", "semantic", "competition", "admission", "survives")}
    output_rows = []
    for item_number, item in enumerate(items, 1):
        filename = str(item["file_name"])
        if filename not in records and not (args.images and args.model):
            raise ValueError(f"manifest lacks {filename}")
        record = records.get(filename)
        if record is not None:
            if int(record["schema_version"]) != 1:
                raise ValueError(
                    "stage decomposition requires schema v1/p0c_full records with raw mask RLE; "
                    "compact P1 v2 artifacts cannot recover the raw stage"
                )
            artifact = Path(record["artifact_file"])
            if not artifact.is_absolute():
                artifact = args.manifest.parent / artifact
            with np.load(artifact) as data:
                class_logits = np.asarray(data["class_logits"], dtype=np.float32)
                mask_logits = np.asarray(data["mask_logits"], dtype=np.float32)
            raw_masks = np.stack([
                decode_binary_mask(query["mask_rle"]) for query in record["queries"]
            ])
            query_joint_scores = np.asarray([query["joint_score"] for query in record["queries"]], dtype=float)
        else:
            if processor is None or model is None or args.images is None:
                raise ValueError(f"manifest lacks {filename} and one-pass mode is not configured")
            image_path = args.images / filename
            if not image_path.is_file():
                image_path = args.images / Path(filename).name
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            model_inputs = {key: value.to(args.device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
            with torch.inference_mode():
                outputs = model(**model_inputs)
            class_logits = outputs.class_queries_logits[0].float().cpu().numpy()
            mask_logits = outputs.masks_queries_logits[0].float().cpu().numpy()
            raw_probabilities = upsampled_query_probabilities(torch.tensor(mask_logits), (int(item["height"]), int(item["width"]))).numpy()
            raw_masks = raw_probabilities > args.mask_threshold
            foreground = _softmax(class_logits)[:, :-1]
            foreground_labels = foreground.argmax(axis=1)
            query_joint_scores = foreground.max(axis=1)
        if len(raw_masks) != len(class_logits):
            raise ValueError(f"query/artifact count mismatch for {filename}")
        probabilities = _softmax(class_logits)
        labels = probabilities.argmax(axis=1)
        scores = probabilities.max(axis=1)
        if record is not None:
            foreground_labels = np.asarray(
                [query["predicted_class"] for query in record["queries"]], dtype=int
            )
        no_object = class_logits.shape[1] - 1
        semantic_ids = np.flatnonzero((labels != no_object) & (scores > args.threshold))
        target_size = (
            int(record["height"] if record is not None else item["height"]),
            int(record["width"] if record is not None else item["width"]),
        )
        state = pre_admission_state(
            torch.tensor(class_logits), torch.tensor(mask_logits), target_size,
            args.threshold, args.mask_threshold, args.overlap_mask_area_threshold,
        )
        candidates = state["candidates"]
        probabilities = upsampled_query_probabilities(
            torch.tensor(mask_logits), target_size
        ).numpy()
        competition_ids = np.asarray([int(candidate["query_id"]) for candidate in candidates], dtype=int)
        competition_masks = np.stack([
            np.asarray(candidate["winning_mask"], dtype=bool) for candidate in candidates
        ]) if candidates else np.zeros((0, *target_size), dtype=bool)
        competition_labels = np.asarray([int(candidate["label_id"]) for candidate in candidates], dtype=int)
        native_ids = tuple(int(candidate["query_id"]) for candidate in candidates if candidate["native_keep"])
        stored_native_ids = tuple(int(query["query_id"]) for query in record["queries"] if query.get("official_keep_flag")) if record is not None else native_ids
        if native_ids != stored_native_ids:
            raise AssertionError(f"native query provenance mismatch for {filename}")
        native_id_set = set(native_ids)
        native_indices = np.asarray([
            index for index, query_id in enumerate(competition_ids) if int(query_id) in native_id_set
        ], dtype=int)
        gt_mask = load_index_mask(args.gt_seg_root / item["pan_seg_file_name"], item["segments_info"])
        gt_labels = np.asarray([row["category_id"] for row in item["annotations"]], dtype=int)
        stage_masks = {
            "raw": raw_masks,
            "semantic": raw_masks[semantic_ids],
            "competition": competition_masks,
            "admission": competition_masks[native_indices],
        }
        stage_labels = {
            "raw": foreground_labels,
            "semantic": foreground_labels[semantic_ids],
            "competition": competition_labels,
            "admission": competition_labels[native_indices],
        }
        stage_query_ids = {
            "raw": np.arange(len(raw_masks), dtype=int),
            "semantic": semantic_ids,
            "competition": competition_ids,
            "admission": competition_ids[native_indices],
        }
        mappings = {
            stage: _mapping(gt_mask, stage_masks[stage], gt_labels, stage_labels[stage], args.iou_threshold)
            for stage in stage_masks
        }
        lifecycle_mappings = {
            "raw": mappings["raw"], "semantic": mappings["semantic"],
            "competition": mappings["competition"],
            "admission": mappings["admission"],
        }
        lifecycle = build_endpoint_lifecycle(
            lifecycle_mappings, len(gt_labels),
            query_id_orders=stage_query_ids,
        )
        for key, value in stage_transition_counts(lifecycle).items():
            transitions[key] += int(value)
        relations = np.asarray(item["relations"], dtype=int)
        relation_endpoints = {
            int(endpoint)
            for subject, obj, _predicate in relations.reshape(-1, 3)
            for endpoint in (subject, obj)
        }
        semantic_match_by_query = {
            int(query_id): int(gt_id)
            for query_id, gt_id in zip(
                semantic_ids, np.asarray(mappings["semantic"].candidate_to_gt, dtype=int)
            ) if int(gt_id) >= 0
        }
        winner_map = np.asarray(state["winner_map"], dtype=np.int32)
        weighted = probabilities * scores[:, None, None]
        near_miss = []
        competition_id_set = set(int(value) for value in competition_ids)
        native_id_set = set(int(value) for value in native_ids)
        for query_id, gt_index in semantic_match_by_query.items():
            if query_id in native_id_set:
                continue
            # Match Mask2Former's check_segment_validity: compute_segments
            # multiplies mask probabilities by class scores before this test.
            confident = weighted[query_id] >= args.mask_threshold
            original_area = int(confident.sum())
            won_area = int((winner_map == query_id).sum())
            deficit = max(
                0,
                int(np.floor(args.overlap_mask_area_threshold * original_area)) + 1 - won_area,
            )
            lost = confident & (winner_map != query_id) & (winner_map >= 0)
            margins = weighted[winner_map.clip(min=0), np.indices(target_size)[0], np.indices(target_size)[1]] - weighted[query_id]
            margins = margins[lost]
            if deficit and len(margins):
                rescue_cost = float(np.sort(np.maximum(margins, 0.0))[:min(deficit, len(margins))].sum())
            else:
                rescue_cost = 0.0
            near_miss.append({
                "query_id": int(query_id), "gt_index": int(gt_index),
                "relation_critical": int(gt_index) in relation_endpoints,
                "competition_candidate": query_id in competition_id_set,
                "original_area": original_area, "won_area": won_area,
                "area_ratio": float(won_area / original_area) if original_area else 0.0,
                "deficit_pixels": int(deficit),
                "lost_pixels": int(lost.sum()),
                "rescue_cost": rescue_cost,
                "margin_min": float(np.min(margins)) if len(margins) else None,
                "margin_mean": float(np.mean(margins)) if len(margins) else None,
            })
        per_image = {
            "image_id": str(item["image_id"]), "file_name": filename,
            "native_k": int(len(native_ids)),
            "transitions": stage_transition_counts(lifecycle),
            "near_miss": near_miss,
            "endpoints": [
                {"gt_index": index,
                 "raw_queries": list(lifecycle.query_ids["raw"][index]),
                 "semantic_queries": list(lifecycle.query_ids["semantic"][index]),
                 "competition_queries": list(lifecycle.query_ids["competition"][index]),
                 "admission_queries": list(lifecycle.query_ids["admission"][index]),
                 "death_stage": lifecycle.death_stage(index)}
                for index in range(lifecycle.num_entities)
            ],
        }
        output_rows.append(per_image)
        for stage in ("raw", "semantic", "competition"):
            k = len(native_ids)
            # Stage candidates are supersets of native IDs by construction.
            if k > len(stage_masks[stage]):
                raise AssertionError(f"stage {stage} has fewer candidates than native K for {filename}")
            native_selected = np.asarray([
                index for index, query_id in enumerate(stage_query_ids[stage]) if int(query_id) in native_id_set
            ], dtype=int)
            oracle_selected = gt_set_oracle(
                mappings[stage],
                np.asarray([query_joint_scores[int(query_id)] for query_id in stage_query_ids[stage]], dtype=float),
                relations, k, weights,
            )
            count_native = image_counts(len(gt_labels), relations, native_selected, mappings[stage], num_predicates)
            count_oracle = image_counts(len(gt_labels), relations, oracle_selected, mappings[stage], num_predicates)
            for count, destination, strategy in ((count_native, stage_rows[stage], "native_projection"),
                                                  (count_oracle, oracle_rows[stage], "balanced_fixed_k_oracle")):
                count.update({"image_id": str(item["image_id"]), "file_name": filename,
                              "bootstrap_group": filename, "strategy": strategy,
                              "stage": stage, "selected_nodes": int(len(native_selected if strategy == "native_projection" else oracle_selected))})
                destination.append(count)
        admission_count = image_counts(
            len(gt_labels), relations, native_indices, mappings["competition"], num_predicates,
        )
        admission_count.update({"image_id": str(item["image_id"]), "file_name": filename,
                                "bootstrap_group": filename, "strategy": "native_admission",
                                "stage": "admission", "selected_nodes": int(len(native_indices))})
        stage_rows["admission"].append(admission_count)
        if item_number % 25 == 0 or item_number == len(items):
            print(f"progress={item_number}/{len(items)}", flush=True)
    summaries = {}
    for stage in stage_rows:
        summaries[stage] = aggregate_counts(stage_rows[stage], num_predicates)
    oracle_summaries = {stage: aggregate_counts(oracle_rows[stage], num_predicates) for stage in oracle_rows}
    args.output.mkdir(parents=True)
    with (args.output / "per_image.jsonl").open("w") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    bootstrap_rows = {stage: oracle_rows[stage] for stage in ("raw", "semantic", "competition")}
    bootstrap_rows["admission"] = stage_rows["admission"]
    bootstrap = _bootstrap_deltas(bootstrap_rows, num_predicates)
    document = {
        "schema_version": 1,
        "contract": "raw -> semantic eligibility -> full-pool pixel competition -> native admission",
        "split": args.split, "partition": args.partition, "images": len(items),
        "native": summaries["admission"], "stage_native_projection": summaries,
        "stage_fixed_k_oracle": oracle_summaries, "endpoint_transitions": transitions,
        "paired_bootstrap": bootstrap,
        "matching": {"class_compatible_iou_gt": args.iou_threshold},
        "native_k": "per-image final native entity count",
        "official_test_rows_present": args.split == "test",
        "official_test_used_for_model_selection": False,
        "note": "Test mode is an implementation smoke only and must not guide model selection.",
    }
    (args.output / "summary.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
