#!/usr/bin/env python3
"""Export raw Mask2Former queries, features, masks, and native keep decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from transformers import AutoModelForUniversalSegmentation, AutoProcessor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.gssr_p0c_v1.native_admission import (
    native_panoptic_admission,
    pre_admission_state,
    upsampled_query_probabilities,
)
from models.gssr_p0c_v1.raw_query_schema import (
    SCHEMA_VERSION,
    encode_binary_mask,
    mask_bbox,
    query_scores,
    validate_image_record,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--psg", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--split", choices=("train", "test", "all"), default="test",
                        help="PSG image split to export (P1 uses train)")
    parser.add_argument("--file-list", type=Path,
                        help="optional newline-delimited image basenames; overrides split")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--overlap-mask-area-threshold", type=float, default=0.8)
    return parser.parse_args()


def pooled_pixel_features(pixel_features: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
    masks = functional.interpolate(
        mask_logits.unsqueeze(0),
        size=pixel_features.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0].sigmoid()
    flattened_masks = masks.flatten(1)
    flattened_features = pixel_features.flatten(1).transpose(0, 1)
    denominator = flattened_masks.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return flattened_masks @ flattened_features / denominator


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite output directory: {args.output}")
    with args.psg.open() as stream:
        psg = json.load(stream)
    test_ids = {str(value) for value in psg.get("test_image_ids", [])}
    if args.file_list:
        wanted_files = {line.strip().split("/", 1)[-1] for line in args.file_list.read_text().splitlines()
                        if line.strip()}
    else:
        if args.split == "test":
            wanted_ids = test_ids
        elif args.split == "train":
            wanted_ids = {str(row["image_id"]) for row in psg.get("data", [])
                          if str(row["image_id"]) not in test_ids}
        else:
            wanted_ids = {str(row["image_id"]) for row in psg.get("data", [])}
        wanted_files = {str(row["file_name"]).split("/", 1)[-1] for row in psg.get("data", [])
                        if str(row["image_id"]) in wanted_ids}
    image_paths = sorted(path for path in args.images.glob("*.jpg") if path.name in wanted_files)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards > 0 and 0 <= shard_index < num_shards")
    total_selected_images = len(image_paths)
    image_paths = image_paths[args.shard_index::args.num_shards]
    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]
    if not image_paths:
        raise ValueError(f"no PSG {args.split} images found under --images")

    shard_image_count = len(image_paths)
    manifest_path = args.output / "manifest.jsonl"
    completed_files: set[str] = set()
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError("resume requires an existing manifest.jsonl")
        with manifest_path.open() as stream:
            for line in stream:
                existing = json.loads(line)
                validate_image_record(existing)
                completed_files.add(str(existing["file_name"]).split("/", 1)[-1])
        unexpected = completed_files.difference(path.name for path in image_paths)
        if unexpected:
            raise ValueError(f"resume manifest contains files outside this shard: {sorted(unexpected)[:5]}")
        image_paths = [path for path in image_paths if path.name not in completed_files]

    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=Path(args.model).exists())
    model = AutoModelForUniversalSegmentation.from_pretrained(
        args.model, local_files_only=Path(args.model).exists()
    ).to(device).eval()
    class_names = psg["thing_classes"] + psg["stuff_classes"]
    if len(model.config.id2label) != len(class_names):
        raise ValueError("model and PSG class counts differ")
    for index, name in enumerate(class_names):
        if model.config.id2label[index] != name:
            raise ValueError(f"class {index} differs: {model.config.id2label[index]} != {name}")

    args.output.mkdir(parents=True, exist_ok=args.resume)
    artifact_dir = args.output / "artifacts"
    artifact_dir.mkdir(exist_ok=args.resume)
    started_at = time.monotonic()
    manifest_mode = "a" if args.resume else "w"
    with manifest_path.open(manifest_mode) as manifest:
        with torch.inference_mode():
            for image_number, image_path in enumerate(image_paths, start=1):
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                model_inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                    if isinstance(value, torch.Tensor)
                }
                outputs = model(**model_inputs, output_hidden_states=True)
                class_logits = outputs.class_queries_logits[0]
                mask_logits = outputs.masks_queries_logits[0]
                decoder_features = outputs.transformer_decoder_last_hidden_state[0]
                pixel_features = outputs.pixel_decoder_last_hidden_state[0]
                pooled_features = pooled_pixel_features(pixel_features, mask_logits)
                target_size = (image.height, image.width)
                query_probabilities = upsampled_query_probabilities(mask_logits, target_size)
                binary_masks = query_probabilities > 0.5
                score_values = query_scores(
                    class_logits.float().cpu().numpy(),
                    mask_logits.float().cpu().numpy(),
                )
                native_segmentation, native_segments = native_panoptic_admission(
                    class_logits,
                    mask_logits,
                    target_size,
                    args.threshold,
                    args.mask_threshold,
                    args.overlap_mask_area_threshold,
                )
                admission_state = pre_admission_state(
                    class_logits, mask_logits, target_size, args.threshold,
                    args.mask_threshold, args.overlap_mask_area_threshold,
                )
                official = processor.post_process_panoptic_segmentation(
                    outputs,
                    threshold=args.threshold,
                    mask_threshold=args.mask_threshold,
                    overlap_mask_area_threshold=args.overlap_mask_area_threshold,
                    label_ids_to_fuse=set(),
                    target_sizes=[target_size],
                )[0]
                official_segmentation = official["segmentation"].cpu().numpy()
                if not np.array_equal(native_segmentation, official_segmentation):
                    raise AssertionError("query-aware native admission differs from official segmentation")
                native_without_query = [
                    {key: value for key, value in segment.items() if key != "query_id"}
                    for segment in native_segments
                ]
                if native_without_query != official["segments_info"]:
                    raise AssertionError("query-aware native admission differs from official segments_info")
                native_by_query = {
                    int(segment["query_id"]): int(segment["id"])
                    for segment in native_segments
                }
                # Derive query provenance independently from the official map
                # and the frozen full-pool winner map (never from native output).
                official_segment_query_ids = []
                winner_map = admission_state["winner_map"]
                for segment in official["segments_info"]:
                    segment_id = int(segment["id"])
                    winners = winner_map[official_segmentation == segment_id]
                    winners = winners[winners >= 0]
                    unique = np.unique(winners)
                    if len(unique) != 1:
                        raise AssertionError("official segment is not explained by one winning query")
                    official_segment_query_ids.append(int(unique[0]))

                artifact_name = f"{image_path.stem}.npz"
                np.savez_compressed(
                    artifact_dir / artifact_name,
                    class_logits=class_logits.float().cpu().numpy(),
                    mask_logits=mask_logits.half().cpu().numpy(),
                    decoder_query_feature=decoder_features.half().cpu().numpy(),
                    pixel_pooled_feature=pooled_features.half().cpu().numpy(),
                    native_panoptic_segmentation=native_segmentation.astype(np.int32),
                    pre_admission_winner_map=admission_state["winner_map"].astype(np.int16),
                    pre_admission_query_ids=np.asarray(
                        [c["query_id"] for c in admission_state["candidates"]], dtype=np.int16),
                    pre_admission_won_area=np.asarray(
                        [c["won_area"] for c in admission_state["candidates"]], dtype=np.int32),
                    pre_admission_original_area=np.asarray(
                        [c["original_area"] for c in admission_state["candidates"]], dtype=np.int32),
                    pre_admission_area_ratio=np.asarray(
                        [c["area_ratio"] for c in admission_state["candidates"]], dtype=np.float32),
                )
                queries = []
                for query_id in range(len(class_logits)):
                    binary = binary_masks[query_id].cpu().numpy()
                    queries.append({
                        "query_id": query_id,
                        "predicted_class": int(score_values["predicted_class"][query_id]),
                        "class_score": float(score_values["class_score"][query_id]),
                        "mask_quality": float(score_values["mask_quality"][query_id]),
                        "joint_score": float(score_values["joint_score"][query_id]),
                        "bbox": mask_bbox(binary),
                        "mask_rle": encode_binary_mask(binary),
                        "official_keep_flag": query_id in native_by_query,
                        "official_panoptic_segment_id": native_by_query.get(query_id),
                    })
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "image_id": int(image_path.stem),
                    "file_name": f"{image_path.parent.name}/{image_path.name}",
                    "height": image.height,
                    "width": image.width,
                    "artifact_file": f"artifacts/{artifact_name}",
                    "native_processor_verified": True,
                    "official_panoptic_segmentation": official_segmentation.astype(np.int32).tolist(),
                    "official_segments_info": official["segments_info"],
                    "official_segment_query_ids": official_segment_query_ids,
                    "pre_admission_candidate_query_ids": [int(c["query_id"]) for c in admission_state["candidates"]],
                    "queries": queries,
                }
                validate_image_record(record)
                manifest.write(json.dumps(record, separators=(",", ":")) + "\n")
                manifest.flush()
                total_done = len(completed_files) + image_number
                if total_done % args.progress_every == 0 or image_number == len(image_paths):
                    elapsed = time.monotonic() - started_at
                    seconds_per_image = elapsed / image_number
                    remaining = seconds_per_image * (shard_image_count - total_done)
                    print(
                        f"progress={total_done}/{shard_image_count} "
                        f"seconds_per_image={seconds_per_image:.2f} eta_seconds={remaining:.0f}",
                        flush=True,
                    )

    contract = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "model": args.model,
        "model_config_sha256": sha256(Path(args.model) / "config.json")
        if Path(args.model).is_dir() else None,
        "psg": {"path": str(args.psg.resolve()), "sha256": sha256(args.psg)},
        "images": str(args.images.resolve()),
        "total_selected_images": total_selected_images,
        "exported_images": shard_image_count,
        "resumed_from_images": len(completed_files),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "split": args.split,
        "score_definitions": {
            "class_score": "max softmax probability over foreground classes",
            "mask_quality": "mean sigmoid(mask_logit) over pixels > 0.5",
            "joint_score": "class_score * mask_quality",
        },
        "artifact_dtypes": {
            "class_logits": "float32",
            "mask_logits": "float16 (learner feature only; not used by v2 replay)",
            "decoder_query_feature": "float16",
            "pixel_pooled_feature": "float16",
            "pre_admission_winner_map": "int16",
            "native_panoptic_segmentation": "int32",
        },
        "native_admission": {
            "threshold": args.threshold,
            "mask_threshold": args.mask_threshold,
            "overlap_mask_area_threshold": args.overlap_mask_area_threshold,
            "stuff_fusion": False,
        },
        "python": sys.version,
        "platform": platform.platform(),
        "device": str(device),
        "environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES",)
            if os.environ.get(key)
        },
    }
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
