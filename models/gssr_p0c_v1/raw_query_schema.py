"""Versioned raw Mask2Former query schema and score definitions."""

from __future__ import annotations

from typing import Any

import numpy as np


SCHEMA_VERSION = 1
REQUIRED_QUERY_FIELDS = {
    "query_id",
    "predicted_class",
    "class_score",
    "mask_quality",
    "joint_score",
    "bbox",
    "mask_rle",
    "official_keep_flag",
    "official_panoptic_segment_id",
}


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    output = np.empty_like(values)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def query_scores(class_logits: np.ndarray, mask_logits: np.ndarray) -> dict[str, np.ndarray]:
    """Compute explicitly independent class, mask-quality, and joint scores."""
    class_logits = np.asarray(class_logits, dtype=np.float64)
    mask_logits = np.asarray(mask_logits, dtype=np.float64)
    if class_logits.ndim != 2 or mask_logits.ndim != 3:
        raise ValueError("expected class logits [Q,C+1] and mask logits [Q,H,W]")
    if len(class_logits) != len(mask_logits):
        raise ValueError("class/mask query counts differ")
    shifted = class_logits - class_logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    foreground = probabilities[:, :-1]
    predicted_class = foreground.argmax(axis=1).astype(np.int64)
    class_score = foreground[np.arange(len(foreground)), predicted_class]
    mask_probability = sigmoid(mask_logits)
    binary = mask_probability > 0.5
    numerator = (mask_probability * binary).reshape(len(binary), -1).sum(axis=1)
    denominator = binary.reshape(len(binary), -1).sum(axis=1)
    mask_quality = np.divide(
        numerator,
        denominator,
        out=np.zeros(len(binary), dtype=np.float64),
        where=denominator > 0,
    )
    return {
        "predicted_class": predicted_class,
        "class_score": class_score,
        "mask_quality": mask_quality,
        "joint_score": class_score * mask_quality,
    }


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask as portable, uncompressed COCO-order RLE."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    flat = mask.ravel(order="F")
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.r_[0, changes, len(flat)]).astype(np.int64)
    if len(flat) and flat[0]:
        counts = np.r_[0, counts]
    return {
        "size": [int(mask.shape[0]), int(mask.shape[1])],
        "counts": counts.tolist(),
    }


def decode_binary_mask(rle: dict[str, Any]) -> np.ndarray:
    height, width = (int(value) for value in rle["size"])
    counts = [int(value) for value in rle["counts"]]
    if any(value < 0 for value in counts) or sum(counts) != height * width:
        raise ValueError("invalid RLE counts")
    values = np.arange(len(counts), dtype=np.int64) % 2
    flat = np.repeat(values, counts).astype(bool)
    return flat.reshape((height, width), order="F")


def mask_bbox(mask: np.ndarray) -> list[int]:
    y, x = np.where(np.asarray(mask, dtype=bool))
    if not len(x):
        return [0, 0, 0, 0]
    return [int(x.min()), int(y.min()), int(x.max()), int(y.max())]


def validate_image_record(record: dict) -> None:
    required = {
        "schema_version", "image_id", "file_name", "height", "width",
        "artifact_file", "queries",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"image record lacks fields: {sorted(missing)}")
    if int(record["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {record['schema_version']}")
    query_ids = []
    for query in record["queries"]:
        missing_query = REQUIRED_QUERY_FIELDS.difference(query)
        if missing_query:
            raise ValueError(f"query lacks fields: {sorted(missing_query)}")
        query_ids.append(int(query["query_id"]))
        expected_joint = float(query["class_score"]) * float(query["mask_quality"])
        if not np.isclose(float(query["joint_score"]), expected_joint, rtol=1e-5, atol=1e-7):
            raise ValueError("joint_score is not class_score * mask_quality")
        if list(query["mask_rle"]["size"]) != [int(record["height"]), int(record["width"])]:
            raise ValueError("query RLE dimensions do not match the image")
    if query_ids != list(range(len(query_ids))):
        raise ValueError("query_id must be contiguous and preserve decoder order")
