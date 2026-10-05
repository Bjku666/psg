#!/usr/bin/env python3
"""B0 qualification for relation-specific visual reacquisition.

This is deliberately an oracle/diagnostic runner.  It keeps the frozen carrier
pair support and ranking, runs the pinned DSFormer once on a local crop for
each physical-image/pair slot, and then measures the exact GT-gated benefit of
substituting the local predicate.  No router or model is trained here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
else:
    ROOT = Path(__file__).resolve().parents[2]

FAIR = Path("/data1/liuhaoran/psg/third_party/fair_psg_git_retry")
if str(FAIR) not in sys.path:
    sys.path.insert(0, str(FAIR))

from fair_psgg.config import Config  # type: ignore
from fair_psgg import from_config  # type: ignore
from fair_psgg.data.augmentation import get_standard_transforms  # type: ignore
from fair_psgg.utils import open_segmask  # type: ignore

from models.relation_decision_regret_v1.official_metric_adapter import (
    evaluate_population,
)
from models.relation_decision_regret_v1.legal_oracle import baseline_selection
from models.relation_decision_regret_v2.population_metric_utility import (
    PopulationDenominators,
    exact_action_utility,
)
from models.relation_decision_regret_v2.predicate_substitution_regret import (
    _native_pred,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def crop_box(boxes: np.ndarray, pair: tuple[int, int], shape: tuple[int, int], mode: str):
    h, w = shape
    union = np.array([
        boxes[list(pair), 0].min(), boxes[list(pair), 1].min(),
        boxes[list(pair), 2].max(), boxes[list(pair), 3].max(),
    ], dtype=float)
    if mode == "tight":
        margin = 0.0
    elif mode == "context":
        margin = 0.25 * max(union[2] - union[0], union[3] - union[1])
    else:
        raise ValueError(mode)
    x1 = max(0, int(np.floor(union[0] - margin)))
    y1 = max(0, int(np.floor(union[1] - margin)))
    x2 = min(w, int(np.ceil(union[2] + margin)))
    y2 = min(h, int(np.ceil(union[3] + margin)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty crop for pair {pair}: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def prepare_crop(entry: dict, pair: tuple[int, int], img_root: Path, seg_root: Path,
                 transforms, mode: str):
    image = Image.open(img_root / entry["file_name"]).convert("RGB")
    width, height = image.size
    boxes = np.asarray([a["bbox"] for a in entry["annotations"]], dtype=np.float32)
    x1, y1, x2, y2 = crop_box(boxes, pair, (height, width), mode)
    image = image.crop((x1, y1, x2, y2))
    # torchvision's NumPy bridge is ABI-incompatible in this environment.
    # Materialize the PIL crop directly as a torch CHW float tensor and skip
    # the standard ToTensor transform below.
    image = torch.tensor(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1).float().div(255.0)
    panoptic = open_segmask(seg_root / entry["pan_seg_file_name"])
    masks = []
    for index in pair:
        segment_id = int(entry["segments_info"][index]["id"])
        masks.append(panoptic[y1:y2, x1:x2] == segment_id)
    # Torch/NumPy ABI in this environment can reject the ndarray subclass
    # returned by the panoptic decoder even though it prints as ndarray.
    # Canonicalize through an owning copy before crossing the boundary.
    seg = torch.tensor(np.array(np.stack(masks, axis=0), dtype=np.bool_, copy=True)).bool()
    pair_boxes = torch.tensor(boxes[list(pair)].copy())
    pair_boxes[:, (0, 2)] -= x1
    pair_boxes[:, (1, 3)] -= y1
    img, seg, pair_boxes = image, seg, pair_boxes
    for transform in transforms:
        if transform.__class__.__name__ == "ToTensor":
            continue
        img, seg, pair_boxes = transform(img, seg, pair_boxes)
    return img, seg, pair_boxes


@torch.inference_mode()
def run_local_scores(model, samples, device: torch.device, batch_size: int):
    output = []
    for start in range(0, len(samples), batch_size):
        if start == 0 or start % (batch_size * 10) == 0:
            print(f"local crop inference {start}/{len(samples)}", flush=True)
        chunk = samples[start:start + batch_size]
        imgs = torch.stack([x[0] for x in chunk]).to(device)
        seg = torch.cat([x[1] for x in chunk], dim=0).to(device)
        boxes = torch.cat([x[2] for x in chunk], dim=0)
        num = torch.full((len(chunk),), 2, dtype=torch.long)
        pairs = torch.arange(0, 2 * len(chunk), 2, dtype=torch.long)
        pair_ids = torch.stack((pairs, pairs + 1), dim=1)
        cats = torch.cat([x[3] for x in chunk], dim=0).to(device)
        data = {
            "img": imgs,
            "bboxes": boxes,
            "num_boxes": num,
            # Fair-PSG's forward uses pair_ids to index CPU num_boxes; keep
            # the index tensor on CPU as the native inference path does.
            "pair_ids": pair_ids,
            "box_categories": cats,
            "segmentation": seg,
        }
        _, _, rel = model(data, max_relations=len(chunk))
        output.extend(rel.sigmoid().detach().cpu().numpy())
    return output


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--psg", type=Path, required=True)
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--img-root", type=Path, required=True)
    p.add_argument("--seg-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("tight", "context"), default="tight")
    p.add_argument("--budget", type=int, nargs="+", default=(20, 50))
    p.add_argument("--max-physical", type=int)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    psg = json.loads(args.psg.read_text())
    by_id = {str(x["image_id"]): x for x in psg["data"]}
    split = json.loads(args.split_manifest.read_text())
    fit_ids = {str(x) for x in split["fit"]}
    with args.compact.open("rb") as f:
        compact = pickle.load(f)
    images = [x for x in compact["images"] if str(x["image_id"]) in fit_ids]
    by_file = {}
    for row in images:
        by_file.setdefault(str(row["file_name"]), row)
    files = sorted(by_file)
    if args.max_physical is not None:
        files = files[:args.max_physical]
    images = [row for row in images if str(row["file_name"]) in set(files)]
    if not images:
        raise ValueError("no compact rows overlap grouped fit split")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = Config.from_file(args.model / "config.json")
    model = from_config.get_model(config, num_node_outputs=133, num_rel_outputs=57)
    state = torch.load(args.model / "best_state.pth", map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device).eval()
    transforms = get_standard_transforms(is_train=False)

    # One local forward per unique physical file/pair, then fan the result out
    # to duplicate PSG scene rows.  This is the corrected image identity rule.
    jobs, job_keys = [], []
    for filename in files:
        entry = by_id[str(by_file[filename]["image_id"])]
        seen = set()
        for row in [x for x in images if str(x["file_name"]) == filename]:
            for cand in row["candidates"]:
                pair = tuple(map(int, cand["pair"]))
                if pair in seen:
                    continue
                seen.add(pair)
                img, seg, boxes = prepare_crop(entry, pair, args.img_root, args.seg_root,
                                               transforms, args.mode)
                cats = torch.tensor([entry["annotations"][i]["category_id"] for i in pair], dtype=torch.long)
                jobs.append((img, seg, boxes, cats))
                job_keys.append((filename, pair))
    local_scores = run_local_scores(model, jobs, device, args.batch_size)
    local_by_key = {key: score for key, score in zip(job_keys, local_scores)}

    num_pred = len(psg["predicate_classes"])
    denominators = PopulationDenominators.from_records(images, num_pred)
    report = {
        "schema_version": 1,
        "name": "bap_psg_v1_b0_crop_oracle",
        "status": "complete",
        "mode": args.mode,
        "population": {"scene_rows": len(images), "physical_files": len(files),
                        "group_key": "file_name", "official_test_used": False},
        "carrier": {"model": str(args.model), "checkpoint_sha256": sha256(args.model / "best_state.pth"),
                     "compact": str(args.compact), "compact_contract": compact.get("contract")},
        "local_jobs": len(jobs),
        "local_scores": {
            f"{filename}|{pair[0]}-{pair[1]}": np.asarray(score, dtype=np.float32).tolist()
            for (filename, pair), score in local_by_key.items()
        },
        "budgets": {},
    }
    for budget in args.budget:
        native_rows, oracle_rows = [], []
        action_count = 0
        for image in images:
            rows = []
            for candidate in image["candidates"]:
                row = dict(candidate)
                row["pair"] = tuple(row["pair"])
                row["gt_pair"] = None if row.get("gt_pair") is None else tuple(row["gt_pair"])
                row["local_scores"] = local_by_key.get((str(image["file_name"]), row["pair"]))
                rows.append(row)
            baseline = baseline_selection(rows, min(int(budget), len(rows)))
            for row in baseline:
                row["pred"] = _native_pred(row)
            oracle = [dict(row) for row in baseline]
            for index, row in enumerate(baseline):
                scores = row.get("local_scores")
                if scores is None:
                    continue
                scores = np.asarray(scores, dtype=float)
                options = [int(i) - 1 for i in np.argsort(-scores, kind="stable") if int(i) > 0]
                if not options:
                    continue
                action = exact_action_utility(image["relations"], row, options[0], denominators, num_pred)
                if float(action["delta_mr"]) > 0:
                    oracle[index]["pred"] = int(options[0])
                    oracle[index]["pred_score"] = float(scores[options[0] + 1])
                    action_count += 1
            native_rows.append({"image_id": image["image_id"], "file_name": image["file_name"],
                                "bootstrap_group": image["file_name"], "relations": image["relations"],
                                "selected": baseline, "budget": int(budget)})
            oracle_rows.append({"image_id": image["image_id"], "file_name": image["file_name"],
                                "bootstrap_group": image["file_name"], "relations": image["relations"],
                                "selected": oracle, "budget": int(budget)})
        native = evaluate_population(native_rows, num_pred)
        oracle = evaluate_population(oracle_rows, num_pred)
        report["budgets"][str(budget)] = {
            "native_mR": native[f"mR@{budget}"], "native_R": native[f"R@{budget}"],
            "acquisition_oracle_mR": oracle[f"mR@{budget}"], "acquisition_oracle_R": oracle[f"R@{budget}"],
            "delta_mR_pp": 100.0 * (oracle[f"mR@{budget}"] - native[f"mR@{budget}"]),
            "delta_R_pp": 100.0 * (oracle[f"R@{budget}"] - native[f"R@{budget}"]),
            "positive_actions": action_count,
            "baseline_slots": sum(len(x["selected"]) for x in native_rows),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x) + "\n")
    print(json.dumps(report, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))


if __name__ == "__main__":
    main()
