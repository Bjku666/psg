#!/usr/bin/env python3
"""Export frozen DSFormer relation logits plus final relation tokens.

This is an inference-only companion to the pinned Fair-PSG carrier export.
The 384-D ``pair_features`` are the input to ``final_layers`` (before the
relation classifier), captured with a forward-pre-hook.  Pair order and all
mapping inputs are kept identical to the upstream inference implementation.
"""
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import pickle

import torch
from tqdm import tqdm


def cat0(tensors, dim=0):
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim=dim)


@torch.inference_mode()
def inference(model_folder, output_path, anno_path, img_dir, seg_dir,
              batch_size, num_workers, split="test"):
    import sys
    fair = Path(model_folder).parents[2]
    # The caller supplies PYTHONPATH for fair_psg_git_retry; this fallback
    # keeps the script usable when invoked directly from the repository.
    if str(fair) not in sys.path:
        sys.path.insert(0, str(fair))
    from fair_psgg.config import Config
    from fair_psgg.data import get_loader
    from fair_psgg.data.split_batch import split_batch_iter
    from fair_psgg.trainer import prepare_batch
    from fair_psgg import from_config

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_folder = Path(model_folder)
    config = Config.from_file(model_folder / "config.json")
    loader = get_loader(
        anno_path=anno_path, split=split, img_dir=img_dir, seg_dir=seg_dir,
        batch_size=batch_size, num_workers=num_workers,
        augmentations=from_config.get_augmentations(config, split="test"),
    )
    model = from_config.get_model(
        config, num_node_outputs=len(loader.dataset.node_names),
        num_rel_outputs=len(loader.dataset.rel_names),
    )
    checkpoint = torch.load(model_folder / "best_state.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    captured = []
    def capture(_module, inputs):
        captured.append(inputs[0].detach().cpu().clone())
    hook = model.final_layers.register_forward_pre_hook(capture)

    results = defaultdict(lambda: {
        "pairs": [], "rel_scores": [], "rel_rank": [], "pair_features": []
    })
    for batch in split_batch_iter(tqdm(loader, unit="batch"), max_relations=512):
        captured.clear()
        model_input, _sbj_target, _obj_target, _rel_target = prepare_batch(batch, device)
        _sbj_out, _obj_out, rel_out = model(model_input, max_relations=512)
        rel_out = rel_out.sigmoid().cpu().clone()
        pair_features = cat0(captured).float()
        if len(pair_features) != len(rel_out):
            raise RuntimeError(
                f"pair feature/logit length mismatch: {len(pair_features)} vs {len(rel_out)}"
            )
        batch_pairs = batch["sampled_relations"][:, :2]
        bboxes = batch["bboxes"]
        box_labels = batch["box_categories"]
        rel_img_ids = torch.repeat_interleave(batch["image_id"], batch["num_relations"])
        box_img_ids = torch.repeat_interleave(batch["image_id"], batch["num_boxes"])
        for img_id, data_idx, in_img in zip(batch["image_id"], batch["idx"], batch["img"]):
            key = str(int(img_id))
            raw_entry = loader.dataset.entries[data_idx]
            scale_x = raw_entry["width"] / in_img.size(2)
            scale_y = raw_entry["height"] / in_img.size(1)
            scale = max(scale_x, scale_y)
            results[key]["bboxes"] = bboxes[box_img_ids == img_id] * scale
            results[key]["raw_bboxes"] = torch.tensor([b["bbox"] for b in raw_entry["annotations"]])
            results[key]["box_label"] = box_labels[box_img_ids == img_id]
            seg_mask = loader.dataset._load_seg(raw_entry["pan_seg_file_name"], raw_entry["segments_info"])
            seg_mask_ids = seg_mask.long().argmax(0)
            seg_mask_ids[seg_mask.sum(0) == 0] = -1
            mask = rel_img_ids == img_id
            results[key]["raw_mask"] = seg_mask_ids
            results[key]["pairs"].append(batch_pairs[mask])
            results[key]["rel_scores"].append(rel_out[mask])
            results[key]["rel_rank"].append(1 - rel_out[mask, 0])
            results[key]["pair_features"].append(pair_features[mask])
    hook.remove()

    output_data = []
    for img_id, values in results.items():
        output_data.append({
            "img_id": img_id,
            "bboxes": values["raw_bboxes"].numpy(),
            "mask": values["raw_mask"].numpy(),
            "box_label": values["box_label"].numpy(),
            "pairs": cat0(values["pairs"]).numpy(),
            "rel_scores": cat0(values["rel_scores"]).numpy(),
            "rel_rank": cat0(values["rel_rank"]).numpy(),
            "pair_features": cat0(values["pair_features"]).numpy(),
        })
    with open(output_path, "wb") as stream:
        pickle.dump(output_data, stream, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = ArgumentParser()
    parser.add_argument("anno")
    parser.add_argument("img")
    parser.add_argument("seg")
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", choices=("all", "val", "test", "train"), default="test")
    args = parser.parse_args()
    inference(args.model, args.output, args.anno, args.img, args.seg,
               args.bs, args.workers, args.split)


if __name__ == "__main__":
    main()
