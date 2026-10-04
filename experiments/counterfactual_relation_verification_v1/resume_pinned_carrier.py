"""Resume the pinned DSFormer run from its immutable epoch checkpoint."""
from pathlib import Path
import json
import sys
import torch

ROOT = Path("/data2/liuhaoran/project/cv/psg")
FAIR = Path("/data1/liuhaoran/psg/third_party/fair_psg_git_retry")
sys.path[:0] = [str(FAIR), str(ROOT)]

from fair_psgg.config import Config
from fair_psgg.trainer import Trainer

ANNO = Path("/data1/liuhaoran/psg/datasets/psg.json")
IMG = Path("/data1/liuhaoran/psg/datasets/openpsg/coco")
SEG = Path("/data1/liuhaoran/psg/datasets/openpsg/coco_panoptic_gt/annotations")
OUT = Path("/data1/liuhaoran/psg/checkpoints/dsformer_relation_decomp_v1_seed0")

ckpt = torch.load(OUT / "last_state.pth", map_location="cpu")
start = int(ckpt["epoch"]) + 1
if start >= 40:
    print(json.dumps({"status": "already_complete", "epoch": int(ckpt["epoch"])}))
    raise SystemExit(0)

cfg = Config.from_file(FAIR / "configs/table2/masks-loc-sem.json")
trainer = Trainer(
    anno_path=ANNO, img_dir=IMG, seg_dir=SEG, out_dir=OUT,
    config=cfg, num_workers=4, start_state_dict=ckpt["model"],
    hide_batch_progress=True,
)
trainer.optimizer.load_state_dict(ckpt["optim"])
trainer.best_metric_value = float(ckpt["metric"])
for epoch in range(start, 40):
    print(f"resume epoch {epoch}/39", flush=True)
    trainer.one_epoch(epoch)
(OUT / "done.txt").write_text("done\n")
print(json.dumps({"status": "complete", "last_epoch": 39, "best_metric": trainer.best_metric_value}))
