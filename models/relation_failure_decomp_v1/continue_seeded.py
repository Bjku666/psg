"""Resume an interrupted Fair PSG run from its serialized model state."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

from third_party.fair_psg.fair_psgg.config import Config
from third_party.fair_psg.fair_psgg.trainer import Trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anno", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    args = ap.parse_args()

    seed = 0
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    out = Path(args.out)
    state_path = out / "last_state.pth"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if int(state["epoch"]) + 1 != args.start:
        raise RuntimeError(
            f"expected start={int(state['epoch']) + 1} from {state_path}, got {args.start}"
        )
    config = Config.from_file(args.config)
    trainer = Trainer(
        anno_path=args.anno,
        img_dir=args.img,
        seg_dir=args.seg,
        out_dir=args.out,
        config=config,
        num_workers=0,
        start_state_dict=state["model"],
        hide_batch_progress=True,
    )
    trainer.optimizer.load_state_dict(state["optim"])
    best_metrics = torch.load(out / "best_metrics.pth", map_location="cpu", weights_only=False)
    trainer.best_metric_value = float(best_metrics[trainer.critical_metric])

    for epoch in range(args.start, args.end):
        metrics = trainer.one_epoch(epoch)
        print(
            f"epoch={epoch} {trainer.critical_metric}={metrics[trainer.critical_metric]:.8f}",
            flush=True,
        )
    (out / "done.txt").write_text("done")


if __name__ == "__main__":
    main()
