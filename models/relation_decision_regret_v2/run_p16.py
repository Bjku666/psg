#!/usr/bin/env python3
"""Run P1.6 exact predicate-surgery qualification and CPU MERS.

The command consumes already exported fit/dev carrier pickles.  It never
reads GT at decode time: GT is used only to construct fit utilities and to
evaluate the locked dev result.
"""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.relation_decision_regret_v1.legal_oracle import baseline_selection
from models.relation_decision_regret_v1.official_metric_adapter import mapped_candidates, evaluate_population
from models.relation_decision_regret_v2.predicate_substitution_regret import substitution_teacher, mean_recall, substitution_additivity
from models.relation_decision_regret_v2.selective_repair_oracle import keep_vs_any_topl
from models.relation_decision_regret_v2.mers import MERS
from models.relation_decision_regret_v2.flip_accounting import account_flips


def _records(annotation, predictions, gt_root, budget, depth, with_labels=False):
    by_id = {str(x["img_id"]): x for x in predictions}
    records, labels, missing = [], [], []
    for entry in annotation["data"]:
        if not entry.get("relations"):
            continue
        image_id = str(entry["image_id"])
        item = by_id.get(image_id)
        if item is None:
            missing.append(image_id); continue
        rows = mapped_candidates(entry, item, gt_root, include_unmapped=True)[0]
        baseline = baseline_selection(rows, budget)
        record = {"image_id": image_id, "file_name": str(entry["file_name"]),
                  "bootstrap_group": str(entry["file_name"]), "relations": entry["relations"],
                  "selected": baseline, "budget": int(budget), "candidates": rows}
        records.append(record)
        if with_labels:
            labels.append({"image_id": image_id, "rows": rows, "baseline": baseline,
                           "relations": entry["relations"],
                           "actions": substitution_teacher(entry["relations"], baseline,
                                                            len(annotation["predicate_classes"]), depth)})
    return records, labels, missing


def _metric(records, n, budget):
    result = evaluate_population(records, n)
    return {"mR": float(result[f"mR@{budget}"]), "R": float(result[f"R@{budget}"])}


def _bootstrap_delta(base, repaired, n, budget, repeats=500, seed=0):
    # Grouped bootstrap over physical files; each subset already has one row
    # per image and fixed denominators.
    b = evaluate_population(base, n)["per_image"]
    r = evaluate_population(repaired, n)["per_image"]
    groups = sorted({str(x["bootstrap_group"]) for x in base})
    by_group = {g: [i for i,x in enumerate(base) if str(x["bootstrap_group"]) == g] for g in groups}
    rng = np.random.default_rng(seed); vals=[]
    def m(rows, idx):
        arr=np.stack([rows[i]["per_predicate_recall"] for i in idx]); return float(np.nanmean(np.nanmean(arr,axis=0)))
    all_idx=list(range(len(base)))
    for _ in range(int(repeats)):
        sampled=rng.choice(groups,size=len(groups),replace=True); idx=[i for g in sampled for i in by_group[str(g)]]
        vals.append((m(r,idx)-m(b,idx))*100)
    point=(m(r,all_idx)-m(b,all_idx))*100
    return {"estimate_pp":point,"ci95_pp":[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],"groups":len(groups),"replicates":int(repeats)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--psg",required=True,type=Path); p.add_argument("--fit-predictions",required=True,type=Path)
    p.add_argument("--dev-psg",required=True,type=Path); p.add_argument("--dev-predictions",required=True,type=Path)
    p.add_argument("--gt-seg-root",required=True,type=Path); p.add_argument("--output",required=True,type=Path)
    p.add_argument("--budget",type=int,default=50); p.add_argument("--depth",type=int,default=3)
    p.add_argument("--bootstrap-replicates",type=int,default=500)
    args=p.parse_args(); fit_ann=json.loads(args.psg.read_text()); dev_ann=json.loads(args.dev_psg.read_text())
    with args.fit_predictions.open("rb") as f: fit_pred=pickle.load(f)
    with args.dev_predictions.open("rb") as f: dev_pred=pickle.load(f)
    n=len(fit_ann["predicate_classes"]); k=int(args.budget); depth=int(args.depth)
    fit_base, fit_labels, fit_missing = _records(fit_ann,fit_pred,args.gt_seg_root,k,depth,True)
    dev_base, dev_labels, dev_missing = _records(dev_ann,dev_pred,args.gt_seg_root,k,depth,True)
    # Exact KEEP/SWAP selective oracle on dev.
    dev_rep=[]; additivity=[]; repair_counts=[]; flip_counts={}
    for item in dev_labels:
        selected, diag=keep_vs_any_topl(item["relations"],item["baseline"],n,depth)
        e=dict(next(x for x in dev_base if x["image_id"]==item["image_id"])); e["selected"]=selected; dev_rep.append(e)
        additivity.append(substitution_additivity(item["relations"],item["baseline"],item["actions"],n))
        repair_counts.extend(d["chosen"]=="SWAP" for d in diag["decisions"])
        flips = account_flips(item["relations"], item["baseline"], selected)["counts"]
        for key, value in flips.items(): flip_counts[key] = flip_counts.get(key, 0) + int(value)
    # MERS labels: one utility target per legal action. Fit only on fit images.
    train=[]
    for item in fit_labels:
        byrow={int(x["row"]):x for x in item["baseline"]}
        for a in item["actions"]:
            train.append((byrow[int(a["row"])],int(a["predicate"]),float(a["delta_mr"])))
    # A bounded CPU probe is sufficient for P1.6 qualification.  HistGBDT on
    # every fit action is unnecessarily expensive; the ridge MERS probe keeps
    # the exact action target and makes the run reproducible in minutes.
    if len(train) > 30000:
        train = train[::max(1, len(train) // 30000)]
    model=MERS.fit(train,depth=depth,kind="ridge")
    dev_mers=[]; mers_swaps=0
    for item in dev_labels:
        selected=[dict(x) for x in item["baseline"]]
        byrow={int(x["row"]):x for x in selected}
        predictions = model.choose_many(list(selected))
        for row, pred in zip(list(selected), predictions):
            if pred != int(row.get("pred",0)): mers_swaps += 1
            row["pred"]=int(pred)
            scores=np.asarray(row.get("pred_scores",[]),dtype=float)
            if scores.size>pred+1: row["pred_score"]=float(scores[pred+1])
        e=dict(next(x for x in dev_base if x["image_id"]==item["image_id"])); e["selected"]=selected; dev_mers.append(e)
    base_metrics=_metric(dev_base,n,k); oracle_metrics=_metric(dev_rep,n,k); mers_metrics=_metric(dev_mers,n,k)
    output={"schema_version":1,"contract":{"support":"frozen mapped physical pairs","action":"KEEP or same-pair SWAP","decode_gt":False,"evaluator":"Fair PSG image-wise"},
            "fit_images":len(fit_base),"dev_images":len(dev_base),"missing":{"fit":fit_missing,"dev":dev_missing},"budget":k,"depth":depth,
            "results":{"baseline":base_metrics,"keep_vs_any_topl_oracle":oracle_metrics,"mers":mers_metrics},
            "deltas_pp":{"oracle_mR":(oracle_metrics["mR"]-base_metrics["mR"])*100,"mers_mR":(mers_metrics["mR"]-base_metrics["mR"])*100,"mers_R":(mers_metrics["R"]-base_metrics["R"])*100},
            "oracle_bootstrap":_bootstrap_delta(dev_base,dev_rep,n,k,args.bootstrap_replicates),
            "oracle_diagnostics":{"repair_rate":float(np.mean(repair_counts)) if repair_counts else 0.0,"additivity_max_abs_residual":max((x["max_abs_residual"] for x in additivity),default=0.0)},
            "flip_accounting":flip_counts,
            "mers":{"train_actions":len(train),"dev_swaps":int(mers_swaps)},
            "gates":{"confirm_authorized":False,"next":"evaluate full population before promotion"}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(output,indent=2,allow_nan=True)+"\n"); print(json.dumps(output,indent=2,allow_nan=True))

if __name__=="__main__": main()
