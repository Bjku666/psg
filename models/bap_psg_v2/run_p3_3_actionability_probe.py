#!/usr/bin/env python3
"""Post-acquisition cross-view actionability probe (CPU only).

This consumes the frozen p3.2 global carrier and local crop score dumps.  It
does not train a vision model: only the three pre-registered light-weight
scikit-learn models are fitted inside grouped cross-fitting folds.  A policy
is allowed to be a no-op and its threshold is selected on training groups.
"""
from __future__ import annotations

import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

from models.relation_decision_regret_v1.legal_oracle import baseline_selection
from models.relation_decision_regret_v1.official_metric_adapter import evaluate_population
from models.relation_decision_regret_v2.population_metric_utility import PopulationDenominators, exact_action_utility
from models.relation_decision_regret_v2.predicate_substitution_regret import _native_pred


def entropy(p):
    p = np.asarray(p, float); q = np.clip(p, 1e-8, 1 - 1e-8)
    return float(-(q * np.log(q) + (1-q) * np.log(1-q)).sum())


def prob_stats(p):
    p = np.asarray(p, float); order = np.argsort(-p, kind="stable")
    vals = p[order]
    return float(entropy(p)), float(vals[0]), float(vals[1] if len(vals) > 1 else 0), int(order[0]), int(order[1] if len(order) > 1 else order[0])


def js(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = .5 * (p + q)
    def kl(a, b):
        return float(np.sum(np.where(a > 0, a * np.log((a + 1e-8)/(b + 1e-8)), 0)))
    return .5 * kl(p, m) + .5 * kl(q, m)


def kl(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(np.sum(np.where(p > 0, p * np.log((p + 1e-8)/(q + 1e-8)), 0)))


def geom(psg_by_id, image_id, pair):
    e = psg_by_id[str(image_id)]; w, h = float(e["width"]), float(e["height"])
    bs = [np.asarray(e["annotations"][i]["bbox"], float) for i in pair]
    areas = [max(0., (b[2]-b[0])*(b[3]-b[1]))/(w*h) for b in bs]
    ratio = (max(areas)+1e-8)/(min(areas)+1e-8)
    union = [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]
    centers = [((b[0]+b[2])/(2*w), (b[1]+b[3])/(2*h)) for b in bs]
    dist = float(np.hypot(centers[0][0]-centers[1][0], centers[0][1]-centers[1][1]))
    ua = max(0., union[2]-union[0]) * max(0., union[3]-union[1])/(w*h)
    mag = 1.0/max(ua, 1e-8)
    return [areas[0], areas[1], ratio, dist, ua, mag]


def feature(row, local, group, psg_by_id, image_id):
    g = np.asarray(row["pred_scores"], float)[1:]
    l = np.asarray(local, float)[1:]
    ge, gm1, gm2, ga, ga2 = prob_stats(g); le, lm1, lm2, la, la2 = prob_stats(l)
    base = {
        "F0": [ge, gm1-gm2, gm1, gm2],
        "F1": [le, lm1-lm2, lm1, lm2],
        "F2": [kl(g,l), js(g,l), (lm1-lm2)-(gm1-gm2), l[ga], g[la], float(ga==la),
               float(np.where(np.argsort(-l, kind="stable")==ga)[0][0]), float(np.where(np.argsort(-g, kind="stable")==la)[0][0])],
    }
    base["F3"] = base["F2"] + geom(psg_by_id, image_id, tuple(row["pair"]))
    return base[group], int(la), int(ga)


def make_estimator(name):
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0))
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=8, learning_rate=.05, l2_regularization=1e-3, random_state=0)
    raise ValueError(name)


def predict(est, name, x):
    if name == "logistic": return est.predict_proba(x)[:, 1]
    return est.predict(x)


def choose_threshold(score, utility):
    vals = np.unique(np.r_[score, [np.inf]])
    # score >= threshold means take local.  Include no-op explicitly.
    sums = [float(np.sum(utility[score >= t])) if np.isfinite(t) else 0.0 for t in vals]
    return float(vals[int(np.argmax(sums))])


def evaluate_policy(images, rows, take, num_pred, denominators, budget):
    native_rows, selected_rows = [], []
    positive = harmful = selected = 0
    for im, rr, mask in zip(images, rows, take):
        base = [dict(x) for x in rr]
        for x in base: x["pred"] = _native_pred(x)
        out = [dict(x) for x in base]
        for x, yes in zip(out, mask):
            if yes:
                local = np.asarray(x["local_scores"], float); pred = int(np.argmax(local[1:]))
                pred = pred
                u = exact_action_utility(im["relations"], x, pred, denominators, num_pred)
                selected += 1
                positive += int(u["delta_mr"] > 0); harmful += int(u["delta_mr"] < 0)
                x["pred"] = pred; x["pred_score"] = float(local[pred+1])
        native_rows.append({"image_id": im["image_id"], "file_name": im["file_name"], "bootstrap_group": im["file_name"], "relations": im["relations"], "selected": base, "budget": budget})
        selected_rows.append({"image_id": im["image_id"], "file_name": im["file_name"], "bootstrap_group": im["file_name"], "relations": im["relations"], "selected": out, "budget": budget})
    n = evaluate_population(native_rows, num_pred); s = evaluate_population(selected_rows, num_pred)
    return {"native_mR": n[f"mR@{budget}"], "selected_mR": s[f"mR@{budget}"], "delta_mR_pp": 100*(s[f"mR@{budget}"]-n[f"mR@{budget}" ]), "native_R": n[f"R@{budget}"], "selected_R": s[f"R@{budget}"], "delta_R_pp": 100*(s[f"R@{budget}"]-n[f"R@{budget}" ]), "selected_pairs": selected, "positive_actions": positive, "harmful_actions": harmful}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--compact", type=Path, required=True); ap.add_argument("--tight", type=Path, required=True); ap.add_argument("--context", type=Path, required=True); ap.add_argument("--psg", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--max-images", type=int); ap.add_argument("--seed", type=int, default=0)
    a=ap.parse_args()
    compact=pickle.load(a.compact.open("rb")); tight=json.loads(a.tight.read_text()); context=json.loads(a.context.read_text()); psg=json.loads(a.psg.read_text()); psg_by_id={str(x["image_id"]):x for x in psg["data"]}; num_pred=len(psg["predicate_classes"])
    tmap={k:np.asarray(v,float) for k,v in tight["local_scores"].items()}; cmap={k:np.asarray(v,float) for k,v in context["local_scores"].items()}
    imgs=[im for im in compact["images"] if any(k.startswith(str(im["file_name"])+"|") for k in tmap)]
    imgs.sort(key=lambda x:str(x["file_name"]));
    if a.max_images: imgs=imgs[:a.max_images]
    den=PopulationDenominators.from_records(imgs,num_pred); groups=np.array([str(x["file_name"]) for x in imgs])
    out={"schema_version":1,"name":"p3.3_cross_view_actionability_probe","status":"complete","population":{"scene_rows":len(imgs),"physical_files":len(set(groups)),"group_key":"file_name","official_test_used":False},"models":["logistic","ridge","hgb"],"groups":["F0","F1","F2","F3"],"budgets":{},"stratification":{}}
    for budget in (20,50):
      rows_by_img=[]; X={g:[] for g in out["groups"]}; y=[]; meta=[]
      for im in imgs:
        rr=[]
        for c in im["candidates"]:
          r=dict(c); key=f"{im['file_name']}|{tuple(r['pair'])[0]}-{tuple(r['pair'])[1]}"
          if key not in tmap: continue
          r["pair"]=tuple(r["pair"]); r["gt_pair"]=None if r.get("gt_pair") is None else tuple(r["gt_pair"]); r["local_scores"]=tmap[key]; rr.append(r)
        base=baseline_selection(rr,min(budget,len(rr))); rows_by_img.append(base)
        for r in base:
          fs, la, ga=feature(r,tmap[f"{im['file_name']}|{r['pair'][0]}-{r['pair'][1]}"],"F0",psg_by_id,im["image_id"])
          allf={g:feature(r,tmap[f"{im['file_name']}|{r['pair'][0]}-{r['pair'][1]}"],g,psg_by_id,im["image_id"])[0] for g in out["groups"]}
          for g in out["groups"]: X[g].append(allf[g])
          pred=la; u=exact_action_utility(im["relations"],r,pred,den,num_pred); y.append(float(u["delta_mr"])); meta.append((im,r,pred))
      y=np.asarray(y,float); groups_row=np.repeat(groups,[len(x) for x in rows_by_img]);
      budget_out={"oracle_headroom_mR_pp":float((3.86626737046139 if budget==20 else 4.492929046964589)),"models":{},"label":{"rows":len(y),"positive":int(np.sum(y>0)),"negative":int(np.sum(y<0))}}
      cv=GroupKFold(5); folds={name:{g:[] for g in out["groups"]} for name in out["models"]}
      for tr,te in cv.split(np.zeros(len(y)),y,groups_row):
        for name in out["models"]:
          for g in out["groups"]:
            target=(y[tr]>0).astype(int) if name=="logistic" else y[tr]
            if name == "logistic" and np.unique(target).size < 2:
              score=np.zeros(len(te), dtype=float); train_score=np.zeros(len(tr), dtype=float); thr=float("inf")
            else:
              est=make_estimator(name); est.fit(np.asarray(X[g])[tr], target); score=predict(est,name,np.asarray(X[g])[te]); train_score=predict(est,name,np.asarray(X[g])[tr]); thr=choose_threshold(train_score,y[tr])
            folds[name][g].append((te,score,thr))
      for name in out["models"]:
        budget_out["models"][name]={}
        for g in out["groups"]:
          pred_score=np.zeros(len(y)); chosen=np.zeros(len(y),bool); thresholds=[]
          for te,sc,thr in folds[name][g]: pred_score[te]=sc; chosen[te]=sc>=thr; thresholds.append(thr)
          masks=[]; off=0
          for rr in rows_by_img: masks.append(chosen[off:off+len(rr)]); off+=len(rr)
          res=evaluate_policy(imgs,rows_by_img,masks,num_pred,den,budget); res.update({"thresholds":thresholds,"threshold_median":float(np.median(thresholds)) if thresholds else None,"actionability_ratio":res["delta_mR_pp"]/budget_out["oracle_headroom_mR_pp"] if budget_out["oracle_headroom_mR_pp"] else None})
          budget_out["models"][name][g]=res
      out["budgets"][str(budget)]=budget_out
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))

if __name__=="__main__": main()
