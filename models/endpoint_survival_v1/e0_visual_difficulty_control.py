#!/usr/bin/env python3
"""E0: image-within visual-difficulty matched control for endpoint survival.

Each GT entity is represented once.  Relation criticality is predicate-balanced
relation mass; visual covariates are measured from the best semantic query
matched to that entity.  Controls are nearest-neighbour matched within image on
area, thing/stuff, category, mask IoU, class score, mask quality and raw query
rank.  A logistic model then estimates the residual graph-criticality effect.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from models.endpoint_survival_v1.common import select_items, load_records, artifact_path, criticality
from models.gssr_p0b_v1.mask_matcher import load_index_mask, single_mpo_binary_mask_mapping
from models.gssr_p0c_v1.raw_query_schema import decode_binary_mask

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument('--psg',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--gt-seg-root',type=Path,required=True); p.add_argument('--stage-audit',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--population-manifest',type=Path); p.add_argument('--max-images',type=int); p.add_argument('--iou-threshold',type=float,default=.5)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    psg=json.loads(a.psg.read_text()); records=load_records(a.manifest); items={str(x['image_id']):x for x in select_items(psg,a.population_manifest,a.max_images)}
    audit={}
    for line in a.stage_audit.read_text().splitlines():
        if line.strip():
            r=json.loads(line); audit[str(r['image_id'])]=r
    freq=np.zeros(len(psg['predicate_classes']),dtype=int)
    for x in items.values():
        for _,_,q in np.asarray(x['relations'],dtype=int).reshape(-1,3): freq[int(q)]+=1
    rows=[]
    for iid,item in items.items():
        rec=records.get(str(item['file_name'])); ar=audit.get(iid)
        if rec is None or ar is None: continue
        gt=load_index_mask(a.gt_seg_root/item['pan_seg_file_name'],item['segments_info']); labels=np.asarray([x['category_id'] for x in item['annotations']],int)
        masks=np.stack([decode_binary_mask(q['mask_rle']) for q in rec['queries']]); qlabels=np.asarray([q['predicted_class'] for q in rec['queries']],int)
        mapping=single_mpo_binary_mask_mapping(gt,masks,labels,qlabels,a.iou_threshold)
        c=criticality(np.asarray(item['relations']),len(labels),freq)
        endpoint_rows={int(e['gt_index']):e for e in ar['endpoints']}
        for gi,gl in enumerate(labels):
            e=endpoint_rows.get(gi,{}); qids=e.get('semantic_queries',[])
            candidates=[int(q) for q in qids if 0<=int(q)<len(rec['queries'])]
            q=min(candidates,key=lambda q: -float(rec['queries'][q]['joint_score'])) if candidates else None
            area=float((gt==gi).sum())/(gt.shape[0]*gt.shape[1]); feat={
                'image_id':iid,'gt_index':gi,'relation_criticality':float(c[gi]),'relation_critical':bool(c[gi]>0),
                'survived':int(bool(e.get('admission_queries'))),'semantic_survived':int(bool(e.get('semantic_queries'))),
                'area':area,'thing':int(int(gl)<80),'category':int(gl),'mask_iou':float(mapping.candidate_iou[q]) if q is not None and int(mapping.candidate_to_gt[q])==gi else 0.,
                'class_score':float(rec['queries'][q]['class_score']) if q is not None else 0.,'mask_quality':float(rec['queries'][q]['mask_quality']) if q is not None else 0.,
                'query_rank':int(q) if q is not None else 200}
            rows.append(feat)
    if not rows: raise ValueError('no endpoint rows')
    cov=['area','thing','category','mask_iou','class_score','mask_quality','query_rank']; X=np.asarray([[r[k] for k in cov] for r in rows],float); y=np.asarray([r['survived'] for r in rows],int)
    Xs=StandardScaler().fit_transform(X); model=LogisticRegression(max_iter=2000,class_weight='balanced').fit(np.c_[Xs,np.asarray([r['relation_criticality'] for r in rows])],y)
    beta=float(model.coef_[0,-1]); odds=float(np.exp(beta));
    # Greedy image-within nearest matching, one control per critical endpoint.
    matched=[]
    by_img={}
    for i,r in enumerate(rows): by_img.setdefault(r['image_id'],[]).append(i)
    for iid,idxs in by_img.items():
        crit=[i for i in idxs if rows[i]['relation_critical']]; ctrl=[i for i in idxs if not rows[i]['relation_critical']]; used=set()
        for i in crit:
            pool=[j for j in ctrl if j not in used]
            if not pool: continue
            d=[np.linalg.norm(Xs[i]-Xs[j]) for j in pool]; j=pool[int(np.argmin(d))]; used.add(j); matched.append((i,j,min(d)))
    deltas=[rows[i]['survived']-rows[j]['survived'] for i,j,_ in matched]
    out={'schema_version':1,'contract':'image-within visual-difficulty matched control + logistic residual criticality','images':len(by_img),'entities':len(rows),'matched_pairs':len(matched),'covariates':cov,'logistic':{'beta_graph_criticality':beta,'odds_ratio':odds,'intercept':float(model.intercept_[0])},'matched_survival_delta_mean':float(np.mean(deltas)) if deltas else None,'matched_survival_delta_median':float(np.median(deltas)) if deltas else None,'relation_survival':float(np.mean([r['survived'] for r in rows if r['relation_critical']])),'control_survival':float(np.mean([r['survived'] for r in rows if not r['relation_critical']])),'rows':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__': main()
