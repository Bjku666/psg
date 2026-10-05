#!/usr/bin/env python3
"""Evaluate no-learning controls from a completed B0 local-score artifact."""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.relation_decision_regret_v1.official_metric_adapter import evaluate_population
from models.relation_decision_regret_v1.legal_oracle import baseline_selection

def main():
    p=argparse.ArgumentParser(); p.add_argument('--b0',type=Path,required=True); p.add_argument('--compact',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--seed',type=int,default=0); a=p.parse_args()
    b=json.loads(a.b0.read_text()); c=pickle.load(a.compact.open('rb')); scores=b['local_scores']; files={k.split('|',1)[0] for k in scores}; images=[x for x in c['images'] if x['file_name'] in files]; n=56
    out={'schema_version':1,'name':'bap_psg_v1_b0_controls','source':str(a.b0),'budgets':{}}
    rng=np.random.default_rng(a.seed)
    for K in (20,50):
      native=[]; local=[]; randoms=[]; trigger={t:[] for t in (0.05,0.1,0.2)}
      for im in images:
        rows=[]
        for x in im['candidates']:
          r=dict(x); r['pair']=tuple(r['pair']); r['gt_pair']=None if r.get('gt_pair') is None else tuple(r['gt_pair'])
          raw=scores.get(f"{im['file_name']}|{r['pair'][0]}-{r['pair'][1]}")
          if raw is None: continue
          s=np.asarray(raw,dtype=float)
          r['local_scores']=s; rows.append(r)
        base=baseline_selection(rows,min(K,len(rows)))
        for r in base:
          native_pred=int(np.argmax(np.asarray(r['pred_scores'])[1:]))
          r['pred']=native_pred
        upd=[]
        for r in base:
          q=dict(r); ls=np.asarray(r['local_scores']); q['pred']=int(np.argmax(ls[1:])); q['pred_score']=float(ls[q['pred']+1]); upd.append(q)
        for t in trigger:
          selected=[]
          for r in base:
            q=dict(r); ns=np.asarray(r['pred_scores']); ls=np.asarray(r['local_scores']); npred=int(np.argmax(ns[1:])); lpred=int(np.argmax(ls[1:]));
            order=np.sort(ls[1:]); margin=float(order[-1]-order[-2]) if len(order)>1 else 0.0
            if lpred != npred and margin >= t: q['pred']=lpred; q['pred_score']=float(ls[lpred+1])
            selected.append(q)
          trigger[t].append({**common,'selected':selected}) if False else None
        rr=[dict(r) for r in base]
        if rr:
          take=rng.choice(len(rr),size=max(1,len(rr)//2),replace=False)
          for j in take:
            ls=np.asarray(rr[j]['local_scores']); rr[j]['pred']=int(np.argmax(ls[1:])); rr[j]['pred_score']=float(ls[rr[j]['pred']+1])
        common={'image_id':im['image_id'],'file_name':im['file_name'],'bootstrap_group':im['file_name'],'relations':im['relations'],'budget':K}
        native.append({**common,'selected':base}); local.append({**common,'selected':upd}); randoms.append({**common,'selected':rr})
        # Recompute trigger rows after common is available.
        for t in trigger:
          selected=[]
          for r in base:
            q=dict(r); ns=np.asarray(r['pred_scores']); ls=np.asarray(r['local_scores']); npred=int(np.argmax(ns[1:])); lpred=int(np.argmax(ls[1:])); order=np.sort(ls[1:]); margin=float(order[-1]-order[-2]) if len(order)>1 else 0.0
            if lpred != npred and margin >= t: q['pred']=lpred; q['pred_score']=float(ls[lpred+1])
            selected.append(q)
          trigger[t].append({**common,'selected':selected})
      nm=evaluate_population(native,n); lm=evaluate_population(local,n); rm=evaluate_population(randoms,n)
      rec={'native_mR':nm[f'mR@{K}'],'local_argmax_mR':lm[f'mR@{K}'],'random_half_mR':rm[f'mR@{K}'],'local_delta_mR_pp':100*(lm[f'mR@{K}']-nm[f'mR@{K}']),'random_delta_mR_pp':100*(rm[f'mR@{K}']-nm[f'mR@{K}'])}
      rec['trigger_delta_mR_pp']={str(t):100*(evaluate_population(trigger[t],n)[f'mR@{K}']-nm[f'mR@{K}']) for t in trigger}
      out['budgets'][str(K)]=rec
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
