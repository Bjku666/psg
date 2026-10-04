"""Paired-reference robust-oracle gate on a frozen DSFormer carrier.

The oracle is diagnostic only: fixed mapped physical pair support, no new
pairs, and an edit is accepted only when its signed mR utility is positive in
both references of a physical-image group.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, pickle
from pathlib import Path
import numpy as np

from models.counterfactual_relation_verification_v1.reference_ambiguity_audit import map_panoptic_entities
from models.relation_decision_regret_v1.official_metric_adapter import mapped_candidates, evaluate_image
from models.relation_decision_regret_v1.legal_oracle import baseline_selection, _predicate_options
from models.relation_decision_regret_v2.predicate_substitution_regret import apply_action, _native_pred

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def mr(x):
    a=np.asarray(x['per_predicate_recall'],float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else 0.0

def util(rel, base, row, pred, n):
    b=evaluate_image(rel,base,n); a=evaluate_image(rel,apply_action(base,row,pred),n)
    return mr(a)-mr(b), float(a['r'])-float(b['r'])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--psg',type=Path,required=True); ap.add_argument('--predictions',type=Path,required=True); ap.add_argument('--panoptic-root',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); src=json.loads(a.psg.read_text()); n=len(src['predicate_classes'])
    with a.predictions.open('rb') as f: preds={str(x['img_id']):x for x in pickle.load(f)}
    by=collections.defaultdict(list)
    for e in src['data']: by[e['file_name']].append(e)
    base_rows=[]; robust_rows=[]; oracle_rows=[]; stats=collections.Counter(); groups=0
    for fn, pair in sorted(by.items()):
        if len(pair)!=2: continue
        ea, eb=pair; mapping,_,mode=map_panoptic_entities(ea,eb,a.panoptic_root)
        if mode!='mask' or len(mapping)!=len(ea.get('annotations',[])): continue
        inv={v:k for k,v in mapping.items()}; pa,pb=preds.get(str(ea['image_id'])),preds.get(str(eb['image_id']))
        if pa is None or pb is None: stats['missing_prediction_groups']+=1; continue
        ca,_=mapped_candidates(ea,pa,a.panoptic_root); cb,_=mapped_candidates(eb,pb,a.panoptic_root)
        for x in cb:
            if x.get('gt_pair') is not None: x['gt_pair']=tuple(inv[int(v)] for v in x['gt_pair'])
        ba,bb=baseline_selection(ca,50),baseline_selection(cb,50)
        aa={tuple(x['gt_pair']):x for x in ba if x.get('gt_pair') is not None}; ab={tuple(x['gt_pair']):x for x in bb if x.get('gt_pair') is not None}
        common=sorted(set(aa)&set(ab)); rel_a=[tuple(map(int,x)) for x in ea.get('relations',[])]; rel_b=[(inv[int(s)],inv[int(o)],int(p)) for s,o,p in eb.get('relations',[]) if int(s) in inv and int(o) in inv]
        robust_a=list(ba); robust_b=list(bb); oracle_a=list(ba); oracle_b=list(bb)
        base_rows += [{'relations':rel_a,'selected':ba,'file_name':fn,'image_id':str(ea['image_id']),'bootstrap_group':fn},{'relations':rel_b,'selected':bb,'file_name':fn,'image_id':str(eb['image_id']),'bootstrap_group':fn}]
        for key in common:
            xa,xb=aa[key],ab[key]; native_a,native_b=_native_pred(xa),_native_pred(xb)
            options=sorted(set(_predicate_options(xa,3))|set(_predicate_options(xb,3)))
            # Choose the strongest action that is positive in both references.
            # Utilities are measured against each reference's untouched
            # baseline, so the accepted support is fixed and deterministic;
            # the final aggregate is then evaluated after all accepted edits.
            common_best = None
            for pred in options:
                ua,ra=util(rel_a,ba,xa,pred,n); ub,rb=util(rel_b,bb,xb,pred,n)
                if ua>1e-12 and ub>1e-12:
                    score = (min(ua, ub), ua + ub, -int(pred))
                    if common_best is None or score > common_best[0]:
                        common_best = (score, pred, ua, ub)
            if common_best is not None:
                pred = int(common_best[1])
                robust_a = apply_action(robust_a, xa, pred)
                robust_b = apply_action(robust_b, xb, pred)
                stats['stable_positive_actions'] += 1
            # Per-reference exact oracle, for headroom only.
            for x,rel,base,out in ((xa,rel_a,ba,oracle_a),(xb,rel_b,bb,oracle_b)):
                best=None
                for pred in options:
                    u,r=util(rel,base,x,pred,n)
                    if best is None or u>best[0]: best=(u,r,pred)
                if best and best[0]>1e-12:
                    if x is xa: oracle_a=apply_action(oracle_a,x,best[2])
                    else: oracle_b=apply_action(oracle_b,x,best[2])
        robust_rows += [{'relations':rel_a,'selected':robust_a,'file_name':fn,'image_id':str(ea['image_id']),'bootstrap_group':fn},{'relations':rel_b,'selected':robust_b,'file_name':fn,'image_id':str(eb['image_id']),'bootstrap_group':fn}]
        oracle_rows += [{'relations':rel_a,'selected':oracle_a,'file_name':fn,'image_id':str(ea['image_id']),'bootstrap_group':fn},{'relations':rel_b,'selected':oracle_b,'file_name':fn,'image_id':str(eb['image_id']),'bootstrap_group':fn}]
        groups+=1
    def agg(rows):
        x=[]
        for r in rows:
            q=evaluate_image(r['relations'],r['selected'],n); x.append(q)
        m=np.stack([q['per_predicate_recall'] for q in x]); pm=np.array([np.nanmean(m[:,i]) if np.isfinite(m[:,i]).any() else np.nan for i in range(n)])
        return {'mR':float(np.nanmean(pm)),'R':float(np.nanmean([q['r'] for q in x])),'rows':len(x)}
    b,r,o=agg(base_rows),agg(robust_rows),agg(oracle_rows)
    report={'schema_version':1,'name':'reference_ambiguity_audit_v1_robust_oracle','status':'complete','checkpoint':str(a.checkpoint),'checkpoint_sha256':sha256(a.checkpoint),'prediction_sha256':sha256(a.predictions),'groups_evaluated':groups,'stats':dict(stats),'baseline':b,'robust_oracle':r,'exact_paired_oracle':o,'gains_pp':{'robust_mR':100*(r['mR']-b['mR']),'robust_R':100*(r['R']-b['R']),'exact_mR':100*(o['mR']-b['mR'])},'gate':'PROMOTE_ORACLE' if 100*(r['mR']-b['mR'])>=5 else 'KILL_ORACLE_HEADROOM'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
