"""Compute paired-reference action-sign instability from a frozen raw carrier."""
from __future__ import annotations
import argparse, collections, hashlib, json, pickle
from pathlib import Path
import numpy as np

from models.relation_decision_regret_v1.official_metric_adapter import mapped_candidates, evaluate_image
from models.relation_decision_regret_v1.legal_oracle import baseline_selection, _predicate_options
from models.relation_decision_regret_v2.predicate_substitution_regret import apply_action, _native_pred
from models.counterfactual_relation_verification_v1.reference_ambiguity_audit import map_panoptic_entities

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def mr(x):
    a=np.asarray(x['per_predicate_recall'],float); return float(np.nanmean(a)) if np.isfinite(a).any() else 0.0

def utility(relations, base, row, pred, n):
    before=mr(evaluate_image(relations,base,n))
    after=mr(evaluate_image(relations,apply_action(base,row,pred),n))
    rb=float(evaluate_image(relations,base,n)['r']); ra=float(evaluate_image(relations,apply_action(base,row,pred),n)['r'])
    return after-before, ra-rb

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--psg',type=Path,required=True); ap.add_argument('--predictions',type=Path,required=True); ap.add_argument('--panoptic-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    a=ap.parse_args(); src=json.loads(a.psg.read_text());
    with a.predictions.open('rb') as f: preds={str(x['img_id']):x for x in pickle.load(f)}
    by=collections.defaultdict(list)
    for e in src['data']: by[e['file_name']].append(e)
    n=len(src['predicate_classes']); counts=collections.Counter(); actions=[]; groups=0
    for fn, rows in sorted(by.items()):
        if len(rows)!=2: continue
        ra, rb=rows; mapping,_,mode=map_panoptic_entities(ra,rb,a.panoptic_root)
        if mode!='mask' or len(mapping)!=len(ra.get('annotations',[])): continue
        inv={v:k for k,v in mapping.items()}; pa=preds.get(str(ra['image_id'])); pb=preds.get(str(rb['image_id']))
        if pa is None or pb is None: counts['missing_prediction_groups']+=1; continue
        ca,_=mapped_candidates(ra,pa,a.panoptic_root); cb,_=mapped_candidates(rb,pb,a.panoptic_root)
        # Canonicalize B's mapped GT pair into A's index space.
        for x in cb:
            if x.get('gt_pair') is not None: x['gt_pair']=tuple(inv[int(v)] for v in x['gt_pair'])
        # Keep fixed Top-50 support independently in each reference.
        ba=baseline_selection(ca,50); bb=baseline_selection(cb,50)
        bya={tuple(x['gt_pair']):x for x in ba if x.get('gt_pair') is not None}; byb={tuple(x['gt_pair']):x for x in bb if x.get('gt_pair') is not None}
        common=sorted(set(bya)&set(byb)); groups+=1
        rel_a=[tuple(map(int,x)) for x in ra.get('relations',[])]; rel_b=[]
        for s,o,p in rb.get('relations',[]):
            if int(s) in inv and int(o) in inv: rel_b.append((inv[int(s)],inv[int(o)],int(p)))
        for pair in common:
            xa,xb=bya[pair],byb[pair]; opts=sorted(set(_predicate_options(xa,3))|set(_predicate_options(xb,3)))
            na,nb=_native_pred(xa),_native_pred(xb)
            for pred in opts:
                if pred==na and pred==nb: continue
                ua,ra_delta=utility(rel_a,ba,xa,pred,n); ub,rb_delta=utility(rel_b,bb,xb,pred,n)
                sa=0 if abs(ua)<1e-12 else (1 if ua>0 else -1); sb=0 if abs(ub)<1e-12 else (1 if ub>0 else -1)
                if sa and sb: counts['nonneutral_actions']+=1; counts['sign_agree']+=int(sa==sb); counts['sign_unstable']+=int(sa!=sb)
                elif sa or sb: counts['one_neutral']+=1
                counts['actions_total']+=1
                actions.append({'file_name':fn,'pair':list(pair),'predicate':int(pred),'native_a':int(na),'native_b':int(nb),'u_a_mR':ua,'u_b_mR':ub,'u_a_R':ra_delta,'u_b_R':rb_delta,'sign_a':sa,'sign_b':sb})
    denom=counts['nonneutral_actions']; instability=counts['sign_unstable']/denom if denom else None
    report={'schema_version':1,'name':'reference_ambiguity_audit_v1_A4','status':'complete_epoch2_diagnostic','checkpoint':str(a.checkpoint),'checkpoint_sha256':sha256(a.checkpoint),'prediction_artifact':str(a.predictions),'prediction_sha256':sha256(a.predictions),'groups_evaluated':groups,'counts':dict(counts),'sign_instability':instability,'gate':('PROMOTE' if instability is not None and instability>=0.10 else 'KILL' if instability is not None and instability<0.02 else 'EXPAND_VALIDATION'),'note':'epoch-2 diagnostic; repeat with final pinned 40-epoch carrier before method promotion','actions':actions}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({k:report[k] for k in ('status','groups_evaluated','counts','sign_instability','gate')},indent=2))
if __name__=='__main__': main()
