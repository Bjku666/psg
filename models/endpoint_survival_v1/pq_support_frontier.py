#!/usr/bin/env python3
"""Aggregate semantic/competition margin outputs into a PQ-support frontier."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--inputs',nargs='+',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--baseline',type=Path)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    points=[]
    for f in a.inputs:
        d=json.loads(f.read_text()); m=d['metrics']; points.append({'mode':d.get('mode'),'margin':d.get('margin'),'endpoint_support':m['endpoint_support'],'predicate_balanced_endpoint_support':m['predicate_balanced_endpoint_support'],'pq':m['pq'],'pq_th':m.get('pq_th',0.),'pq_st':m.get('pq_st',0.),'r@20':m.get('r@20',0.),'r@50':m.get('r@50',0.),'mr@20':m.get('mr@20',0.),'mr@50':m.get('mr@50',0.),'source':str(f)})
    base={'endpoint_support':0.,'predicate_balanced_endpoint_support':0.,'pq':0.}
    if a.baseline:
        d=json.loads(a.baseline.read_text()); base.update({k:d.get('metrics',d).get(k,base[k]) for k in base})
    for x in points:
        x['delta_endpoint_support']=x['endpoint_support']-base['endpoint_support']; x['delta_predicate_balanced_support']=x['predicate_balanced_endpoint_support']-base['predicate_balanced_endpoint_support']; x['delta_pq']=x['pq']-base['pq']; x['support_gate_passed']=x['delta_predicate_balanced_support']>=.03; x['pq_gate_passed']=x['delta_pq']>=-.01
    best=max(points,key=lambda x:x['delta_predicate_balanced_support'],default=None)
    out={'schema_version':1,'contract':'structured oracle PQ-vs-endpoint-support Pareto frontier','baseline':base,'points':points,'best_support_point':best,'gates':{'endpoint_support_delta_min':.03,'pq_delta_min':-.01,'structured_oracle_passed':bool(best and best['support_gate_passed'] and best['pq_gate_passed'])},'note':'PQ is normalized to [0,1], therefore the -0.01 gate corresponds to -1 percentage point.'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
