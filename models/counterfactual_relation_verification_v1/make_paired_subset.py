"""Create a non-destructive annotation subset for the 241 paired references."""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--psg',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); src=json.loads(a.psg.read_text()); by=collections.defaultdict(list)
    for row in src['data']: by[row['file_name']].append(row)
    rows=[r for name in sorted(by) if len(by[name])==2 for r in by[name]]
    out={k:v for k,v in src.items() if k not in ('data','test_image_ids')}
    out['data']=rows; out['test_image_ids']=[r['image_id'] for r in rows]
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out))
    print(json.dumps({'rows':len(rows),'groups':len(rows)//2,'output':str(a.output)}))
if __name__=='__main__': main()
