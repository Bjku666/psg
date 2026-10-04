"""Conservative audit of file names explicitly present in prior artifacts."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

PAT=re.compile(r'(?:(?:train|val)2017)/\d{12}\.jpg')
def sha256(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--psg',type=Path,required=True); ap.add_argument('--repo',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 src=json.loads(a.psg.read_text()); test={str(x) for x in src.get('test_image_ids',[])}; rows={str(x['file_name']):x for x in src['data']}
 used={}; files=[]
 for root in (a.repo/'results',a.repo/'experiments',a.repo/'records'):
  if not root.exists(): continue
  for p in root.rglob('*'):
   if not p.is_file(): continue
   try: txt=p.read_text(errors='ignore')
   except OSError: continue
   hits=sorted(set(PAT.findall(txt)))
   if hits:
    files.append({'artifact':str(p),'sha256':sha256(p),'file_count':len(hits)})
    for fn in hits: used.setdefault(fn,[]).append(str(p))
 train=[fn for fn,row in rows.items() if str(row.get('image_id')) not in test]
 clean=[fn for fn in train if fn not in used]
 report={'schema_version':1,'name':'reference_stable_editing_v2_history_manifest','status':'conservative_artifact_scan_complete',
  'source':{'psg':str(a.psg),'dataset_sha256':sha256(a.psg)},
  'scan':{'roots':[str(a.repo/'results'),str(a.repo/'experiments'),str(a.repo/'records')],'artifacts_with_file_names':len(files),'note':'prediction caches and external command history are not self-describing; this is a lower-bound usage manifest.'},
  'population':{'unique_source_files':len(rows),'logical_train_files':len(train),'explicitly_referenced_files':len(used),'candidate_unused_train_files':len(clean)},
  'policy':{'official_test_quarantined':True,'clean_pool_authorized':False,'reason':'lower-bound scan cannot prove historical non-use'},
  'explicitly_referenced':{k:sorted(v) for k,v in sorted(used.items())},'artifacts':files}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report['population'],indent=2))
if __name__=='__main__': main()
