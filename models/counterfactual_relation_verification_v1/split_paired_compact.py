from pathlib import Path
import argparse, pickle

ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--fit',type=Path,required=True); ap.add_argument('--dev',type=Path,required=True)
a=ap.parse_args()
with a.input.open('rb') as f: doc=pickle.load(f)
groups={}
for row in doc['images']: groups.setdefault(row['bootstrap_group'], []).append(row)
names=sorted(groups)
cut=len(names)//2
def out(path, nameset):
 d=dict(doc); d['images']=[r for n in nameset for r in groups[n]]; d['missing_predictions']=[]
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('wb') as f: pickle.dump(d,f,pickle.HIGHEST_PROTOCOL)
out(a.fit,names[:cut]); out(a.dev,names[cut:])
print({'fit_groups':cut,'dev_groups':len(names)-cut,'fit_rows':2*cut,'dev_rows':2*(len(names)-cut)})
