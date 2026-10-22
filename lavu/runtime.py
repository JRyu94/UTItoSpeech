"""Portable inputs and caller-selected outputs for the frozen statistical analysis."""
from pathlib import Path
import gzip,json,os
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
def load_json(path):
    p=Path(path)
    payload=gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()
    return json.loads(payload)
def context():
    out=Path(os.environ.get('LAVU_OUTPUT',str(ROOT/'generated'))).resolve();out.mkdir(parents=True,exist_ok=True)
    bundle=Path(os.environ.get('LAVU_PREDICTIONS',str(ROOT/'data/predictions.json.gz'))).resolve()
    return out,bundle
def recipe_records():
    data=load_json(ROOT/'data/recipe_records.json.gz');files=dict(data['files'])
    for recipe,splits in data['memberships'].items():
        for split,ids in splits.items():
            prefix=recipe+'/data/'+split+'/';by_speaker=defaultdict(list)
            files[prefix+'text']=''.join(u+' '+data['utterances'][u]+'\n' for u in ids)
            files[prefix+'utt2spk']=''.join(u+' '+u.split('_',1)[0]+'\n' for u in ids)
            for u in ids:by_speaker[u.split('_',1)[0]].append(u)
            files[prefix+'spk2utt']=''.join(s+' '+' '.join(us)+'\n' for s,us in sorted(by_speaker.items()))
    return {'files':files,**data['metadata']}
def edit_distance(ref,hyp):
    a=ref.split();b=hyp.split();last=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):row.append(min(row[-1]+1,last[j]+1,last[j-1]+(x!=y)))
        last=row
    return last[-1]
