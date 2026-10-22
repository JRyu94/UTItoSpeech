"""All matched filtering contrasts; contains no additional-sentence experiment."""
from collections import defaultdict
import numpy as np
from .runtime import edit_distance,load_json
def analyse(path):
    data=load_json(path);out=[];checked=0
    draws=np.random.default_rng(20261021).integers(0,81,size=(5000,81))
    for run in data['public']['results']:
        entry={'seed':run['seed'],'jobs':run['jobs'],'direction':'primary conditioning minus public filtering'}
        for mode in ['test_modal','test_silent']:
            a={r['utt']:r for r in data['primary']['runs'][str(run['seed'])][mode]}
            b={r['utt']:r for r in run['test'][mode]['rows']};assert a.keys()==b.keys()
            counts=defaultdict(lambda:[0,0,0])
            for utt,x in a.items():
                y=b[utt];assert x['ref']==y['ref'] and x['words']==y['words']
                for row in [x,y]:
                    assert row['words']==len(row['ref'].split())
                    assert edit_distance(row['ref'],row['hyp'])==row['errors']
                    checked+=1
                c=counts[utt.split('_')[0]];c[0]+=x['errors'];c[1]+=y['errors'];c[2]+=x['words']
            assert len(counts)==81
            z=np.array([counts[s] for s in sorted(counts)],dtype=float);boot=z[draws].sum(1);e1,e2,n=z.sum(0)
            entry[mode]={'a_errors':int(e1),'b_errors':int(e2),'words':int(n),'a_wer':100*e1/n,'b_wer':100*e2/n,'delta_pp':100*(e1-e2)/n,'ci95':np.quantile(100*(boot[:,0]-boot[:,1])/boot[:,2],[.025,.975]).tolist(),'speakers':81,'utterances':len(a),'negative_speakers':int((z[:,0]<z[:,1]).sum()),'positive_speakers':int((z[:,0]>z[:,1]).sum()),'ties':int((z[:,0]==z[:,1]).sum())}
        out.append(entry)
    assert len(out)==3 and checked==14508
    return out,checked
