from pathlib import Path
from collections import defaultdict
import json,math
import numpy as np
from lavu.runtime import context, load_json
H, BUNDLE = context();data=load_json(BUNDLE);results=data['results']
sub=json.loads((H/'subset_analysis.json').read_text());texts=sub['method']['repeated_texts']
speakers=sorted({r['utt'].split('_')[0] for r in results['shared_32_DNN+fMLLR']['splits']['test_modal']['rows']})
draws=np.random.default_rng(20261021).integers(0,81,size=(5000,81))
def rows(keys,mode):return [r for k in keys for r in results[k]['splits'][mode]['rows']]
def aggregate(keys,mode,pick=lambda r:True):
    d=defaultdict(lambda:[0,0,0])
    for r in rows(keys,mode):
        if pick(r):
            x=d[r['utt'].split('_')[0]];x[0]+=r['errors'];x[1]+=r['words'];x[2]+=1
    return np.array([d[s] for s in speakers],dtype=float)
def compare(a,b):
    assert np.array_equal(a[:,1:],b[:,1:]);n=a[:,1].sum();x=a[draws].sum(1);y=b[draws].sum(1)
    boot=100*(x[:,0]-y[:,0])/x[:,1];diff=a[:,0]-b[:,0];minus=int((diff<0).sum());plus=int((diff>0).sum());nt=minus+plus
    p=min(1.,2*sum(math.comb(nt,k) for k in range(min(minus,plus)+1))/2**nt) if nt else 1.
    da=a.sum(0)-a;db=b.sum(0)-b
    loo=100*((da[:,0]-db[:,0])/da[:,1])
    return {'a_wer':100*a[:,0].sum()/n,'b_wer':100*b[:,0].sum()/n,'errors_a':int(a[:,0].sum()),'errors_b':int(b[:,0].sum()),'words':int(n),'utterances':int(a[:,2].sum()),
            'delta_pp':100*(a[:,0].sum()-b[:,0].sum())/n,'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'negative_speakers':minus,'positive_speakers':plus,'ties':81-nt,'sign_p':p,
            'speaker_loo_range':[float(loo.min()),float(loo.max())]},boot
def corpus_keys(protocol,units,model='DNN+fMLLR'):
    return [f'shared_{units}_{model}'] if protocol=='shared' else [f'disjoint_{units}_{model}_fold{i}' for i in range(9)]
def lmkeys(protocol,units,seed=1):
    return [f'trainlm_shared{units}'+(f'_s{seed}' if seed!=1 else '')] if protocol=='shared' else [f'trainlm_fold{i}_{units}' for i in range(9)]
out={'lm_comparisons':[],'sentence_robustness':{},'shared_vs_disjoint':{},'repair_vs_smoothing':[]}
out['repair_primary']={}
for model in ['GMM+fMLLR','DNN+fMLLR']:
    out['repair_primary'][model]={}
    for mode in ['test_modal','test_silent']:
        out['repair_primary'][model][mode],_=compare(aggregate([f'factor_f_clean_ours_g_{model}_s1'],mode),aggregate([f'factor_f_smooth_5tap_g_{model}_s1'],mode))
for protocol in ['shared','disjoint']:
    a=corpus_keys(protocol,32);b=corpus_keys(protocol,40)
    rep=lambda r:r['ref'] in texts;specific=lambda r:r['ref'] not in texts
    pair={}
    for subset,pick in [('all',lambda r:True),('repeated',rep),('specific',specific)]:pair[subset],_=compare(aggregate(a,'test_silent',pick),aggregate(b,'test_silent',pick))
    _,specboot=compare(aggregate(a,'test_silent',specific),aggregate(b,'test_silent',specific))
    pair['sentences']=[]
    for text in texts:
        pick=lambda r:r['ref']==text
        ra=rows(a,'test_silent');rb={r['utt']:r for r in rows(b,'test_silent')}
        chosen=[r for r in ra if pick(r)];e1=sum(r['errors'] for r in chosen);e2=sum(rb[r['utt']]['errors'] for r in chosen);n=sum(r['words'] for r in chosen)
        keep=lambda r:r['ref'] in texts and r['ref']!=text
        without,boot=compare(aggregate(a,'test_silent',keep),aggregate(b,'test_silent',keep))
        pair['sentences'].append({'text':text,'errors32':e1,'errors40':e2,'delta_errors':e1-e2,'words':n,'delta_pp':100*(e1-e2)/n,
            'other_five':without,'specific_minus_other_five':{'delta_pp':pair['specific']['delta_pp']-without['delta_pp'],'ci95':np.quantile(specboot-boot,[.025,.975]).tolist()}})
    out['sentence_robustness'][protocol]=pair
    for mode in ['test_modal','test_silent']:
        entry={'protocol':protocol,'mode':mode}
        entry['inventory_effect'],_=compare(aggregate(lmkeys(protocol,32),mode),aggregate(lmkeys(protocol,40),mode))
        for units in [32,40]:
            entry[str(units)],_=compare(aggregate(lmkeys(protocol,units),mode),aggregate(corpus_keys(protocol,units),mode))
        out['lm_comparisons'].append(entry)
for mode in ['test_modal','test_silent']:
    out['shared_vs_disjoint'][mode],_=compare(aggregate(corpus_keys('disjoint',32),mode),aggregate(corpus_keys('shared',32),mode))
for seed in [1,2,3]:
    e={'seed':seed}
    for mode in ['test_modal','test_silent']:
        e[mode],_=compare(aggregate([f'factor_f_clean_ours_g_DNN+fMLLR_s{seed}'],mode),aggregate([f'factor_f_smooth_5tap_g_DNN+fMLLR_s{seed}'],mode))
    out['repair_vs_smoothing'].append(e)
out['trainlm_replicates']=[]
for seed in [1,2,3]:
    e={'seed':seed,'jobs':8 if seed==1 else 6}
    for mode in ['test_modal','test_silent']:e[mode],_=compare(aggregate(lmkeys('shared',32,seed),mode),aggregate(lmkeys('shared',40,seed),mode))
    out['trainlm_replicates'].append(e)
(H/'extended_analysis.json').write_text(json.dumps(out,indent=2),encoding='utf8')
for protocol,pair in out['sentence_robustness'].items():print(protocol,{k:pair[k]['delta_pp'] for k in ['all','repeated','specific']})
print('LM',[(e['protocol'],e['mode'],e['32']['a_wer'],e['40']['a_wer']) for e in out['lm_comparisons']])
print('REPAIR',[(e['seed'],e['test_silent']['delta_pp']) for e in out['repair_vs_smoothing']])
