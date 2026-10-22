"""Inventory effects across the existing LM conditions and lexical error diagnostics.

All inference is conditional on fitted models, the observed texts and the
complete LM/adaptation recipe. No model is trained or decoded here.
"""
from collections import Counter,defaultdict
from .runtime import load_json,edit_distance
import numpy as np

def align_words(reference,hypothesis):
    """Minimum word-edit path; traceback priority match, substitution, deletion, insertion."""
    a=reference.split();b=hypothesis.split();d=np.zeros((len(a)+1,len(b)+1),dtype=int)
    d[:,0]=np.arange(len(a)+1);d[0,:]=np.arange(len(b)+1)
    for i,x in enumerate(a,1):
        for j,y in enumerate(b,1):d[i,j]=min(d[i-1,j-1]+(x!=y),d[i-1,j]+1,d[i,j-1]+1)
    out=[];i=len(a);j=len(b)
    while i or j:
        if i and j and a[i-1]==b[j-1] and d[i,j]==d[i-1,j-1]:op='C';i-=1;j-=1;out.append((op,a[i],b[j]))
        elif i and j and d[i,j]==d[i-1,j-1]+1:op='S';i-=1;j-=1;out.append((op,a[i],b[j]))
        elif i and d[i,j]==d[i-1,j]+1:op='D';i-=1;out.append((op,a[i],None))
        else:assert j and d[i,j]==d[i,j-1]+1;j-=1;out.append(('I',None,b[j]))
    out.reverse();assert sum(op!='C' for op,_,_ in out)==edit_distance(reference,hypothesis)
    return out

def lexical_groups(data):
    entries={w:(a,b) for w,a,b in data['entries']};assert len(entries)==len(data['entries'])
    groups40=defaultdict(set);groups32=defaultdict(set)
    for w,(a,b) in entries.items():groups40[a].add(w);groups32[b].add(w)
    for w,(a,b) in entries.items():assert groups40[a]<=groups32[b],'32-unit classes must coarsen 40-phone classes'
    newly={w for w,(a,b) in entries.items() if len(groups40[a])==1 and len(groups32[b])>1}
    expanded={w for w,(a,b) in entries.items() if groups40[a]!=groups32[b]}
    stats={'lexicon_entries':len(entries),'lexical_words':len(set(entries)-{'!SIL','<UNK>'}),
           'homophone_classes_40':sum(len(g)>1 for g in groups40.values()),'homophone_classes_32':sum(len(g)>1 for g in groups32.values()),
           'previously_unique_entries_becoming_homophonous':len(newly),'entries_with_expanded_competitor_sets':len(expanded),
           'already_homophonous_entries_with_added_competitors':len(expanded-newly)}
    return entries,newly,stats

def analyse(bundle_path,relations_path):
    data=load_json(bundle_path)['results'];relations=load_json(relations_path)
    entries,newly,lexical=lexical_groups(relations)
    ref=data['shared_32_DNN+fMLLR']['splits']['test_modal']['rows'];freq=Counter(r['ref'] for r in ref)
    repeated={s for s,c in freq.most_common(6)};assert len(repeated)==6 and min(freq[s] for s in repeated)>=80
    speakers=sorted({r['utt'].split('_')[0] for r in ref});assert len(speakers)==81
    draws=np.random.default_rng(20261021).integers(0,len(speakers),size=(5000,len(speakers)))
    subsets=['all','repeated','specific']
    def pick(r,s):return s=='all' or (r['ref'] in repeated)==(s=='repeated')
    def get(protocol,lm,units,run,mode):
        if protocol=='shared':
            k=(f'shared_{units}_DNN+fMLLR' if run==1 else f'shared_{units}_DNN_s{run}') if lm=='corpus' else f'trainlm_shared{units}'+(f'_s{run}' if run!=1 else '')
            keys=[k]
        else:
            assert run==1;keys=[f'disjoint_{units}_DNN+fMLLR_fold{i}' if lm=='corpus' else f'trainlm_fold{i}_{units}' for i in range(9)]
        rs=[r for k in keys for r in data[k]['splits'][mode]['rows']]
        assert len(rs)==len({r['utt'] for r in rs})==1209
        return rs
    def matrix(rs,s):
        d=defaultdict(lambda:[0,0,0])
        for r in rs:
            if pick(r,s):
                x=d[r['utt'].split('_')[0]];x[0]+=r['errors'];x[1]+=r['words'];x[2]+=1
        a=np.array([d[p] for p in speakers],dtype=float);assert (a[:,1]>0).all();return a
    def compare(a,b):
        assert np.array_equal(a[:,1:],b[:,1:]);da=a.sum(0);db=b.sum(0)
        x=a[draws].sum(1);y=b[draws].sum(1);boot=100*(x[:,0]-y[:,0])/x[:,1]
        return {'errors32':int(da[0]),'errors40':int(db[0]),'words':int(da[1]),'utterances':int(da[2]),'wer32':100*da[0]/da[1],'wer40':100*db[0]/db[1],
                'delta_errors':int(da[0]-db[0]),'delta_pp':100*(da[0]-db[0])/da[1],'ci95':np.quantile(boot,[.025,.975]).tolist()},boot
    out={'method':{'speakers':81,'draws':5000,'seed':20261021,'scope':'Exploratory fixed-model, fixed-text paired speaker analysis; LM condition includes smoothing and first-pass fMLLR hypotheses.',
                   'word_alignment_tie_policy':'Traceback prioritizes exact match, substitution, deletion, insertion. Word-error categories are descriptive and may depend on the optimal-path tie policy.'},
         'conditions':[],'lm_interaction_changes':[],'lexical':lexical,'word_error_diagnostics':[],'examples':[]}
    cache={}
    for protocol in ['shared','disjoint']:
        for run in ([1,2,3] if protocol=='shared' else [1]):
            for mode in ['test_modal','test_silent']:
                for lm in ['corpus','training_only']:
                    a=get(protocol,lm,32,run,mode);b=get(protocol,lm,40,run,mode)
                    canonical=lambda rs:{r['utt']:(r['ref'],r['words']) for r in rs}
                    assert canonical(a)==canonical(b)
                    e={'protocol':protocol,'run':run,'jobs':8 if run==1 else 6,'mode':mode,'lm':lm,'subsets':{}};bs={}
                    for s in subsets:e['subsets'][s],bs[s]=compare(matrix(a,s),matrix(b,s))
                    interaction=bs['specific']-bs['repeated'];delta=e['subsets']['specific']['delta_pp']-e['subsets']['repeated']['delta_pp']
                    e['interaction']={'delta_pp':delta,'ci95':np.quantile(interaction,[.025,.975]).tolist()}
                    out['conditions'].append(e);cache[(protocol,run,mode,lm)]=(delta,interaction)
                x,xb=cache[(protocol,run,mode,'corpus')];y,yb=cache[(protocol,run,mode,'training_only')]
                out['lm_interaction_changes'].append({'protocol':protocol,'run':run,'jobs':8 if run==1 else 6,'mode':mode,'direction':'corpus condition minus training-only condition',
                    'delta_pp':x-y,'ci95':np.quantile(xb-yb,[.025,.975]).tolist()})
    for s in subsets:
        rs=[r for r in ref if pick(r,s)];tokens=[w for r in rs for w in r['ref'].split()]
        assert set(tokens)<=entries.keys();n=sum(w in newly for w in tokens)
        lexical[s]={'reference_tokens':len(tokens),'newly_ambiguous_reference_tokens':n,'coverage_percent':100*n/len(tokens),
                    'utterances':len(rs),'distinct_texts':len({r['ref'] for r in rs})}
    def is_new_homophone(a,b):return a in entries and b in entries and a!=b and entries[a][1]==entries[b][1] and entries[a][0]!=entries[b][0]
    def error_types(row):
        aligned=align_words(row['ref'],row['hyp']);c=Counter(op for op,_,_ in aligned if op!='C')
        assert sum(c.values())==row['errors']
        return {'substitutions':c['S'],'deletions':c['D'],'insertions':c['I'],'new_homophone_substitutions':sum(op=='S' and is_new_homophone(a,b) for op,a,b in aligned)}
    candidates=[]
    for protocol in ['shared','disjoint']:
        pairs={u['utt']:{'40':u} for u in get(protocol,'corpus',40,1,'test_silent')}
        for row in get(protocol,'corpus',32,1,'test_silent'):pairs[row['utt']]['32']=row
        for p in pairs.values():
            for u in ['40','32']:p[u+'_types']=error_types(p[u])
        for s in subsets:
            chosen=[p for p in pairs.values() if pick(p['32'],s)]
            e={'protocol':protocol,'subset':s,'utterances':len(chosen),'words':sum(p['32']['words'] for p in chosen)}
            for u in ['40','32']:
                e[u]={k:sum(p[u+'_types'][k] for p in chosen) for k in ['substitutions','deletions','insertions','new_homophone_substitutions']}
                e[u]['errors']=sum(e[u][k] for k in ['substitutions','deletions','insertions'])
                e[u]['other_substitutions']=e[u]['substitutions']-e[u]['new_homophone_substitutions']
            e['delta32minus40']={k:e['32'][k]-e['40'][k] for k in e['32']}
            out['word_error_diagnostics'].append(e)
        if protocol=='disjoint':
            for p in pairs.values():
                r=p['32'];candidates.append({'utt':r['utt'],'reference':r['ref'],'hyp40':p['40']['hyp'],'hyp32':r['hyp'],'words':r['words'],
                    'subset':'repeated' if r['ref'] in repeated else 'specific','errors40':p['40']['errors'],'errors32':r['errors'],'delta_errors':r['errors']-p['40']['errors'],
                    'new_homophone40':p['40_types']['new_homophone_substitutions'],'new_homophone32':p['32_types']['new_homophone_substitutions']})
    # Deterministic post hoc display rules illustrate signs of change, not an effect estimate.
    cost=[p for p in candidates if p['subset']=='repeated' and p['delta_errors']>0 and p['new_homophone32']>p['new_homophone40']]
    if not cost:cost=[p for p in candidates if p['subset']=='repeated' and p['delta_errors']>0]
    cost=sorted(cost,key=lambda p:(-p['delta_errors'],p['utt']))[0]
    benefit=sorted([p for p in candidates if p['subset']=='specific' and p['delta_errors']<0 and p['words']<=12],key=lambda p:(p['delta_errors'],p['utt']))[0]
    out['examples']=[cost,benefit]
    out['example_selection']='Disjoint corpus-LM silent run 1: largest repeated-sentence error increase with an added newly homophonous substitution if available; largest specific-sentence error decrease among references of at most 12 words; utterance ID breaks ties. Illustrative selection, not a separate test.'
    out['validation']={'paired_references_match':True,'word_alignments_checked':2*2*1209,'lexicon_source_sha256':relations['source_sha256']}
    return out
