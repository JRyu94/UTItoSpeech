"""Highest-probability MFA pronunciation selection, including deterministic ties."""
from collections import defaultdict
from pathlib import Path
import re
MERGE={p:m for a,b,m in [('P','B','vP'),('T','D','vT'),('K','G','vK'),('F','V','vF'),('TH','DH','vTH'),('S','Z','vS'),('SH','ZH','vSH'),('CH','JH','vCH')] for p in [a,b]}
def mfa_best(path):
    variants=defaultdict(list)
    for line in Path(path).read_text(encoding='utf8').splitlines():
        fields=line.split('\t')
        if len(fields)<2:continue
        probability=float(fields[1]) if len(fields)>2 else 1.
        phones=[re.sub(r'\d','',p).upper() for p in fields[-1].split()]
        phones=[p for p in phones if p.isalpha()]
        if phones:variants[fields[0].lower()].append((probability,phones))
    return {word:max(v,key=lambda x:x[0])[1] for word,v in variants.items()}
def corrected_lexicon(dictionary,fallback,output,units):
    best=mfa_best(dictionary);out=[]
    for line in Path(fallback).read_text(encoding='utf8').splitlines():
        word,*phones=line.split()
        if word in best:phones=best[word]
        if units==32:phones=[MERGE.get(p,p) for p in phones]
        out.append(' '.join([word]+phones))
    Path(output).write_text('\n'.join(out)+'\n',encoding='utf8')
