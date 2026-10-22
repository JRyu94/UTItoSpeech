"""Validated training-only absolute-discount bigram; not the corpus Witten-Bell LM."""
from pathlib import Path
from collections import Counter,defaultdict
import math
D=.75

def train(corpus,vocab):
 unigram=Counter({w:1. for w in sorted(vocab|{'</s>'})})
 bigram=defaultdict(Counter)
 for words in corpus:
  unigram.update(words+['</s>'])
  for a,b in zip(['<s>']+words,words+['</s>']):bigram[a][b]+=1
 total=sum(unigram.values());pu={w:c/total for w,c in unigram.items()};pb={};bo={};worst=0.
 for a,fs in bigram.items():
  n=sum(fs.values());pb[a]={b:(c-D)/n for b,c in fs.items()}
  denom=1-sum(pu[b] for b in fs);bo[a]=(D*len(fs)/n)/max(denom,1e-12)
  mass=sum(pb[a].values())+bo[a]*(1-sum(pu[b] for b in fs));worst=max(worst,abs(mass-1))
 assert worst<1e-9,worst
 return pu,pb,bo,worst

def arpa(path,pu,pb,bo):
 log=lambda x:math.log10(max(x,1e-12))
 words=sorted(pu);vocab=words+['<s>'];nb=sum(map(len,pb.values()))
 with path.open('w') as f:
  f.write('\\data\\\nngram 1=%d\nngram 2=%d\n\n\\1-grams:\n'%(len(vocab),nb))
  for w in words:f.write('%.6f\t%s\t%.6f\n'%(log(pu[w]),w,log(bo[w]) if w in bo else 0.))
  f.write('-99.000000\t<s>\t%.6f\n\n\\2-grams:\n'%(log(bo['<s>']) if '<s>' in bo else 0.))
  for a in vocab:
   for b,p in sorted(pb.get(a,{}).items()):f.write('%.6f\t%s %s\n'%(log(p),a,b))
  f.write('\n\\end\\\n')
