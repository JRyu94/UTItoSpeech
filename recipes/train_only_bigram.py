"""Build the exact training-only LM estimator used in the reported control."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lavu.language_model import train,arpa
def main():
    p=argparse.ArgumentParser();p.add_argument('--text',required=True);p.add_argument('--lexicon',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    lines=dict(line.split(maxsplit=1) for line in Path(a.text).read_text().splitlines())
    corpus=[lines[u].split() for u in sorted(lines)]
    vocab={line.split()[0] for line in Path(a.lexicon).read_text().splitlines()}-{'<UNK>','!SIL'}
    if any(w not in vocab for row in corpus for w in row):p.error('Training text has words outside the retained lexicon')
    pu,pb,bo,error=train(corpus,vocab);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);arpa(out,pu,pb,bo)
    out.with_suffix('.json').write_text(json.dumps({'lexical_vocabulary':len(vocab),'absolute_discount':.75,'unigram_prior':1.,'max_mass_error':error,'training_utterances':len(corpus)},indent=2)+'\n')
if __name__=='__main__':main()
