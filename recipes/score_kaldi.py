"""Select a word-LM weight on voiced development data and retain it for both test modes."""
from pathlib import Path
import argparse,hashlib,json,os,shlex,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lavu.runtime import edit_distance
def main():
    p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--kaldi-root',default=os.environ.get('KALDI_ROOT'));a=p.parse_args()
    if not a.kaldi_root:p.error('Set KALDI_ROOT or pass --kaldi-root')
    root=Path(a.recipe).resolve();binary=Path(a.kaldi_root)/'src/latbin/lattice-best-path';out=root/'evaluation';out.mkdir(exist_ok=True)
    words={int(i):w for w,i in (line.split() for line in (root/'data/lang/words.txt').read_text().splitlines())}
    def score(split,weight):
        lattice=root/'exp/sat'/('decode_'+split)
        rspec='ark:gunzip -c '+shlex.quote(str(lattice))+'/lat.*.gz|'
        result=subprocess.run([str(binary),'--lm-scale='+str(weight),rspec,'ark,t:-'],text=True,capture_output=True,check=True)
        hyp={v[0]:' '.join(words[int(i)] for i in v[1:] if words[int(i)] not in ['<UNK>','<unk>','!SIL']) for v in (line.split() for line in result.stdout.splitlines())}
        ref=dict(line.split(maxsplit=1) for line in (root/'data'/split/'text').read_text().splitlines());assert hyp.keys()==ref.keys()
        rows=[{'utt':u,'ref':ref[u],'hyp':hyp[u],'words':len(ref[u].split()),'errors':edit_distance(ref[u],hyp[u])} for u in sorted(ref)]
        e=sum(x['errors'] for x in rows);n=sum(x['words'] for x in rows)
        return {'split':split,'lmwt':weight,'errors':e,'words':n,'wer':e/n,'utterances':len(rows),'rows':rows}
    selection=out/'selection.json'
    if selection.exists():chosen=json.loads(selection.read_text())
    else:
        grid=[score('dev',w) for w in range(7,18)];best=min(grid,key=lambda x:(x['errors'],x['lmwt']));chosen={k:v for k,v in best.items() if k!='rows'}
        selection.write_text(json.dumps(chosen,indent=2)+'\n')
        (out/'dev_grid.json').write_text(json.dumps([{k:v for k,v in x.items() if k!='rows'} for x in grid],indent=2)+'\n')
    tests=[]
    for split in ['test_modal','test_silent']:
        x=score(split,chosen['lmwt']);(out/(split+'.json')).write_text(json.dumps(x,indent=2)+'\n');tests.append({k:v for k,v in x.items() if k!='rows'})
    (out/'summary.json').write_text(json.dumps({'dev':chosen,'test':tests},indent=2)+'\n')
if __name__=='__main__':main()
