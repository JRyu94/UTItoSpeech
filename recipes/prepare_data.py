"""Materialize paper split membership with caller-supplied Kaldi feature references."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lavu.runtime import recipe_records
def main():
    p=argparse.ArgumentParser();p.add_argument('--record-recipe',required=True,help='e.g. bfeat_group, bfeat_full, xspk/fold0, xspk40/fold0');p.add_argument('--features-scp',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    features=dict(line.split(maxsplit=1) for line in Path(a.features_scp).read_text().splitlines())
    records=recipe_records()['files'];out=Path(a.output).resolve()
    for split in ['train','dev','test_modal','test_silent']:
        prefix=a.record_recipe+'/data/'+split+'/'
        if prefix+'text' not in records:p.error('Unknown record recipe: '+a.record_recipe)
        text=records[prefix+'text'];ids=[line.split()[0] for line in text.splitlines()]
        if any(u not in features for u in ids):p.error('Missing feature references for '+split)
        target=out/'data'/split;target.mkdir(parents=True,exist_ok=True)
        for name in ['text','utt2spk','spk2utt']:(target/name).write_text(records[prefix+name],encoding='utf8')
        (target/'feats.scp').write_text(''.join(u+' '+features[u]+'\n' for u in ids),encoding='utf8')
    print('Prepared exact split membership at',out)
if __name__=='__main__':main()
