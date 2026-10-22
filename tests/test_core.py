import tempfile,unittest
from pathlib import Path
import numpy as np
from lavu.alignment import resolve_lip_offset
from lavu.lexicon import mfa_best,MERGE
from lavu.runtime import edit_distance
from lavu.features import make_positions,public_condition
from lavu.language_model import train

class ScientificRegressionTests(unittest.TestCase):
    def test_collapsed_fallback_restored(self):self.assertEqual(resolve_lip_offset(None,59,506,569),59)
    def test_explicit_and_nonzero_offsets_preserved(self):
        self.assertEqual(resolve_lip_offset(57,59,600,540),57)
        self.assertEqual(resolve_lip_offset(None,59,600,540),60)
    def test_invalid_metadata_rejected(self):
        for value in [-1,float('nan'),600]:
            with self.assertRaises(ValueError):resolve_lip_offset(None,value,506,569)
    def test_high_probability_pronunciation_and_tie(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'mfa.dict';p.write_text('about\t0.01\tAH0 B AW1\nabout\t0.99\tAH0 B AW1 T\ntie\t0.5\tT AY1\ntie\t0.5\tT IY1\n')
            lex=mfa_best(p);self.assertEqual(lex['about'],['AH','B','AW','T']);self.assertEqual(lex['tie'],['T','AY'])
    def test_merge_creates_intended_lexical_collision(self):
        pig=['P','IH','G'];big=['B','IH','G']
        self.assertEqual([MERGE.get(p,p) for p in pig],[MERGE.get(p,p) for p in big]);self.assertEqual(len(set(MERGE.values())),8)
    def test_edit_distance_handles_insertions_and_empty_hypothesis(self):
        self.assertEqual(edit_distance('a b c','a x c d'),2);self.assertEqual(edit_distance('a b',''),2)
    def test_crop_order_and_shape(self):
        rng=np.random.default_rng(7);t=rng.normal(size=(90,14,2)).astype(np.float32);l=rng.normal(size=(100,8,2)).astype(np.float32)
        x,meta=make_positions(t,np.ones((90,14)),l,np.ones((100,8)),explicit_offset=15)
        self.assertEqual(x.shape,(85,44));self.assertEqual(meta['offset'],15);self.assertTrue(np.isfinite(x).all());np.testing.assert_allclose(x.mean(0),0,atol=1e-5)
    def test_reference_filter_rejects_constant_trajectory(self):
        with self.assertRaises(ValueError):public_condition(np.ones((30,8,2)),np.ones((30,8)))
    def test_train_only_lm_normalization_and_retained_unseen_word(self):
        pu,pb,bo,error=train([['a','b'],['a','a']],{'a','b','unseen'})
        self.assertGreater(pu['unseen'],0);self.assertLess(error,1e-9)
        self.assertAlmostEqual(sum(pb['a'].values())+bo['a']*(1-sum(pu[w] for w in pb['a'])),1.)

if __name__=='__main__':unittest.main()
