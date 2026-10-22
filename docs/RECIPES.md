# External-data recipe components

The statistical verification requires only the supplied predictions. Feature generation and new recognition runs additionally require TaL, original DLC outputs, MFA pronunciation resources, Kaldi and (for the corpus LM) SRILM. No new training is launched by `lavu reproduce`.

## Prepare features and fixed membership

Generate 44-D features with `python -m lavu features --help`. The input CSVs must already be resampled/trimmed on the recording time base described in the manuscript. This tool starts at landmark trajectories; it does not replace video extraction or DLC inference. Explicit voiced offsets and metadata-derived silent onsets are caller inputs. The retrospective guard preserves valid explicit/nonzero historical offsets.

Convert each output matrix to a Kaldi feature archive using your Kaldi `copy-feats` workflow and collect its utterance-to-feature references in a complete `feats.scp`. Materialize a recipe's membership:

```bash
python recipes/prepare_data.py --record-recipe bfeat_group --features-scp /data/positions/feats.scp --output /work/shared32
```

`bfeat_full` gives the shared 40-phone membership. `xspk/fold0` and `xspk40/fold0` through fold8 give the disjoint conditions. The split records include reference text, utt2spk and spk2utt. Features are supplied by the caller rather than copied from an author's path.

## Lexicon and language model

The corrected dictionary selects each word's maximum-probability MFA pronunciation; exact probability ties retain the first entry. Words absent from MFA retain the caller-supplied original fallback pronunciation:

```bash
python -m lavu lexicon --dictionary /data/english_us_arpa.dict --fallback /data/original-lexicon.txt --units 32 --output /work/shared32/data/local/dict/lexicon.txt
python recipes/train_only_bigram.py --text /work/shared32/data/train/text --lexicon /work/shared32/data/local/dict/lexicon.txt --output /work/shared32/data/local/train_only.arpa
```

The training-only estimator uses absolute discount 0.75, unigram prior 1.0 and the retained lexical vocabulary. It excludes the lexicon control entries `<UNK>` and `!SIL` from its lexical vocabulary and handles sentence-boundary symbols in the ARPA model. It is not the Witten–Bell corpus LM.

The primary corpus bigram uses the corpus sentence list, including evaluation texts, and the original command:

```bash
ngram-count -order 2 -wbdiscount -text corpus.txt -lm lm.arpa
```

Use the original corpus-text construction and validated lexicon scope, rather than silently substituting only training text. The exact hashes available from the executed matched comparison are in `data/filter_feature_validation.json`. Resource acquisition and reconstruction must follow the original providers' distributions.

## Kaldi recognition

Prepare `data/local/dict` support files and `data/local/lm.arpa`. Link `steps` and `utils` from your Kaldi recipe, and provide a standard `path.sh` that exports `KALDI_ROOT` and includes Kaldi binaries. The shell runner uses an explicit recipe path:

```bash
export KALDI_ROOT=/opt/kaldi
export LAVU_PYTHON=python3
export LAVU_SCORE_SCRIPT=/path/to/UTItoSpeech/recipes/score_kaldi.py
NJ=6 ALI_NJ=16 NNET_JOBS=8 SEED=1 bash recipes/run_hybrid.sh /work/shared32 sat
NJ=6 ALI_NJ=16 NNET_JOBS=8 SEED=1 bash recipes/run_hybrid.sh /work/shared32 dnn
```

Set GMM/alignment parallelism to the configuration recorded for the specific original recipe. The primary shared DNN uses eight training jobs, and shared runs 2/3 use six with seed offsets 2000/3000. These runs change more than initialization. Do not use one job setting for every original condition and call the output the original experiment. Disjoint recipe records document their own parallelism.

The runner estimates SAT GMM transforms and reuses them for DNN decoding. Test reference transcripts are used for scoring, not transform estimation. `score_kaldi.py` selects the LM weight in 7–17 on voiced development data and retains that selection for both test modes. Changing the LM requires new first-pass hypotheses and fMLLR estimation when following the manuscript condition.

The completed matched public-filter comparison reused the existing feature-specific SAT alignments and trained three public-filter DNNs. Its protocol and actual alignment support are in `configs/filter_comparison.json`. The release includes the completed predictions; rerunning all raw-data stages has not been independently validated in this package.
