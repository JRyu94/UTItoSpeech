# LaVU recognition units and evaluation conditions

Companion code and verification data for **Sentence Composition and Evaluation Conditions in Articulatory Landmark-Based Silent Speech Recognition**, by Donguk Ryu and Boreom Lee.

This release quantifies word-recognition effects of merging eight voicing pairs into a 32-unit inventory. It reports the pooled contrasts, opposing effects in the observed sentence subsets, sentence influence, LM/adaptation conditions, and the completed three-run matched filtering control. All six matched filtering intervals include zero; the release does not establish preprocessing superiority or equivalence.

## Reproduce the statistical results

Use Python 3.12. From this directory:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python -m lavu check-manifest
python -m unittest discover -s tests -v
python -m lavu reproduce
```

The final command recomputes the speaker-bootstrap analyses, word edit distances, quantitative CSVs, and the data underlying manuscript Figures 4–6. It writes `generated/verification.json`, `generated/tables/`, and `generated/figures/`. Expected statistical outputs are checked numerically rather than merely copied. The script checks 99,138 stored prediction rows and 14,508 paired-filter prediction rows (rows repeat utterance IDs across conditions). These are not counts of distinct recordings.

The input predictions and references are included as compressed JSON. Corpus recordings, DLC image models, fitted recognizers, and decoding lattices are not included. Recomputing word recognition from raw recordings requires these external inputs. Do not interpret stored-prediction verification as an independently reproduced training run.

To choose another input or output location, copy `configs/reproduce.json`, edit its paths, and run:

```bash
python -m lavu reproduce --config /path/to/reproduce.json
```

Relative config paths refer to this release root; absolute paths are also accepted. Use `--no-figures` for a statistical-only run. The scripts do not launch training or connect to an author's server.

## Features and recognition recipes

Portable trajectory processing accepts original three-header-row DLC CSVs:

```bash
python -m lavu features --tongue-csv /data/tongue.csv --lip-csv /data/lip.csv --explicit-offset 60 --output positions.npy
```

For a silent recording with no explicit offset, pass `--metadata-offset` when available. The historical nonzero fallback is retained. `--conditioning public` uses the reference public-filter semantics with the same normalizer, stream alignment, and 44-D feature order. Coordinates are conditioned before alignment/cropping, and lips precede tongue in the concatenated vector.

`recipes/prepare_data.py` maps caller-supplied Kaldi feature references onto the exact paper split IDs. `lavu lexicon` selects the highest-probability MFA pronunciation, with deterministic ties and optional 32-unit merging. `recipes/train_only_bigram.py` reproduces the training-only absolute-discount estimator. `recipes/run_hybrid.sh` and `recipes/score_kaldi.py` expose the existing Kaldi recipe and development-only word-LM-weight selection with explicit paths.

See [the recipe instructions](docs/RECIPES.md), [the table and figure map](docs/RESULT_MAP.md), and [provenance](docs/PROVENANCE.md). The original recipe records, including model choices and split text, can be materialized with:

```bash
python -m lavu records --output generated/records
```

These records describe the original runs; historical `archived-runs/` values are provenance labels, not executable local paths. Recipe components have regression tests and command-line checks. This release has not independently rerun the complete raw-data training pipeline.

## Interpretation

- Paired bootstrap: 81 speakers, 5,000 draws, seed 20261021, fixed texts and fitted models; exploratory unadjusted intervals.
- Speaker separation applies to recognizer training. Landmark extractors remain fixed, and test adaptation is offline and unsupervised.
- The corpus LM contains evaluation sentences. The training-only LM retains the lexical vocabulary, changes smoothing, and supplies different first-pass fMLLR hypotheses.
- The vocabulary comprises 9,900 lexical words. The lexicon has 9,902 entries including `!SIL` and `<UNK>`; each ARPA model has 9,902 unigrams including sentence boundaries. These counts describe different symbol sets.
- Matched filtering uses 40 phones in both arms and paired seed/job settings 1/8, 2/6, 3/6. SAT alignment succeeds for 11,616 primary and 11,615 public-filter training utterances. This is a recipe comparison including feature-specific alignment support.
- Historical auxiliary diagnostics cited in the manuscript are retained as explicitly marked frozen records in `data/historical_auxiliary.json.gz`; they are outside the corrected statistical verification. Unrelated experiments are excluded.

## Citation and release

Version: **1.0.0**, 2026-10-22. Repository: <https://github.com/JRyu94/UTItoSpeech>. Cite the exact commit or version archive used. `CITATION.cff` describes this software release. No journal acceptance status or DOI is implied.

Original release code is provided under the MIT license. External software and data retain their own terms; see `NOTICE.md`. Checkpoints and source recordings must be obtained from their respective providers.
