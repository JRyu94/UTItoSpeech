# From manuscript results to executable outputs

This map refers to the 2026-10-22 manuscript revision. Run from the checkout root:

```sh
python -m lavu check-manifest
python -m unittest discover -s tests -v
python -m lavu reproduce --config configs/reproduce.json
```

The command independently recomputes utterance edit distances, paired 5,000-draw
speaker-bootstrap intervals (seed 20261021), sign tests, sentence influence
checks, and three matched-filter comparisons. It compares numerical outputs
with frozen reference JSON; it does not just copy the reference tables.

| Manuscript location | Input | Output after `reproduce` | Scope |
| --- | --- | --- | --- |
| Table I: articulatory observability | Manuscript phonetic definitions and cited literature | Not regenerated | Conceptual sensor/feature mapping; the eight voicing-pair inventory mapping is executable in `lavu/lexicon.py`. |
| Table II: corpus membership | `data/recipe_records.json.gz` | `records` CLI → split text and speaker IDs | Final split counts are executable; original-corpus eligibility stages require external TaL records. |
| Table III: published-system context | Cited publications in the manuscript | Not regenerated | Literature values are not predictions from this release. |
| Table IV: LM conditions | `data/predictions.json.gz` | `generated/extended_analysis.json` → `lm_comparisons`; `generated/tables/training_only_lm.csv` | Fixed recognition models, LM-dependent first-pass adaptation; not a coverage-only control. |
| Table V: matched public-filter comparison | `data/filter_predictions.json.gz` | `generated/filter_analysis.json`; `generated/tables/matched_filter.csv` | All three seed/job pairs and both speaking modes; all six intervals include zero. |
| Table VI: pooled nine-fold results | `data/predictions.json.gz` | `results` → disjoint model/fold entries → `splits` → sum `errors` and `words` across utterance rows | Word-weighted pooled WER, not a mean of fold WERs; also retained as comparison-arm WERs in `unit_comparisons`. |
| Table VII: inventory contrasts | Same | `unit_comparisons`; `generated/tables/inventory_contrasts.csv` | All twelve pooled contrasts and sentence subsets. |
| Table VIII: subset contrast and sentence omission | Same | `unit_comparisons`, `generated/extended_analysis.json` → `sentence_robustness`; `generated/tables/sentence_influence.csv` | Exploratory fixed-text comparisons; no sentence-population bootstrap. |
| Table IX: example hypotheses | Same | Per-utterance `ref` and `hyp` fields, shared 32-unit DNN | Illustrations, not an additional test set. |
| Supplement Table X: variation across inventory training runs | Same | `unit_replicates`; `generated/tables/unit_replicates.csv`; original run summaries in input bundle | Run 1 has eight neural-network jobs; runs 2–3 have six; sample SD describes those configurations. |
| Supplement Table XI: full feature-processing factors | Same | `feature_effects`; `generated/tables/feature_ablation.csv` | Primary run, both model types and speaking modes. |
| Supplement Table XII: twelve pooled inventory contrasts | Same | `unit_comparisons`; `generated/tables/inventory_contrasts.csv` | Full sign tests and ties remain in JSON. |
| Supplement Table XIII: cache alignment audit | `data/onset_audit.json` | Audit records; `data/portable_feature_validation.json` | Frozen cache inspection, not a new processing of all media. |
| Supplement Table XIV: training-only LM comparisons across runs | `data/predictions.json.gz` | `generated/extended_analysis.json` → `trainlm_replicates` | Same vocabulary; smoothing and adaptation differ from the corpus LM. |
| Supplement Table XV: repair against smoothing | Same | `generated/extended_analysis.json` → `repair_primary`, `repair_vs_smoothing` | Primary GMM/DNN comparisons and DNN repetitions. |
| Supplement Table XVI: repeated preprocessing comparisons | Same | `factor_replicates`; `generated/tables/factor_replicates.csv` | Run-specific differences; calculate mean and sample SD over three runs. |
| Supplement Table XVII: silent sentence subsets | Same | `unit_comparisons` → each `subsets` entry | Repeated and speaker-specific texts are fixed observed subsets. |
| Supplement Table XVIII: transforms by subset | Same | `transform_effects`; `generated/tables/transforms.csv` | Transforms withheld/supplied to the same SAT-trained model. |
| Supplement Table XIX: shared-speaker inventory repetitions | Same | `unit_replicates`; `generated/tables/unit_replicates.csv` | No repeated disjoint training is inferred. |
| Supplement Tables XX–XXI: sentence and influence checks | Same | `sentence_robustness`; `generated/tables/sentence_influence.csv` | Sentence-wise counts, omissions, and speaker influence. |
| Supplement Table XXII: implementation/availability details | Recipe records and onset audits | `records` CLI; `data/onset_audit.json`, `data/onset_correction.json` | Historical and corrected timing metadata retained. |
| Figures 4–6: factors, adaptation, folds | `data/predictions.json.gz` | `generated/figures/{factors,adaptation,folds}.{png,svg,pdf}` | Reconstructed plots with the corrected predictions. |
| Figures 1–3 and 7: framework, images, example trajectories | Manuscript artwork and external TaL/DLC media | Not bundled or regenerated | No raw media or trained DLC checkpoints in this release. |

The JSON outputs are authoritative for precision; manuscript values are rounded.
The full JSON preserves additional statistics not flattened into CSV. Inspect its
keys rather than interpreting a missing CSV column as a missing analysis.

`python -m lavu records --output generated/records` expands exact split IDs,
reference texts, lexicons, LM/configuration records and speaker groupings. Raw
trajectories, pronunciation source dictionaries, Kaldi binaries, model weights,
lattices, and all prerequisite raw-data processing are not supplied. The recipes
document those dependencies and provide configurable components, rather than
claiming independently reproduced end-to-end training.
