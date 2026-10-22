# Executed results and exported code

Primary predictions are the corrected-lexicon, guarded-onset results used by the manuscript. The underlying audit independently verified stored edit counts before this release. `expected/` contains the corresponding reference analyses. `lavu reproduce` reruns the computations and compares outputs numerically.

The data export preserves utterance IDs, reference/hypothesis words and error/reference counts. Original machine-specific root strings in provenance metadata are converted to relative provenance labels. No active command relies on these labels. Original internal experiment records are preserved by the authors and are not overwritten by this export.

The filtering export contains only the three completed 40-phone matched runs and their primary comparators. It preserves per-run dev selection, model hashes, both test modes, the fixed filter protocol and the alignment-support audit. Unrelated experiments and projects are excluded.

`data/onset_audit.json` describes the 1,209 silent trajectories. `data/onset_correction.json` records the affected speaker's fixed-model reevaluation. The real historical fallback for `67fe_031_xsil` was zero, metadata onset was 59, and corrected feature length was 447. The original source-length calculation of 443 was not the offset used by the recognizer cache.

The checked statistical scripts retain the original estimands: fixed texts/models, paired resampling of 81 speakers, 5,000 draws, seed 20261021. Intervals are exploratory and unadjusted. No favorable run or test-selected LM weight is substituted.

The portable recipe components expose feature conditioning, offset handling, recognition-unit mapping, pronunciation correction, exact split membership, the training-only bigram and Kaldi command-line entry points. Statistical reproduction and small numerical regressions are tested in this release. Complete feature-extractor training, raw-video conversion and an independent end-to-end recognizer retraining are outside the verified execution scope.

The historical auxiliary record is separate from corrected predictions. Its old values and legacy table labels remain identifiable; see `docs/HISTORICAL_AUXILIARY.md`. The frozen Conformer report and TaL1 rows support the manuscript's historical context, not a controlled architecture or cross-session superiority claim.
