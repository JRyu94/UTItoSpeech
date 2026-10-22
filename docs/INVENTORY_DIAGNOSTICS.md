# Inventory diagnostics and definitions

Run `python -m lavu reproduce`. No training or decoding is launched. All analyses reuse the corrected predictions and the same fixed test membership.

- Inputs: `data/predictions.json.gz` and `data/lexical_relations.json.gz`.
- Implementation: `lavu/inventory_diagnostics.py`.
- Expected output: `expected/inventory_diagnostics.json`.
- Generated output: `generated/inventory_diagnostics.json` and the six CSVs named in `generated/table_exports.json`.
- Inference: 81 paired speaker clusters, 5,000 draws, seed 20261021; models, texts and selected decoders are fixed. Intervals are exploratory, unadjusted, and do not estimate training uncertainty.

## Lexical audit

Each of 9,902 entries has an opaque 40-phone class and a 32-unit class. Every 40-phone class must be contained in a 32-unit class. A homophone class has more than one entry. The executed lexicons contain 209 and 460 such classes. Of the entries, 569 are unique at 40 phones and become homophonous at 32 units. Another 90 already homophonous entries acquire additional competitors, making 659 entries with expanded competitor sets. These are different estimands.

The 569-entry set covers 1,752 of 13,614 reference tokens (12.9%). Coverage is 886/7,828 (11.3%) on repeated sentences and 866/5,786 (15.0%) on specific sentences. Entries include `!SIL` and `<UNK>`; there are 9,900 lexical words. Raw pronunciations are not bundled. The source digests in the relations file record which executed lexicons supplied the equivalence classes.

## Error decomposition and examples

Optimal word-edit traceback prioritizes an exact match, substitution, deletion, then insertion. Counts of substitutions, deletions and insertions sum to each stored error total. A newly homophonous substitution has unequal words in the same 32-unit class but different 40-phone classes. It can occur under either recognizer; its name describes a lexical relationship, not a mechanism. Optimal alignments need not be unique, so these categories are descriptive and depend on the stated tie policy.

In the primary disjoint corpus silent condition, repeated errors change by +62 substitutions, +62 deletions and −1 insertion. Newly homophonous substitutions change from 1 to 0. Specific errors change by −73 substitutions, −76 deletions and +10 insertions; newly homophonous substitutions change from 9 to 7. This decomposition does not support attributing the repeated increase to these lexical collisions. It also does not establish a different causal mechanism.

The two displayed output examples are selected post hoc by explicit deterministic rules, encoded in the module: the largest repeated-sentence increase with an added newly homophonous substitution, if available (otherwise the largest increase), and the largest specific-sentence decrease among references of at most 12 words. Utterance ID breaks ties. They illustrate output differences and are not representative samples or separate tests.

## LM conditions and repetitions

`lm_subset_inventory.csv` reports all 16 protocol/run/mode/LM conditions and all three subsets. `subset_interactions.csv` gives specific-minus-repeated effects. `lm_condition_interaction_changes.csv` estimates the direct corpus-minus-training-only change in that interaction using the same paired draws; comparing whether separate intervals include zero would not test this difference. LM conditions differ in smoothing and first-pass hypotheses used to reestimate fMLLR, so the contrast does not isolate evaluation-text coverage.

Shared runs have matched inventory job counts within each run (8, 6, 6 across runs). The corpus silent interaction point estimates are −2.82, −1.18, −1.32 points, but the last two intervals include zero. The individual repeated effects are +0.52, +0.13, −0.89 points. Thus opposite subset signs are not replicated in all runs. There is one nine-fold disjoint model grid; speaker-bootstrap intervals and shared repetitions do not constitute disjoint training replication.
