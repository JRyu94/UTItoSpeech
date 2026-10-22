# Historical auxiliary records

`data/historical_auxiliary.json.gz` makes the historical diagnostics cited by the
manuscript available separately from the corrected primary analysis. Read it as:

```python
import gzip, json
from pathlib import Path
historical = json.loads(gzip.decompress(Path('data/historical_auxiliary.json.gz').read_bytes()))
```

The `tables` field contains modality ablations, frame discrimination, voicing
pairs, fold membership summaries, landmark-removal diagnostics, earlier test
composition, LM-weight sensitivity, Conformer-CTC configuration, runtime,
rescoring, earlier subset ablations, and the auxiliary model comparison. The
`conformer_model_report` field preserves the executed architecture/training
report, including inactive options and development checkpoint selection. Author
machine paths are replaced by provenance labels. `tal1_inventory` preserves the
two TaL1 rows formerly shown in the main inventory table.

These are frozen historical records. Their original S-number labels, section
references, and references to the earlier archive are not current manuscript
cross-references. They predate the guarded onset reevaluation and must not
replace the corrected main/supplement values. In particular, earlier TaL80 silent
WERs in this file differ from the corrected predictions. The TaL1 split shares
sessions, so it is not a cross-session evaluation. None of these records is an
independently regenerated training result of this release. The `reproduce`
command intentionally verifies the corrected paper analyses and completed
matched filtering comparison instead.
