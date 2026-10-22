# External resources

The MIT license applies to the authors' release code. It does not relicense TaL recordings, pronunciation dictionaries, pretrained landmark extractors, Kaldi, SRILM, or the upstream reference system. Their source code, raw media, and fitted weights are not bundled here.

- TaL corpus: Ribeiro et al., SLT 2021, *TaL: a synchronised multi-speaker corpus of ultrasound tongue imaging, audio, and lip videos*.
- Pretrained extractors: Wrench and Balch-Tomes, Sensors 2022, *Beyond the Edge*. Obtain the original DeepLabCut-for-Speech-Production models from the providers.
- Reference filtering: Beeson and Richmond, Interspeech 2023; `rachelfbeeson/DeepLabCut_ASR_for_TaL`, commit `89d6b39881b4dd1d0e5024041c2f9ed8f7f5cc40`. The original `FeatureMaker` class is not redistributed. The numerical implementation in `lavu/features.py` follows the evaluated filtering procedure; its bounded equivalence checks are recorded in `data/filter_feature_validation.json`.
- Kaldi and SRILM are external dependencies of the optional recognition recipes. The corpus bigram uses SRILM Witten–Bell estimation; it is distinct from the provided training-only absolute-discount estimator.

The prediction/reference records are supplied for verification of this study. Obtain recordings and dictionary resources through their original distribution channels for new training or decoding.

`data/lexical_relations.json.gz` contains word labels and opaque equality-class IDs derived from the executed lexicons. It does not redistribute pronunciation strings. Source lexicon SHA-256 digests identify the two executed resources; the release verifies their class counts and reference-token coverage. This verification does not reconstruct the external source dictionary.
