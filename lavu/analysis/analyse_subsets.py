"""Reaggregate stored hypotheses; never invokes a model or decoder."""
from pathlib import Path
from collections import Counter, defaultdict
import json
import hashlib
import math
import numpy as np

from lavu.runtime import context, load_json
HERE, BUNDLE = context()
data = load_json(BUNDLE)
results = data['results']
reference = results['shared_32_DNN+fMLLR']['splits']['test_modal']['rows']
texts = [text for text, count in Counter(r['ref'] for r in reference).most_common(6)]
assert all(sum(r['ref'] == text for r in reference) >= 80 for text in texts)
shared = set(texts)
speakers = sorted({r['utt'].split('_', 1)[0] for r in reference})
assert len(speakers) == 81
ref_by_id = {r['utt']: (r['ref'], r['words']) for r in reference}
generator = np.random.default_rng(20261021)
draws = generator.integers(0, 81, size=(5000, 81))

def aggregate(keys, mode, subset):
    rows = []
    for key in keys:
        rows.extend(results[key]['splits'][mode]['rows'])
    assert len(rows) == 1209
    assert len({r['utt'] for r in rows}) == 1209
    # Modal and silent IDs differ; content and word totals must agree within every contrast.
    counts = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        selected = (row['ref'] in shared)
        if subset != 'all' and selected != (subset == 'repeated'):
            continue
        assert row['words'] == len(row['ref'].split())
        counts[row['utt'].split('_', 1)[0]][0] += row['errors']
        counts[row['utt'].split('_', 1)[0]][1] += row['words']
        counts[row['utt'].split('_', 1)[0]][2] += 1
    a = np.array([counts[s] for s in speakers], dtype=np.float64)
    assert (a[:, 1] > 0).all()
    return a

def wer(a):
    return 100 * a[:, 0].sum() / a[:, 1].sum()

def difference(a, b):
    assert np.array_equal(a[:, 1:], b[:, 1:])
    sampled_a, sampled_b = a[draws].sum(1), b[draws].sum(1)
    boots = 100 * (sampled_a[:, 0] / sampled_a[:, 1] - sampled_b[:, 0] / sampled_b[:, 1])
    signs = a[:, 0] - b[:, 0]
    negative, positive = int((signs < 0).sum()), int((signs > 0).sum())
    n = negative + positive
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(negative, positive) + 1)) / 2**n) if n else 1.0
    return dict(a_wer=wer(a), b_wer=wer(b), delta_pp=wer(a)-wer(b),
                ci95=np.quantile(boots, [0.025, 0.975]).tolist(),
                negative_speakers=negative, positive_speakers=positive,
                ties=81-negative-positive, sign_p=p, words=int(a[:, 1].sum()),
                utterances=int(a[:, 2].sum())), boots

def keys(protocol, units, model):
    if protocol == 'shared':
        return [f'shared_{units}_{model}']
    return [f'disjoint_{units}_{model}_fold{i}' for i in range(9)]

out = {'method': {'source_sha256': hashlib.sha256(BUNDLE.read_bytes()).hexdigest(),
                 'bootstrap': 'paired resampling of 81 speakers; fixed sentence subsets and fitted models',
                 'resamples': 5000, 'seed': 20261021,
                 'multiplicity': 'unadjusted exploratory confidence intervals',
                 'repeated_texts': texts,
                 'scope': 'existing hypotheses only; no retraining, decoding or test-based selection'},
       'baseline': {}, 'unit_comparisons': [], 'transform_effects': [], 'feature_effects': [],
       'unit_replicates': [], 'factor_replicates': []}
for mode in ['test_modal', 'test_silent']:
    out['baseline'][mode] = {}
    for subset in ['all', 'repeated', 'specific']:
        a = aggregate(keys('shared', 32, 'DNN+fMLLR'), mode, subset)
        out['baseline'][mode][subset] = dict(wer=wer(a), words=int(a[:, 1].sum()), utterances=int(a[:, 2].sum()))

for protocol in ['shared', 'disjoint']:
    models = ['GMM+fMLLR', 'DNN+fMLLR'] if protocol == 'shared' else ['GMM_transform_withheld', 'GMM+fMLLR', 'DNN_transform_withheld', 'DNN+fMLLR']
    for model in models:
        for mode in ['test_modal', 'test_silent']:
            entry = dict(protocol=protocol, model=model, mode=mode, subsets={})
            samples = {}
            for subset in ['all', 'repeated', 'specific']:
                entry['subsets'][subset], samples[subset] = difference(
                    aggregate(keys(protocol, 32, model), mode, subset),
                    aggregate(keys(protocol, 40, model), mode, subset))
            dd = samples['specific'] - samples['repeated']
            entry['specific_minus_repeated'] = dict(delta_pp=entry['subsets']['specific']['delta_pp']-entry['subsets']['repeated']['delta_pp'],
                                                     ci95=np.quantile(dd, [0.025, 0.975]).tolist())
            out['unit_comparisons'].append(entry)

for protocol in ['shared', 'disjoint']:
    models = ['DNN'] if protocol == 'shared' else ['GMM', 'DNN']
    for model in models:
        for mode in ['test_modal', 'test_silent']:
            entry = dict(protocol=protocol, model=model, mode=mode, subsets={})
            for subset in ['all', 'repeated', 'specific']:
                entry['subsets'][subset], _ = difference(
                    aggregate(keys(protocol, 32, model+'_transform_withheld'), mode, subset),
                    aggregate(keys(protocol, 32, model+'+fMLLR'), mode, subset))
            out['transform_effects'].append(entry)

for variant in ['f_smooth_5tap_g','f_clean_ours_g','f_norm_mad_g','f_nosync_g','ours_full_g']:
    for model in ['GMM+fMLLR', 'DNN+fMLLR']:
        for mode in ['test_modal', 'test_silent']:
            entry = dict(variant=variant, model=model, mode=mode, subsets={})
            for subset in ['all', 'repeated', 'specific']:
                entry['subsets'][subset], _ = difference(
                    aggregate([f'factor_{variant}_{model}_s1'], mode, subset),
                    aggregate(keys('shared', 32, model), mode, subset))
            out['feature_effects'].append(entry)

for seed in [1, 2, 3]:
    for mode in ['test_modal', 'test_silent']:
        entry = dict(seed=seed, jobs=8 if seed==1 else 6, mode=mode, subsets={})
        for subset in ['all', 'repeated', 'specific']:
            k32 = f'shared_32_DNN_s{seed}' if seed != 1 else 'shared_32_DNN+fMLLR'
            k40 = f'shared_40_DNN_s{seed}' if seed != 1 else 'shared_40_DNN+fMLLR'
            entry['subsets'][subset], _ = difference(aggregate([k32], mode, subset), aggregate([k40], mode, subset))
        out['unit_replicates'].append(entry)
    for variant in ['f_smooth_5tap_g','f_clean_ours_g','f_norm_mad_g']:
        for mode in ['test_modal', 'test_silent']:
            entry = dict(seed=seed, jobs=8 if seed==1 else 6, variant=variant, mode=mode, subsets={})
            for subset in ['all', 'repeated', 'specific']:
                base = f'shared_32_DNN_s{seed}' if seed!=1 else 'shared_32_DNN+fMLLR'
                entry['subsets'][subset], _ = difference(aggregate([f'factor_{variant}_DNN+fMLLR_s{seed}'], mode, subset), aggregate([base], mode, subset))
            out['factor_replicates'].append(entry)

def edits(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x!=y)))
        previous = current
    return previous[-1]

checked = 0
canonical = {}
for condition in results.values():
    for mode, split in condition['splits'].items():
        expected_summary = next(x for x in condition['summary']['test'] if x['split']==mode)
        assert sum(x['words'] for x in split['rows']) == expected_summary['words']
        assert sum(x['errors'] for x in split['rows']) == expected_summary['errors']
        for row in split['rows']:
            key = (mode, row['utt'])
            value = (row['ref'], row['words'])
            assert canonical.setdefault(key, value) == value
            assert edits(row['ref'].split(), row['hyp'].split()) == row['errors']
            checked += 1
out['validation'] = {'independently_recomputed_utterance_errors': checked,
                     'reference_consistency': 'passed', 'stored_aggregate_reproduction': 'passed'}

(HERE/'subset_analysis.json').write_text(json.dumps(out, indent=2), encoding='utf8')
print('BASELINE', json.dumps(out['baseline']))
for group in ['unit_comparisons','transform_effects','feature_effects']:
    print(group)
    for e in out[group]:
        print(e.get('protocol',e.get('variant')),e['model'],e['mode'],
              [(k, round(v['delta_pp'],2), [round(x,2) for x in v['ci95']]) for k,v in e['subsets'].items() if k!='all'])
print('UNIT_REPLICATES', [(e['seed'], e['mode'], {k:round(v['delta_pp'],2) for k,v in e['subsets'].items()}) for e in out['unit_replicates']])
print('VALIDATION', out['validation'])
