"""Command-line verification of the submitted LaVU results."""
from pathlib import Path
import argparse,contextlib,csv,gzip,hashlib,io,json,os,runpy,sys
from .runtime import ROOT,load_json,recipe_records

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def equal(a,b,path=''):
    if path.endswith('/source_sha256'):return
    if isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for key in a:equal(a[key],b[key],path+'/'+key)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
    elif isinstance(a,(int,float)) and not isinstance(a,bool):assert abs(a-b)<1e-10,(path,a,b)
    else:assert a==b,(path,a,b)

def export_tables(out):
    sub=load_json(out/'subset_analysis.json');ext=load_json(out/'extended_analysis.json');filt=load_json(out/'filter_analysis.json')
    dest=out/'tables';dest.mkdir(exist_ok=True)
    tables={}
    tables['inventory_contrasts']=[['protocol','model','mode','subset','WER_32','WER_40','delta_pp','CI_low','CI_high']]+[[e['protocol'],e['model'],e['mode'],s,x['a_wer'],x['b_wer'],x['delta_pp'],*x['ci95']] for e in sub['unit_comparisons'] for s,x in e['subsets'].items()]
    tables['feature_ablation']=[['variant','model','mode','subset','variant_WER','baseline_WER','delta_pp','CI_low','CI_high']]+[[e['variant'],e['model'],e['mode'],s,x['a_wer'],x['b_wer'],x['delta_pp'],*x['ci95']] for e in sub['feature_effects'] for s,x in e['subsets'].items()]
    tables['transforms']=[['protocol','model','mode','subset','withheld_WER','adapted_WER','reduction_pp','CI_low','CI_high']]+[[e['protocol'],e['model'],e['mode'],s,x['a_wer'],x['b_wer'],x['delta_pp'],*x['ci95']] for e in sub['transform_effects'] for s,x in e['subsets'].items()]
    tables['matched_filter']=[['run','jobs','mode','primary_WER','public_WER','delta_pp','CI_low','CI_high']]+[[e['seed'],e['jobs'],m,x['a_wer'],x['b_wer'],x['delta_pp'],*x['ci95']] for e in filt for m in ['test_modal','test_silent'] for x in [e[m]]]
    tables['sentence_influence']=[['protocol','text','words','errors_32','errors_40','delta_pp','remaining_repeated_delta','remaining_CI_low','remaining_CI_high','subset_interaction','interaction_CI_low','interaction_CI_high']]+[[p,e['text'],e['words'],e['errors32'],e['errors40'],e['delta_pp'],e['other_five']['delta_pp'],*e['other_five']['ci95'],e['specific_minus_other_five']['delta_pp'],*e['specific_minus_other_five']['ci95']] for p,z in ext['sentence_robustness'].items() for e in z['sentences']]
    tables['training_only_lm']=[['protocol','mode','units','train_only_WER','corpus_WER','difference_pp','CI_low','CI_high']]+[[e['protocol'],e['mode'],units,x['a_wer'],x['b_wer'],x['delta_pp'],*x['ci95']] for e in ext['lm_comparisons'] for units in ['32','40'] for x in [e[units]]]
    for group in ['unit_replicates','factor_replicates']:
        tables[group]=[['run','jobs','variant','mode','subset','A_WER','B_WER','delta_pp','CI_low','CI_high']]+[[e['seed'],e['jobs'],e.get('variant','inventory'),e['mode'],s,x['a_wer'],x['b_wer'],x['delta_pp'],*x['ci95']] for e in sub[group] for s,x in e['subsets'].items()]
    diag=load_json(out/'inventory_diagnostics.json')
    tables['lm_subset_inventory']=[['protocol','run','jobs','mode','lm','subset','errors32','errors40','words','WER32','WER40','delta_pp','CI_low','CI_high']]+[[e['protocol'],e['run'],e['jobs'],e['mode'],e['lm'],s,x['errors32'],x['errors40'],x['words'],x['wer32'],x['wer40'],x['delta_pp'],*x['ci95']] for e in diag['conditions'] for s,x in e['subsets'].items()]
    tables['subset_interactions']=[['protocol','run','jobs','mode','lm','specific_minus_repeated','CI_low','CI_high']]+[[e['protocol'],e['run'],e['jobs'],e['mode'],e['lm'],e['interaction']['delta_pp'],*e['interaction']['ci95']] for e in diag['conditions']]
    tables['lm_condition_interaction_changes']=[['protocol','run','jobs','mode','corpus_minus_training_only_interaction','CI_low','CI_high']]+[[e['protocol'],e['run'],e['jobs'],e['mode'],e['delta_pp'],*e['ci95']] for e in diag['lm_interaction_changes']]
    tables['lexical_ambiguity']=[['subset','newly_ambiguous_reference_tokens','reference_tokens','coverage_percent','utterances','distinct_texts']]+[[s,*[diag['lexical'][s][k] for k in ['newly_ambiguous_reference_tokens','reference_tokens','coverage_percent','utterances','distinct_texts']]] for s in ['all','repeated','specific']]
    tables['error_decomposition']=[['protocol','subset','units','words','substitutions','deletions','insertions','new_homophone_substitutions','errors']]+[[e['protocol'],e['subset'],u,e['words'],*[e[u][k] for k in ['substitutions','deletions','insertions','new_homophone_substitutions','errors']]] for e in diag['word_error_diagnostics'] for u in ['40','32']]
    tables['paired_examples']=[['utterance','subset','reference','hyp40','hyp32','errors40','errors32','delta_errors']]+[[e[k] for k in ['utt','subset','reference','hyp40','hyp32','errors40','errors32','delta_errors']] for e in diag['examples']]
    for name,rows in tables.items():
        with (dest/(name+'.csv')).open('w',newline='',encoding='utf8') as f:csv.writer(f).writerows(rows)
    # The full machine-readable analyses also retain sign tests, subset interactions,
    # run summaries, and shared/disjoint contrasts; see docs/RESULT_MAP.md.
    dump(out/'table_exports.json',{name:len(rows)-1 for name,rows in tables.items()})

def reproduce(config,no_figures=False):
    config=Path(config).resolve();cfg=load_json(config)
    if (cfg.get('seed'),cfg.get('speaker_draws'))!=(20261021,5000):
        raise ValueError('This release verifies the frozen analysis: seed 20261021 and 5000 draws')
    # Config paths are relative to the release root, independent of the caller's CWD.
    out=Path(cfg['output']);out=out if out.is_absolute() else ROOT/out
    bundle=Path(cfg['predictions']);bundle=bundle if bundle.is_absolute() else ROOT/bundle
    filters=Path(cfg['filter_predictions']);filters=filters if filters.is_absolute() else ROOT/filters
    out.mkdir(parents=True,exist_ok=True);os.environ['LAVU_OUTPUT']=str(out.resolve());os.environ['LAVU_PREDICTIONS']=str(bundle.resolve())
    for module in ['lavu.analysis.analyse_subsets','lavu.analysis.extended_analysis']:
        log=io.StringIO()
        with contextlib.redirect_stdout(log):runpy.run_module(module,run_name='__main__')
        (out/(module.rsplit('.',1)[-1]+'.log')).write_text(log.getvalue(),encoding='utf8')
    from .filter_analysis import analyse
    contrasts,checked=analyse(filters);dump(out/'filter_analysis.json',contrasts)
    from .inventory_diagnostics import analyse as diagnose
    relations=Path(cfg.get('lexical_relations','data/lexical_relations.json.gz'))
    relations=relations if relations.is_absolute() else ROOT/relations
    dump(out/'inventory_diagnostics.json',diagnose(bundle,relations))
    expected=Path(cfg['expected']);expected=expected if expected.is_absolute() else ROOT/expected
    for name in ['subset_analysis.json','extended_analysis.json','filter_analysis.json','inventory_diagnostics.json']:equal(load_json(expected/name),load_json(out/name),name)
    export_tables(out)
    if not no_figures:runpy.run_module('lavu.analysis.make_figures',run_name='__main__')
    count=load_json(out/'subset_analysis.json')['validation']['independently_recomputed_utterance_errors']
    report={'status':'passed','stored_prediction_row_checks':count,'filter_prediction_row_checks':checked,'word_alignment_checks':load_json(out/'inventory_diagnostics.json')['validation']['word_alignments_checked'],'numeric_outputs_match_reference':True,'figures_regenerated':not no_figures,'numpy_version':__import__('numpy').__version__}
    dump(out/'verification.json',report);print(json.dumps(report,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('reproduce');p.add_argument('--config',default=str(ROOT/'configs/reproduce.json'));p.add_argument('--no-figures',action='store_true')
    p=sub.add_parser('records');p.add_argument('--output',required=True)
    p=sub.add_parser('features');p.add_argument('--tongue-csv',required=True);p.add_argument('--lip-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--conditioning',choices=['primary','public'],default='primary');p.add_argument('--explicit-offset',type=int);p.add_argument('--metadata-offset',type=float)
    p=sub.add_parser('lexicon');p.add_argument('--dictionary',required=True);p.add_argument('--fallback',required=True);p.add_argument('--output',required=True);p.add_argument('--units',type=int,choices=[32,40],required=True)
    p=sub.add_parser('check-manifest')
    args=parser.parse_args()
    if args.command=='reproduce':reproduce(args.config,args.no_figures)
    elif args.command=='records':
        root=Path(args.output).resolve();records=recipe_records()
        for name,content in records['files'].items():
            p=(root/name).resolve();assert p.is_relative_to(root)
            p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content,encoding='utf8')
        print('Exported',len(records['files']),'record files to',root)
    elif args.command=='features':
        import numpy as np
        from .features import read_dlc_csv,make_positions,TONGUE,LIP
        tx,tc=read_dlc_csv(args.tongue_csv,TONGUE);lx,lc=read_dlc_csv(args.lip_csv,LIP)
        x,meta=make_positions(tx,tc,lx,lc,args.explicit_offset,args.metadata_offset,args.conditioning)
        np.save(args.output,x);dump(Path(args.output).with_suffix('.json'),meta)
    elif args.command=='lexicon':
        from .lexicon import corrected_lexicon
        corrected_lexicon(args.dictionary,args.fallback,args.output,args.units)
    elif args.command=='check-manifest':
        manifest=load_json(ROOT/'SHA256SUMS.json')
        for name,digest in manifest.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        print('Verified',len(manifest),'file hashes')

if __name__=='__main__':main()
