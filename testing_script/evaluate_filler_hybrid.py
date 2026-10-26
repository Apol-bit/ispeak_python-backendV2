"""Compare pinned detectors on identical constructed labels; never load Whisper.

Run from the Python repo with its existing venv. Requires local Git history and
the existing local filler classifier. No network or model download is used.
"""
import argparse
import ast
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audio_analysis_processing_files import filler_words as hybrid
from audio_analysis_processing_files.word_alignment import WORD
from audio_analysis_processing_files.filler_vocabulary import ENGLISH, FILIPINO, ADDITIONAL_REVIEW_PHRASES

BASELINE = 'b210402897e08fb1f6d60dbc24134eca7a3b8aa0'
LEGACY_PATH = 'audio_analysis_processing_files/clarity_analysis_module/voice_fillerwords_detection.py'


def git_source(ref, path):
    return subprocess.check_output(['git', '-c', 'safe.directory=' + ROOT.as_posix(),
        '-C', str(ROOT), 'show', ref + ':' + path], text=True, encoding='utf-8')


def module(source, name, file):
    result = types.ModuleType(name); result.__file__ = str(file)
    sys.modules[name] = result
    exec(compile(source, str(file), 'exec'), result.__dict__)
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def add_expression_metrics(evidence, fixtures):
    """Exact one-to-one expression matches, alongside the token-occurrence metric."""
    by_id = {item['id']: item for item in fixtures['cases']}
    aggregate = defaultdict(lambda: dict(expression_tp=0, expression_fp=0,
        expression_fn=0, ambiguous_confirmed_spans=0))
    for item in evidence['cases']:
        fixture = by_id[item['id']]
        text, aligned = hybrid._build_transcript(fixture['word_segments'], fixture['transcript'])
        tokens = list(WORD.finditer(text))
        def token_mask(start, end):
            return {i for i,t in enumerate(tokens) if t.start()<end and t.end()>start}
        gold = [token_mask(a['start'],a['end']) for a in fixture['annotations'] if a['label']=='filler']
        ambiguous = set().union(*(token_mask(a['start'],a['end']) for a in fixture['annotations']
                                  if a['label']=='ambiguous'))
        for name, result in item['results'].items():
            matched, fp, ambiguous_spans = set(), 0, 0
            for span in result['confirmed']:
                words = [w for w in aligned if w['start'] < span['end'] and w['end'] > span['start']]
                predicted = set().union(*(token_mask(w['char_start'],w['char_end']) for w in words))
                known = predicted - ambiguous
                if predicted and not known:
                    ambiguous_spans += 1
                    continue
                candidates = [i for i,g in enumerate(gold) if i not in matched and g==known]
                if candidates:
                    matched.add(candidates[0])
                else:
                    fp += 1
            stats = dict(expression_tp=len(matched), expression_fp=fp,
                         expression_fn=len(gold)-len(matched), ambiguous_confirmed_spans=ambiguous_spans)
            result.update(stats)
            for group in ['all','language:'+item['language'],'partition:'+item['split'],
                          'language-partition:'+item['language']+':'+item['split']]:
                for key,value in stats.items(): aggregate[(name,group)][key] += value
    for row in evidence['summaries']:
        stats = aggregate[(row['detector'],row['group'])]
        row.update(stats)
        tp,fp,fn = stats['expression_tp'],stats['expression_fp'],stats['expression_fn']
        row['expression_precision'] = round(tp/(tp+fp),4) if tp+fp else None
        row['expression_recall'] = round(tp/(tp+fn),4) if tp+fn else None
        assert tp+fp+stats['ambiguous_confirmed_spans'] == row['confirmed_spans']
    evidence['expression_metric_unit'] = 'exact one-to-one lexical-token-set match to a labeled expression; entirely ambiguous spans excluded and reported separately; duplicate predictions are false positives'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--summarize-existing', type=Path,
                        help='Add expression metrics to saved real-model results without rerunning inference')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = ROOT / 'testing_script/fixtures/filler_hybrid_labeled.json'
    fixtures = json.loads(fixture_path.read_text(encoding='utf-8'))
    if args.summarize_existing:
        evidence = json.loads(args.summarize_existing.read_text(encoding='utf-8'))
        assert evidence['fixture_sha256'] == sha256(fixture_path)
        assert all(sha256(ROOT/'audio_analysis_processing_files'/name)==digest
                   for name,digest in evidence['hybrid_source_sha256'].items())
        add_expression_metrics(evidence, fixtures)
        (args.output_dir/'comparison.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps([r for r in evidence['summaries'] if r['group']=='all']))
        return
    old_vocab = module(git_source(BASELINE, 'audio_analysis_processing_files/filler_vocabulary.py'),
                       'baseline_vocab', ROOT / 'audio_analysis_processing_files/filler_vocabulary.py')
    baseline = module(git_source(BASELINE, 'audio_analysis_processing_files/filler_words.py'),
                      'baseline_fillers', ROOT / 'audio_analysis_processing_files/filler_words.py')
    baseline.VOCABULARY = old_vocab.VOCABULARY  # Preserve baseline coverage, not the expanded live module.
    old_source = git_source('e80f08a^', LEGACY_PATH)
    legacy = module(old_source, 'legacy_fillers', ROOT / LEGACY_PATH)
    (args.output_dir / 'legacy_detector.py').write_text(old_source, encoding='utf-8')
    classifier = hybrid._default_classifier()
    # Reuse the ACTUAL existing API dictionary expression, not a parallel mapper.
    tree = ast.parse((ROOT / 'whisper_service.py').read_text(encoding='utf-8'))
    expressions = [value for node in ast.walk(tree) if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Dict)
        for key, value in zip(node.value.keys, node.value.values)
        if isinstance(key, ast.Constant) and key.value == 'fillers']
    assert len(expressions) == 1
    serializer = compile(ast.Expression(expressions[0]), 'whisper_service.py:fillers', 'eval')
    totals = defaultdict(lambda: dict(tp=0, fp=0, fn=0, confirmed_spans=0, review_spans=0,
                                     ambiguous_confirmed_tokens=0, cases=0, gold_expressions=0))
    records = []
    for index, item in enumerate(fixtures['cases']):
        text, aligned = hybrid._build_transcript(item['word_segments'], item['transcript'])
        assert text == item['transcript'], 'All detectors must receive identical text'
        tokens = list(WORD.finditer(text))
        labels = {label: {i for i,t in enumerate(tokens) if any(
            a['label'] == label and t.start() < a['end'] and t.end() > a['start']
            for a in item['annotations'])} for label in ['filler', 'ambiguous']}
        predictions = classifier(text)
        current = baseline.analyze_fillers(item['word_segments'], classifier=lambda _: predictions, full_transcript=text)
        updated = hybrid.analyze_fillers(item['word_segments'], classifier=lambda _: predictions, full_transcript=text)
        assert current['analysis_available'] and updated['analysis_available']
        assert current['coverage']['status'] == updated['coverage']['status'] == 'matched'
        detectors = {
            'legacy': dict(filler_words=legacy._detect_fillers_from_text(item['word_segments']), filler_candidates=[]),
            'current': current, 'hybrid': updated,
        }
        case_results = {}
        for name, result in detectors.items():
            predicted = set()
            for span in result['filler_words']:
                covered = [w for w in aligned if w['start'] < span['end'] and w['end'] > span['start']]
                for i,token in enumerate(tokens):
                    if any(token.start() < w['char_end'] and token.end() > w['char_start'] for w in covered):
                        predicted.add(i)
            measured = predicted - labels['ambiguous']
            stats = dict(tp=len(measured & labels['filler']), fp=len(measured - labels['filler']),
                         fn=len(labels['filler'] - measured), confirmed_spans=len(result['filler_words']),
                         review_spans=len(result['filler_candidates']),
                         ambiguous_confirmed_tokens=len(predicted & labels['ambiguous']), cases=1,
                         gold_expressions=sum(a['label'] == 'filler' for a in item['annotations']))
            for group in ['all', 'language:' + item['language'], 'partition:' + item['split'],
                          'language-partition:' + item['language'] + ':' + item['split']]:
                aggregate = totals[(name, group)]
                for key,value in stats.items(): aggregate[key] += value
            case_results[name] = dict(**stats, confirmed=result['filler_words'], review=result['filler_candidates'])
        api = eval(serializer, {'round': round, 'filler_stats': updated, 'filler_score': updated['filler_score']})
        assert api['count'] == len(api['words']) and api['candidate_count'] == len(api['candidates'])
        records.append(dict(id=item['id'], language=item['language'], split=item['split'],
            transcript=text, annotations=item['annotations'], results=case_results, api=api,
            model_predictions=[dict(label=p['entity_group'], start=int(p['start']), end=int(p['end']),
                                    score=float(p['score'])) for p in predictions]))
        if (index + 1) % 13 == 0:
            print(json.dumps(dict(progress=index+1, total=len(fixtures['cases']))), flush=True)
    summaries = []
    for (name, group), stats in sorted(totals.items()):
        tp,fp,fn = stats['tp'], stats['fp'], stats['fn']
        precision = tp / (tp+fp) if tp+fp else None
        recall = tp / (tp+fn) if tp+fn else None
        summaries.append(dict(detector=name, group=group, **stats,
            precision=round(precision,4) if precision is not None else None,
            recall=round(recall,4) if recall is not None else None,
            f1=round(2*tp/(2*tp+fp+fn),4) if 2*tp+fp+fn else None))
    weights = Path(classifier.model_path) / 'model.safetensors'
    missing = sorted({term.strip().casefold() for name in [
        '_ENGLISH_ALWAYS_FILLER','_ENGLISH_MULTI_WORD_FILLERS','_ENGLISH_REPEAT_ONLY_FILLERS',
        '_FILIPINO_ALWAYS_FILLER','_FILIPINO_MULTI_WORD_FILLERS','_FILIPINO_REPEAT_ONLY_FILLERS']
        for term in getattr(legacy,name)} - (old_vocab.ENGLISH | old_vocab.FILIPINO))
    recovered = {term.strip().casefold() for name in [
        '_ENGLISH_ALWAYS_FILLER','_ENGLISH_MULTI_WORD_FILLERS','_ENGLISH_REPEAT_ONLY_FILLERS',
        '_FILIPINO_ALWAYS_FILLER','_FILIPINO_MULTI_WORD_FILLERS','_FILIPINO_REPEAT_ONLY_FILLERS']
        for term in getattr(legacy,name)}
    assert recovered.issubset(ENGLISH | FILIPINO), 'Incomplete historical vocabulary recovery'
    evidence = dict(fixture_sha256=sha256(fixture_path), fixture_provenance=fixtures['provenance'],
        partition_note='Development/check partitions were frozen before implementation but author-visible; the heldout name is NOT a blinded independent holdout',
        metric_unit='unique lexical-token occurrences inside confirmed spans; ambiguous labels excluded from precision/recall and reported separately; grouped span counts also reported',
        model_weights_sha256=sha256(weights), model_config_sha256=sha256(Path(classifier.model_path)/'config.json'),
        hybrid_source_sha256={name: sha256(ROOT / 'audio_analysis_processing_files' / name)
                              for name in ['filler_words.py','filler_vocabulary.py','filler_context.py']},
        package_versions={name: importlib.metadata.version(name) for name in ['torch','transformers','tokenizers']},
        baseline_source=BASELINE, legacy_source='e80f08a^', missing_from_previous_review_vocabulary=missing,
        recovered_unique_terms=len(ENGLISH | FILIPINO), additional_review_phrases=sorted(ADDITIONAL_REVIEW_PHRASES),
        whisper_changed=False, scoring_formula_changed=False, confidence_threshold=.60,
        summaries=summaries, cases=records)
    add_expression_metrics(evidence, fixtures)
    (args.output_dir / 'comparison.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([s for s in summaries if s['group'] in ['all','language:English','language:Filipino','language:Taglish']]))


if __name__ == '__main__':
    main()
