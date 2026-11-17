"""Verify saved requests against baseline and produce a conservative comparison.

No inference. Truth labels are explicit, scoped review records, not model judging.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'onair_diag_20261116'
sys.path.insert(0, str(ROOT / 'code'))
import run_diagnostic as base
import run_separated_followup as separated


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    cfg = read(ROOT / 'config.json')
    scenes = read(ROOT / 'scene_inputs.json')
    baseline = {r['scene']: r for r in jsonl(BASE / 'raw_responses.jsonl') if r['condition'] == 'F'}
    current = jsonl(ROOT / 'raw_responses.jsonl')
    ledger = jsonl(ROOT / 'request_ledger.jsonl')
    checks = []

    def check(name, passed):
        checks.append({'check': name, 'passed': bool(passed)})
        if not passed:
            raise AssertionError(name)

    check('exactly four new F calls', len(current) == 4 and all(r['condition'] == 'F' for r in current))
    check('one call per scene', {r['scene'] for r in current} == {s['scene'] for s in scenes})
    check('ledger matches saved requests', [r['request_sha256'] for r in current] == [r['request_sha256'] for r in ledger])
    check('original plus followup equals twenty', len(jsonl(BASE / 'raw_responses.jsonl')) + len(current) == 20)
    check('configuration unchanged', cfg == read(BASE / 'config.json'))
    check('scene specifications unchanged', scenes == read(BASE / 'scene_inputs.json'))
    for sc in scenes:
        for frame in sc['frames']:
            target = ROOT / frame['file']
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(BASE / frame['file'], target)
            check(f"{sc['scene']} {target.name} unchanged image", hashlib.sha256(target.read_bytes()).hexdigest() == frame['sha256'])
            check(f"{sc['scene']} {target.name} no future frame", sc['window_start'] <= frame['actual_pts_seconds'] < sc['window_end_exclusive'])

    base.ROOT = ROOT
    rows = []
    for sc in scenes:
        r = next(r for r in current if r['scene'] == sc['scene'])
        old = baseline[sc['scene']]
        body, manifest = separated.separated_body(sc, 'F', cfg)
        check(sc['scene'] + ' exact request reconstruction', base.sha(base.dumps(body).encode()) == r['request_sha256'])
        check(sc['scene'] + ' sixteen image references', r['image_count'] == 16 and r['images'] == old['images'])
        check(sc['scene'] + ' system prompt unchanged', r['messages'][0] == old['messages'][0])
        check(sc['scene'] + ' original user prompt preserved', r['messages'][1]['content'].split('\n\nCHRONOLOGICAL FRAMES\n')[0] == old['messages'][1]['content'])
        check(sc['scene'] + ' all sixteen explicit markers', r['messages'][1]['content'].count('[img]') == 16)
        check(sc['scene'] + ' independent request', len(body['messages']) == 2)
        check(sc['scene'] + ' settings and model unchanged', r['model_digest'] == old['model_digest'] and r['options'] == old['options'] and r['think'] == old['think'])
        row = {
            'scene': sc['scene'], 'baseline_order': ','.join(map(str, old['format']['order'])) or 'unclear',
            'followup_order': ','.join(map(str, r['format']['order'])) or 'unclear',
            'baseline_text': old['text'], 'followup_text': r['text'],
            'baseline_seconds': old['elapsed_seconds'], 'followup_seconds': r['elapsed_seconds'],
            'followup_load_seconds': r['response_json']['load_duration'] / 1e9,
            'baseline_prompt_tokens': old['response_json']['prompt_eval_count'],
            'followup_prompt_tokens': r['response_json']['prompt_eval_count'],
            'failure': r['failure'], 'format_issues': '|'.join(r['format']['issues']),
            'human_confirmed_leader': 9 if sc['scene'] == 'S2_bend' else '',
            'confirmed_wrong_leader_claims': 1 if sc['scene'] == 'S2_bend' else '',
            'review_status': 'user-confirmed leader in frames at 35.0 and 38.5 seconds' if sc['scene'] == 'S2_bend' else 'not human-verified',
            'review_note': {
                'S1_early': 'Both abstain on ORDER; no leader gold. Similar broad start context.',
                'S2_bend': 'Both assert leader1, contradicting user-confirmed leader9. Ranks2/3 unverified.',
                'S3_wide': 'ORDER unchanged; final leader identity unverified. No correctness score.',
                'S4_straight': 'ORDER leader1 versus body leader2: definite internal contradiction. Both conflict with provisional assistant leader9; no human gold.',
            }[sc['scene']],
            'order_body_leader_conflict': sc['scene'] == 'S4_straight',
            'racecard_prefix_match': r['format']['order'] == [1, 2, 3],
        }
        rows.append(row)
    with (ROOT / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    status = read(ROOT / 'evidence/run_status.json')
    check('first call sixteen decoded image batches', status['first_image_batches'] == 16)
    check('other three calls forty-eight decoded image batches', status['remaining_image_batches'] == 48)
    summary = {
        'new_generations': len(current), 'jev_generations': 0, 'total_generations_including_original': 20,
        'failures': sum(r['failure'] for r in current),
        'format_violations': sum(bool(r['format']['issues']) for r in current),
        'changed_order_scenes': [r['scene'] for r in rows if r['baseline_order'] != r['followup_order']],
        'racecard_prefix_match_scenes': [r['scene'] for r in rows if r['racecard_prefix_match']],
        'order_body_conflict_scenes': [r['scene'] for r in rows if r['order_body_leader_conflict']],
        'human_reviewed_scenes': ['S2_bend'], 'human_confirmed_leader': 9,
        'human_reviewed_scene_wrong_leader_in_both': True,
        'unreviewed_scenes_not_scored': ['S1_early', 'S3_wide', 'S4_straight'],
        'elapsed_seconds_sum': round(sum(r['elapsed_seconds'] for r in current), 3),
        'after_first_call_seconds': [r['elapsed_seconds'] for r in current[1:]],
        'after_first_call_median_seconds': statistics.median(r['elapsed_seconds'] for r in current[1:]),
        'checks_passed': len(checks),
        'limits': ['Single race, four calls, no repeats or new shuffle control.',
                   'Only S2 final-frame leader has user confirmation; remaining correctness unscored.',
                   'Explicit image separation and temporal text labels changed together.',
                   'Cold load and cache state prevent interpreting latency as controlled model-speed comparison.',
                   'Fixed-fact condition B and detector integration not tested in this followup.'],
    }
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'verification.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
