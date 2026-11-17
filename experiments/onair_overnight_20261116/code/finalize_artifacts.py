"""Package completed research without changing source video or model outputs."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
from common import ROOT, append, dump, now, status

verification = json.loads((ROOT / 'evidence/verification_final.json').read_text())
assert verification['checks_passed'] == 688
assert all(x['passed'] for x in verification['checks'])
assert len((ROOT / 'raw/request_ledger.jsonl').read_text().splitlines()) == 97

processes = []
for pid in (2050329, 2051507):
    command_file = Path(f'/proc/{pid}/cmdline')
    command = command_file.read_bytes().replace(b'\0', b' ').decode(errors='replace') if command_file.exists() else None
    processes.append({'pid': pid, 'still_present': command_file.exists(), 'command': command})
assert not any(x['still_present'] for x in processes), processes
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], text=True).strip()
dump(ROOT / 'evidence/completion.json', {
    'time_utc': now(), 'model_generations': 97, 'integrity_checks_passed': 688,
    'batch_processes': processes, 'gpu_at_completion': gpu,
    'state': 'completed; no research job scheduled to continue until 09:00',
    'human_review': 'pending; new reference annotations are assistant drafts',
    'next_review_kst': '2026-11-17 09:00:00+09:00',
})
status('handoff', 'complete', model_generations=97, integrity_checks_passed=688,
       report='MORNING_REPORT_KO.md', highlights='highlights/INDEX_KO.md',
       human_review='pending', remaining_background_jobs=0)

local_base = '/Users/ohead/Documents/ChatGPT/ON-AIR_프로젝트/experiments/onair_overnight_20261116/'
documents = ['README.md', 'MORNING_REPORT_KO.md', 'PLAN_KO.md',
             'highlights/INDEX_KO.md', 'review/USER_REVIEW_KO.md', 'review/USER_REVIEW.csv',
             'results/SUMMARY.json', 'results/geometry_comparison.csv',
             'results/localization_comparison.csv', 'evidence/completion.json']
documents += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'highlights').glob('*.jpg'))]
documents += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'review/geometry').glob('contact*.jpg'))]
with zipfile.ZipFile(ROOT / 'MORNING_REVIEW_BUNDLE.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for name in documents:
        path = ROOT / name
        if path.suffix == '.md':
            # Make only the packaged copies portable; preserve originals.
            import os
            data = path.read_text()
            for target in documents + ['README.md', 'MORNING_REPORT_KO.md', 'results/SUMMARY.json']:
                relative = os.path.relpath(target, str(Path(name).parent))
                data = data.replace(local_base + target, relative)
            z.writestr(name, data)
        else:
            z.write(path, name)
with zipfile.ZipFile(ROOT / 'MORNING_REVIEW_BUNDLE.zip') as z:
    assert z.testzip() is None

excluded = {'model_cache', '__pycache__', 'SHA256SUMS'}
paths = [p for p in sorted(ROOT.rglob('*')) if p.is_file()
         and not any(part in excluded for part in p.relative_to(ROOT).parts)
         and p.suffix not in ('.zip', '.pyc', '.tmp')]
manifest = ''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}\n' for p in paths)
(ROOT / 'SHA256SUMS').write_text(manifest)
print(json.dumps({'files_hashed': len(paths), 'review_bundle_bytes': (ROOT / 'MORNING_REVIEW_BUNDLE.zip').stat().st_size,
                  'gpu': gpu, 'checks': verification['checks_passed']}, ensure_ascii=False))
