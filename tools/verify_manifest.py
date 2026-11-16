"""Verify packaged hashes and detect unexpected distribution files (stdlib only)."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]

def generated(path):
    parts = path.parts
    return ('__pycache__' in parts or '.git' in parts or '.venv' in parts
            or parts[0] == 'local_runs'
            or parts[:2] in [('systems','verification'),('systems','results')]
            or path.as_posix() == 'systems/data/evaluation.parquet')

def verify(root=ROOT):
    manifest = json.loads((root/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    bad = []
    for name, digest in manifest['files'].items():
        p = root/name
        if not p.resolve().is_relative_to(root.resolve()) or p.is_symlink():
            bad.append(name)
        elif not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            bad.append(name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and not generated(p.relative_to(root))}
    unexpected = sorted(actual - set(manifest['files']) - {'MANIFEST_SHA256.json'})
    return {'status':'FAIL' if bad or unexpected else 'PASS',
            'files_checked':len(manifest['files']), 'mismatches':bad,
            'unexpected_files':unexpected, 'training_performed':False,
            'note':'Local generated outputs are ignored for integrity verification; never include them in a public release.'}

if __name__ == '__main__':
    result = verify()
    print(json.dumps(result,indent=2))
    sys.exit(result['status'] != 'PASS')
