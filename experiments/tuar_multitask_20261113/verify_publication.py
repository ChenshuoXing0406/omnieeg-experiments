"""Verify frozen source hashes and the portable nine-run schedule (no GPU/data)."""
import ast
import hashlib
import json
from pathlib import Path
from launch import commands

ROOT=Path(__file__).resolve().parent


def main():
    manifest=json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))
    for path, expected in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    for path in ROOT.rglob('*.py'):
        if '__pycache__' not in path.parts:
            ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    cmds=commands('all')
    names=[c[c.index('--name')+1] for c in cmds]
    result=json.loads((ROOT/'results_summary.json').read_text(encoding='utf-8'))
    assert len(cmds)==len(set(names))==9
    assert set(names)==set(result['runs'])
    for c in cmds:
        name=c[c.index('--name')+1];r=result['runs'][name]
        assert c[c.index('--tasks')+1:c.index('--steps')]==r['config']['tasks']
        assert c[c.index('--mode')+1]==r['config']['mode']
        assert c[c.index('--steps')+1]=='1500'
        assert c[c.index('--batch')+1]=='16'
        assert c[c.index('--seed')+1]=='42'
        assert not r['completion']['test_evaluated']
        for key in ['model.py','train.py']:
            assert r['config']['sources'][key]==manifest['sha256'][key]
        assert max(r['completion']['checkpoint_reload_max_probability_error'].values())==0
    # Check tracked publication contents separately with git; local datasets and
    # checkpoints may legitimately exist after preparation and are ignored.
    print('PASS: original source hashes, Python syntax, nine-run queue/config correspondence, validation-only results')


if __name__=='__main__':
    main()
