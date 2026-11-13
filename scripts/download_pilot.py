"""Fetch original public pilot data and official CBraMod weights, with provenance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]


def fetch(item):
    url, path = item
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = path.with_suffix(path.suffix + '.part')
        with requests.get(url, stream=True, timeout=(20, 90)) as response, temp.open('wb') as f:
            response.raise_for_status()
            final_url = response.url.split('?')[0]
            expected = response.headers.get('Content-Length')
            for chunk in response.iter_content(1024 * 1024):
                f.write(chunk)
        if expected and temp.stat().st_size != int(expected):
            raise RuntimeError(f'Incomplete download: {path.name}')
        if path.suffix == '.mat' and not temp.open('rb').read(20).startswith(b'MATLAB'):
            raise ValueError(f'Not a MATLAB file: {path.name}')
        temp.replace(path)
    else:
        final_url = None
    record = {
        'source_url': url, 'resolved_url': final_url,
        'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
        'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
        'checked_at': datetime.now(timezone.utc).isoformat(),
    }
    path.with_suffix(path.suffix + '.source.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 10)))
    a = p.parse_args()
    if not all(1 <= i <= 9 for i in a.subjects):
        p.error('Subjects must be 1..9')
    jobs = [(f'https://bnci-horizon-2020.eu/database/data-sets/001-2014/A{i:02d}T.mat',
             ROOT / 'data/raw/BNCI2014_001' / f'A{i:02d}T.mat') for i in a.subjects]
    jobs.append(('https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth',
                 ROOT / 'checkpoints/cbramod.pth'))
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(fetch, jobs))


if __name__ == '__main__':
    main()
