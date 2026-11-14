"""Prepare the fixed MI/SSVEP protocol from locally obtained public MAT files.

This wrapper leaves the historical model, training and preprocessing sources
unchanged. Source data must be acquired separately under their original terms.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt
import prepare_aligned as aligned


def prepare_ssvep(raw):
    if (aligned.ROOT / 'processed/ssvep_2s/manifest.json').exists():
        raise FileExistsError('SSVEP cache exists; use a fresh experiment directory for new preprocessing.')
    channels = [line.split()[-1] for line in (raw / '64-channels.loc').read_text().splitlines() if line.strip()]
    assert len(channels) == 64
    xs, labels, subjects, ids, sources = [], [], [], [], []
    sos = butter(4, [6, 90], fs=250, btype='bandpass', output='sos')
    for subject in range(1, 36):
        path = raw / f'S{subject}.mat'
        signal = np.asarray(loadmat(path)['data'], dtype=np.float64)
        assert signal.shape == (64, 1500, 40, 6)
        filtered = sosfiltfilt(sos, signal, axis=1)
        x = resample_poly(filtered[:, 160:660, :, :], 4, 5, axis=1) / 100
        x = x.transpose(2, 3, 0, 1).reshape(240, 64, 2, 200).astype('float32')
        # Match the complete-subject reference check in the frozen preparer.
        ref = sosfiltfilt(sos, signal.transpose(2, 3, 0, 1).reshape(240, 64, 1500), axis=-1)
        ref = resample_poly(ref[:, :, 160:660], 4, 5, axis=-1).astype('float32').reshape(240, 64, 2, 200) / 100
        assert np.allclose(x, ref, rtol=2e-6, atol=2e-6)
        xs.append(x)
        labels.extend(np.repeat(np.arange(40), 6))
        subjects.extend([subject] * 240)
        ids.extend(f'S{subject:02d}-target{t}-block{b}' for t in range(40) for b in range(6))
        sources.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    aligned.save('ssvep_2s', np.concatenate(xs), labels, subjects, ids, channels,
                 {'train':list(range(1,6))+list(range(9,30)), 'val':[6,30,31,32], 'test':[7,8,33,34,35]},
                 {'source':'Tsinghua Benchmark', 'sources':sources, 'offline_context':True,
                  'preprocessing':'64 channels; full-trial order4 zero-phase 6-90Hz; crop160:660 at250Hz; polyphase to200Hz; uV/100',
                  'test_history':'Legacy test subjects have been used in previous team work; not a team-blind test.',
                  'portability_helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['mi', 'ssvep_2s', 'both'], required=True)
    p.add_argument('--raw-root', required=True, type=Path,
                   help='Contains mi/A01T.mat ... A09T.mat and ssvep/S1.mat ... S35.mat, 64-channels.loc.')
    a = p.parse_args()
    if a.task in ['mi', 'both']:
        # Preserve the historical MI function; its legacy OLD/raw/mi path is
        # mapped to the explicitly supplied raw data directory.
        class PublicRoot:
            def __truediv__(self, suffix):
                return a.raw_root / Path(suffix).relative_to('raw')
        aligned.OLD = PublicRoot()
        aligned.mi()
    if a.task in ['ssvep_2s', 'both']:
        prepare_ssvep(a.raw_root / 'ssvep')


if __name__ == '__main__':
    main()
