"""Prepare a public MI pilot; this is not the competition dataset/protocol."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, resample_poly

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2',
            'C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']


def main():
    xs, ys, subjects, trials, artifacts, sources = [], [], [], [], [], []
    sos = butter(4, [0.3, 40], btype='bandpass', fs=250, output='sos')
    for subject in range(1, 10):
        path = ROOT / f'data/raw/BNCI2014_001/A{subject:02d}T.mat'
        runs = loadmat(path, simplify_cells=True)['data']
        sources.append({'file': path.name, 'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()})
        count = 0
        for run_id, run in enumerate(runs):
            labels = np.atleast_1d(run['y']).astype('int64')
            if labels.size == 0:
                continue  # Detect calibration from empty labels, including A04T exception.
            assert int(run['fs']) == 250
            assert list(run['classes']) == ['left hand', 'right hand', 'feet', 'tongue']
            raw = np.asarray(run['X'][:, :22], dtype=np.float64).T
            assert np.isfinite(raw).all()
            raw -= raw.mean(axis=0, keepdims=True)  # CAR across EEG only.
            raw = sosfiltfilt(sos, raw, axis=-1)  # Independent recording; no cross-subject fit.
            starts = np.atleast_1d(run['trial']).astype('int64') - 1  # MATLAB -> Python.
            flags = np.atleast_1d(run['artifacts']).astype('int64')
            assert len(starts) == len(labels) == len(flags)
            for trial, (start, label, flag) in enumerate(zip(starts, labels, flags)):
                signal = raw[:, start + 500:start + 1500]  # Trial-relative [2,6) seconds.
                assert signal.shape == (22, 1000) and 1 <= label <= 4
                signal = resample_poly(signal, 4, 5, axis=-1).astype('float32')
                assert signal.shape == (22, 800)
                xs.append(signal); ys.append(label - 1); subjects.append(subject)
                trials.append(f'A{subject:02d}T-run{run_id}-trial{trial}')
                artifacts.append(flag); count += 1
        assert count == 288, (subject, count)
        print(f'A{subject:02d}T: {count} trials', flush=True)
    out = ROOT / 'data/processed/BNCI2014_001_T_pilot.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=np.stack(xs), y=np.array(ys), subject=np.array(subjects),
                        trial_id=np.array(trials), artifact=np.array(artifacts),
                        channel_names=np.array(CHANNELS), sfreq=np.array(200), unit=np.array('uV'))
    metadata = {
        'scope': 'PUBLIC_DATA_PILOT_NOT_COMPETITION', 'sessions': ['T'],
        'shape': [len(xs), 22, 800], 'source_sampling_rate': 250, 'target_sampling_rate': 200,
        'epoch_seconds_relative_to_trial_start': [2, 6], 'index_conversion': 'MATLAB trial - 1',
        'preprocessing': ['22 EEG only; EOG excluded', 'CAR',
                          'continuous-run 4th-order 0.3-40 Hz Butterworth sosfiltfilt',
                          '4-second crop', 'polyphase resample 4/5'],
        'unit': 'uV', 'training_scale': 'divide by 100, following CBraMod dataset loader',
        'artifacts': 'kept, source flags preserved', 'sources': sources,
        'channel_names': CHANNELS, 'labels': ['left_hand','right_hand','feet','tongue'],
        'npz_sha256': hashlib.file_digest(out.open('rb'), 'sha256').hexdigest(),
        'split': {'train': [1,2,3,4,5], 'val': [6,7], 'test': [8,9]},
        'split_note': 'Author-style 5/2/2 subjects, T session only. Not official 8:1:1. Development seed 42 is not an official seed.',
        'data_source': 'https://bnci-horizon-2020.eu/database/data-sets',
        'license': 'CC BY-ND 4.0; Institute for Knowledge Discovery, Graz University of Technology',
        'preprocessing_note': 'Local offline pilot; differs from author causal per-trial filtering and FFT resampling.',
    }
    out.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out), 'shape': metadata['shape']}), flush=True)


if __name__ == '__main__':
    main()
