"""Read-only TUAR/TUEV header and annotation inventory; no model training or EEG export."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import mne


def audit(root,kind):
    records=[]; labels=Counter(); rates=Counter(); channels=Counter(); subjects=set(); failures=[]
    for p in sorted((root/'edf').rglob('*.edf')):
        try:
            raw=mne.io.read_raw_edf(p,preload=False,verbose='ERROR')
            fs=float(raw.info['sfreq']);nch=len(raw.ch_names)
            rates[str(fs)]+=1;channels[str(nch)]+=1
            ann=p.with_suffix('.csv' if kind=='TUAR' else '.rec')
            counts=Counter();duration=[]
            if ann.exists():
                with ann.open(encoding='utf-8-sig') as f:
                    if kind=='TUAR':
                        for row in csv.DictReader(line for line in f if not line.lstrip().startswith('#') and line.strip()):
                            counts[row['label']]+=1
                            duration.append(float(row['stop_time'])-float(row['start_time']))
                    else:
                        for row in csv.reader(f):
                            if not row or row[0].lstrip().startswith('#'):continue
                            assert len(row)==4,(p,row)
                            counts[str(int(row[3]))]+=1;duration.append(float(row[2])-float(row[1]))
                labels.update(counts)
            if kind=='TUAR':subjects.add(p.stem.split('_')[0])
            rec={'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sampling_rate':fs,'channels':nch,
                 'channel_names':raw.ch_names,'duration_seconds':raw.n_times/fs,'annotation_exists':ann.exists(),
                 'annotation_sha256':hashlib.sha256(ann.read_bytes()).hexdigest() if ann.exists() else None,
                 'annotation_counts':dict(counts),'annotation_duration_range':[min(duration),max(duration)] if duration else None}
            records.append(rec)
        except Exception as exc:failures.append({'path':str(p),'error':repr(exc)})
    return {'dataset':kind,'root':str(root),'files':len(records),'sampling_rate_histogram':dict(rates),
            'channel_count_histogram':dict(channels),'annotation_label_counts':dict(labels),
            'subjects_from_filename':len(subjects) if kind=='TUAR' else None,
            'subject_identity_note':'TUAR filename subject prefix' if kind=='TUAR' else 'TUEV eval directory names are evaluation indices, not verified patient identities; no new subject split generated',
            'missing_annotations':sum(not r['annotation_exists'] for r in records),'failures':failures,'records':records}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tuar',type=Path,required=True);p.add_argument('--tuev',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();results=[audit(a.tuar,'TUAR'),audit(a.tuev,'TUEV')]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'scope':'read-only headers and annotations; no predictions, training or test scoring','datasets':results},indent=2),encoding='utf-8')
    for r in results:print(json.dumps({k:v for k,v in r.items() if k!='records'}),flush=True)
