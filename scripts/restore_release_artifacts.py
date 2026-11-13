"""Download verified experiment artifacts. Does not download licensed EEG datasets."""
import argparse,gzip,hashlib,io,json,pathlib,shutil,tarfile,tempfile
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
class Joined(io.RawIOBase):
    def __init__(self,paths):self.paths=iter(paths);self.f=None
    def readable(self):return True
    def read(self,n=-1):
        out=[];remaining=n
        while remaining:
            if self.f is None:
                try:self.f=next(self.paths).open('rb')
                except StopIteration:break
            b=self.f.read(remaining)
            if not b:self.f.close();self.f=None;continue
            out.append(b)
            if n>=0:remaining-=len(b)
        return b''.join(out)
    def close(self):
        if self.f:self.f.close()
        super().close()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--destination',type=pathlib.Path,required=True,help='new empty directory');ap.add_argument('--manifest',type=pathlib.Path,default=ROOT/'storage/release_artifacts_manifest.json');args=ap.parse_args()
    dest=args.destination.resolve()
    if dest.exists() and any(dest.iterdir()):raise SystemExit('Destination must be empty; existing experiments will not be overwritten')
    dest.mkdir(parents=True,exist_ok=True);m=json.loads(args.manifest.read_text(encoding='utf-8'))
    assert m['all_parts_remote_verified']
    with tempfile.TemporaryDirectory(prefix='omnieeg-restore-') as scratch:
        paths=[]
        for item in m['parts']:
            p=pathlib.Path(scratch)/item['name'];h=hashlib.sha256();size=0
            print('Downloading',item['name'],flush=True)
            with requests.get(item['url'],stream=True,timeout=(40,300)) as r:
                r.raise_for_status()
                with p.open('wb') as f:
                    for b in r.iter_content(4*1024**2):f.write(b);h.update(b);size+=len(b)
            assert size==item['bytes'] and h.hexdigest()==item['sha256'],item['name'];paths.append(p)
        expected={x['path']:x for x in m['files']};seen=set()
        with Joined(paths) as joined,tarfile.open(fileobj=joined,mode='r|gz') as tar:
            for entry in tar:
                target=(dest/entry.name).resolve()
                if not target.is_relative_to(dest) or not entry.isfile():raise ValueError('Unsafe archive member')
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(entry) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024**2)
                if entry.name=='file_manifest.json':continue
                info=expected[entry.name];assert target.stat().st_size==info['bytes'] and sha(target)==info['sha256'];seen.add(entry.name)
        assert seen==set(expected)
    print('Restored and verified',len(seen),'files into',dest)
if __name__=='__main__':main()
