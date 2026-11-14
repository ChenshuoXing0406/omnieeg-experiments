"""Upload the reviewed daily archive using existing Git authentication; never stores credentials."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import requests

REPO='ChenshuoXing0406/omnieeg-experiments'
API='https://api.github.com/repos/'+REPO


def session():
    env=os.environ.copy();env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
    r=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,env=env,timeout=30)
    assert r.returncode==0,'Existing Git credential unavailable (secrets suppressed)'
    values=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
    assert values.get('password'),'No existing Git authentication'
    s=requests.Session();s.headers.update({'Authorization':'Bearer '+values['password'],'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    return s


def check(r):
    if not r.ok:raise RuntimeError(f'GitHub API status {r.status_code}: {r.text[:500]}')
    return r.json()


def main(a):
    s=session();m=json.loads(a.manifest.read_text(encoding='utf-8'))
    releases=check(s.get(API+'/releases',params={'per_page':100},timeout=30))
    release=next((x for x in releases if x['tag_name']==m['tag']),None)
    if a.publish:
        assert release and release['draft'] and m['all_parts_remote_verified']
        # Draft tags normally do not yet exist. Never move an existing mismatched tag.
        tag=s.get(API+'/git/ref/tags/'+m['tag'],timeout=30)
        if tag.status_code==200:assert tag.json()['object']['sha']==a.target,'Existing tag points elsewhere'
        elif tag.status_code!=404:check(tag)
        release=check(s.patch(API+f"/releases/{release['id']}",json={'target_commitish':a.target,'draft':False,'body':a.body_file.read_text(encoding='utf-8')},timeout=30))
        ref=check(s.get(API+'/git/ref/tags/'+m['tag'],timeout=30));assert ref['object']['sha']==a.target
        print(json.dumps({'release_url':release['html_url'],'draft':release['draft'],'tag_commit':a.target}),flush=True);return
    if release is None:
        release=check(s.post(API+'/releases',json={'tag_name':m['tag'],'target_commitish':a.target,'name':'2026-11-14 MI与SSVEP实验完整附件','body':a.body_file.read_text(encoding='utf-8'),'draft':True},timeout=30))
    assert release['draft'],'Refusing to alter an existing public release'
    upload_url=release['upload_url'].split('{')[0]
    assert upload_url.startswith('https://uploads.github.com/repos/'+REPO+'/releases/'),'Unexpected upload destination'
    verified=[]
    for part in m['parts']:
        path=a.archive_dir/part['name'];assert path.stat().st_size==part['bytes']
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==part['sha256']
        existing=check(s.get(API+f"/releases/{release['id']}/assets",params={'per_page':100},timeout=30))
        asset=next((x for x in existing if x['name']==part['name']),None)
        if asset is None:
            with path.open('rb') as f:
                asset=check(s.post(upload_url,params={'name':part['name']},data=f,headers={'Content-Type':'application/octet-stream'},timeout=(30,600)))
        assert asset['size']==part['bytes'],part['name']
        if asset.get('digest'):
            assert asset['digest']=='sha256:'+part['sha256'],part['name']
        else:
            h=hashlib.sha256()
            with s.get(asset['url'],headers={'Accept':'application/octet-stream'},stream=True,timeout=(30,300)) as r:
                r.raise_for_status()
                for block in r.iter_content(4*1024**2):h.update(block)
            assert h.hexdigest()==part['sha256'],part['name']
        # Draft releases expose temporary untagged URLs; recovery must use final tag URLs.
        part['url']='https://github.com/'+REPO+'/releases/download/'+m['tag']+'/'+part['name']
        verified.append({'name':asset['name'],'bytes':asset['size'],'digest':asset.get('digest'),'id':asset['id']})
        print(json.dumps({'uploaded_and_verified':asset['name'],'bytes':asset['size']}),flush=True)
    m['all_parts_remote_verified']=True
    a.manifest.write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding='utf-8')
    receipt={'release_id':release['id'],'release_url':release['html_url'],'draft':True,'assets':verified,'all_parts_remote_verified':True}
    (a.archive_dir/'upload_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    # Attach the exact public recovery manifest as a small independently verifiable asset.
    assets=check(s.get(API+f"/releases/{release['id']}/assets",params={'per_page':100},timeout=30))
    asset=next((x for x in assets if x['name']==a.manifest.name),None)
    if asset is None:
        with a.manifest.open('rb') as f:
            asset=check(s.post(upload_url,params={'name':a.manifest.name},data=f,headers={'Content-Type':'application/json'},timeout=60))
    assert asset['size']==a.manifest.stat().st_size
    if asset.get('digest'):assert asset['digest']=='sha256:'+hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    print(json.dumps({'draft_release_ready':release['html_url'],'parts':len(verified),'manifest_asset':asset['browser_download_url']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','archive-dir','body-file']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--target',required=True);p.add_argument('--publish',action='store_true');main(p.parse_args())
