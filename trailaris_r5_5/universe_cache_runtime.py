from __future__ import annotations
import os,pickle,runpy,sys
from pathlib import Path

CACHE_ENV='TRAILARIS_R55_UNIVERSE_CACHE'

def load_or_build(base,rawdir:Path):
    p=os.getenv(CACHE_ENV,'').strip()
    if p and Path(p).exists():
        with open(p,'rb') as h:data,v,cands,opps,fcache=pickle.load(h)
        if len(data)!=34:raise RuntimeError(f'cached 34-route gate failed: {len(data)}/34')
        return data,v,cands,opps,fcache
    return base.build_universe(rawdir)

def install_patch():
    import R5_BASE_CAUSAL_ENGINE as base
    original=base.build_universe
    def cached(rawdir,*args,**kwargs):
        p=os.getenv(CACHE_ENV,'').strip()
        if p and Path(p).exists():
            with open(p,'rb') as h:data,v,cands,opps,fcache=pickle.load(h)
            if len(data)!=34:raise RuntimeError(f'cached 34-route gate failed: {len(data)}/34')
            return data,v,cands,opps,fcache
        return original(rawdir,*args,**kwargs)
    base.build_universe=cached

def run_script(script:str,args:list[str]):
    install_patch();sys.argv=[script,*args];runpy.run_path(script,run_name='__main__')

if __name__=='__main__':
    if len(sys.argv)<2:raise SystemExit('usage: universe_cache_runtime.py SCRIPT [ARGS...]')
    run_script(sys.argv[1],sys.argv[2:])
