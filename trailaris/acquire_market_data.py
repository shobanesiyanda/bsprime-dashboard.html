#!/usr/bin/env python3
from pathlib import Path
import subprocess
# Reuse the already-approved acquisition implementation from its immutable commit,
# changing only data granularity/output for the discovery pass.
subprocess.run(['git','fetch','--quiet','--depth=20','origin','trailaris-data-runtime-20260813'],check=False)
src=subprocess.check_output(['git','show','bccbe92864d8f1ac8eaf7f19151c4faa7b47272f:trailaris/acquire_market_data.py'],text=True)
src=src.replace("'-t','m1'","'-t','m5'")
src=src.replace("{symbol}-1m-","{symbol}-5m-")
src=src.replace("/{symbol}/1m/{stem}","/{symbol}/5m/{stem}")
src=src.replace("'granularity':60","'granularity':300")
src=src.replace("OUT=Path('trailaris/raw')","OUT=Path('trailaris/raw_discovery')")
exec(compile(src,'trailaris/acquire_market_data_discovery_runtime.py','exec'),{'__name__':'__main__'})
