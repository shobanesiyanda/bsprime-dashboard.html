#!/usr/bin/env python3
from pathlib import Path
src=Path('trailaris/acquire_market_data.py').read_text()
src=src.replace("'-t','m1'","'-t','m5'")
src=src.replace("/{symbol}/1m/{stem}","/{symbol}/5m/{stem}")
src=src.replace("{symbol}-1m-","{symbol}-5m-")
src=src.replace("'granularity':60","'granularity':300")
src=src.replace("OUT=Path('trailaris/raw')","OUT=Path('trailaris/raw_discovery')")
exec(compile(src,'trailaris/acquire_market_data_discovery.py','exec'),{'__name__':'__main__'})
