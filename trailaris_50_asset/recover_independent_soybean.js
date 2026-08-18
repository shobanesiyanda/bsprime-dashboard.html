#!/usr/bin/env node
const fs=require('fs'),path=require('path');const {getHistoricalRates}=require('dukascopy-node');
function arg(n,d=null){const i=process.argv.indexOf(n);return i>=0?process.argv[i+1]:d}
const START=new Date(arg('--start','2024-12-16T00:00:00Z')),END=new Date(arg('--end','2025-07-14T00:00:00Z')),OUT=arg('--outdir','raw16');const ID='soybeancmdusx',ASSET='SOYBEAN.CMD';fs.mkdirSync(OUT,{recursive:true});const S=START.getTime(),E=END.getTime();
function valid(o,h,l,c){return[o,h,l,c].every(Number.isFinite)&&h>=Math.max(o,c)&&l<=Math.min(o,c)&&h>=l}
async function pull(from,to){return getHistoricalRates({instrument:ID,dates:{from,to},timeframe:'m1',format:'json',priceType:'bid',batchSize:10,pauseBetweenBatchesMs:250})}
function merge(map,d,c){for(const b of d||[]){c.total++;const t=+b.timestamp;if(!Number.isFinite(t)||t<S||t>=E)continue;const o=+b.open,h=+b.high,l=+b.low,cl=+b.close;if(!valid(o,h,l,cl)){c.invalid++;continue}const sig=[o,h,l,cl,+b.volume||0].join('|');if(map.has(t)){c.dupes++;if(map.get(t).sig!==sig)c.conflicts++;continue}map.set(t,{sig,row:[new Date(t).toISOString(),o,h,l,cl,cl,'','',b.volume??0,'DUKASCOPY_EXACT_INDEPENDENT_RESEARCH_PROXY',ASSET]})}}
(async()=>{const map=new Map(),c={total:0,invalid:0,dupes:0,conflicts:0};
  // Two independent query shapes: early estate plus the provider shape that the
  // common-endpoint diagnostic proved returns the July tail. Rows are clipped to
  // the frozen validation interval; the overlap is deterministic and conflicts fatal.
  const earlyEnd=new Date('2025-05-15T00:00:00Z');const tailStart=new Date('2025-05-01T00:00:00Z');const tailEnd=new Date('2025-07-21T00:00:00Z');
  merge(map,await pull(START,earlyEnd),c);merge(map,await pull(tailStart,tailEnd),c);
  const rows=[...map.entries()].sort((a,b)=>a[0]-b[0]).map(x=>x[1].row),first=rows.length?new Date(rows[0][0]):null,last=rows.length?new Date(rows.at(-1)[0]):null;const coverage=!!first&&!!last&&first<=new Date(S+3*86400000)&&last>=new Date(E-4*86400000);const ok=rows.length>=1000&&coverage&&c.invalid===0&&c.conflicts===0;
  if(ok){const f=fs.createWriteStream(path.join(OUT,ASSET+'_M1_normalized.csv'));f.write('timestamp,open,high,low,close,bid,ask,spread,volume,source,asset\n');for(const r of rows)f.write(r.join(',')+'\n');await new Promise(res=>f.end(res));}
  const ev={instrument_id:ID,canonical_instrument:ASSET,provider_name:'SOYBEAN.CMD/USX',rows:rows.length,total_provider_rows:c.total,invalid_ohlc:c.invalid,duplicate_timestamps_deduplicated:c.dupes,conflicting_duplicates:c.conflicts,first:first?first.toISOString():'',last:last?last.toISOString():'',full_window_coverage:coverage,status:ok?'PASS':'FAIL',file:ok?path.join(OUT,ASSET+'_M1_normalized.csv'):'',normalization:'ISOLATED_TWO_QUERY_PROVIDER_RECOVERY__EARLY_PLUS_PROVEN_TERMINAL_SHAPE__CLIPPED_TO_FROZEN_WINDOW'};fs.writeFileSync(path.join(OUT,'SOYBEAN_RECOVERY_EVIDENCE.json'),JSON.stringify(ev,null,2));console.log(JSON.stringify(ev,null,2));if(!ok)process.exitCode=2;})();
