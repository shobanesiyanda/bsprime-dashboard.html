#!/usr/bin/env node
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {getHistoricalRates}=require('dukascopy-node');
function arg(n,d=null){const i=process.argv.indexOf(n);return i>=0?process.argv[i+1]:d}
const manifest=arg('--manifest'),outdir=arg('--outdir','cal_raw'),shard=Number(arg('--shard','0')),shards=Number(arg('--shards','12')),concurrency=Number(arg('--concurrency','3'));
const START=new Date(arg('--start','2025-07-21T00:00:00Z')),END=new Date(arg('--end','2026-01-19T00:00:00Z'));
const FIRST_MAX=new Date(START.getTime()+3*86400000),LAST_MIN=new Date(END.getTime()-3*86400000);
if(!manifest)throw new Error('--manifest required');fs.mkdirSync(outdir,{recursive:true});
const all=fs.readFileSync(manifest,'utf8').trim().split(/\r?\n/).slice(1).map(x=>{const [instrument_id,canonical_instrument,provider_name,role]=x.split('\t');return {instrument_id,canonical_instrument,provider_name,role}});
function bucket(x){return crypto.createHash('sha1').update(x.canonical_instrument+'|'+x.instrument_id).digest().readUInt32BE(0)%shards}
const work=all.filter(x=>bucket(x)===shard),evidence=[];const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function safe(x){return crypto.createHash('sha1').update(x).digest('hex').slice(0,16)}
function validBar(b){const o=+b.open,h=+b.high,l=+b.low,c=+b.close;return[o,h,l,c].every(Number.isFinite)&&h>=Math.max(o,c)&&l<=Math.min(o,c)&&h>=l}
async function one(x){let last='';for(let attempt=1;attempt<=3;attempt++){try{
 const d=await getHistoricalRates({instrument:x.instrument_id,dates:{from:START,to:END},timeframe:'m1',format:'json',priceType:'bid',batchSize:10,pauseBetweenBatchesMs:220});
 const seen=new Set(),rows=[];let invalid=0,dupes=0;
 for(const b of d||[]){const t=+b.timestamp;if(!Number.isFinite(t)||t<START.getTime()||t>=END.getTime())continue;if(seen.has(t)){dupes++;continue}seen.add(t);if(!validBar(b)){invalid++;continue}rows.push([new Date(t).toISOString(),b.open,b.high,b.low,b.close,b.close,'','',b.volume??0,'DUKASCOPY_STRUCTURAL_CALIBRATION_M1',x.canonical_instrument])}
 rows.sort((a,b)=>a[0].localeCompare(b[0]));const first=rows.length?new Date(rows[0][0]):null,lastTs=rows.length?new Date(rows.at(-1)[0]):null;
 const coverage=!!first&&!!lastTs&&first<=FIRST_MAX&&lastTs>=LAST_MIN;const ok=rows.length>=1000&&invalid===0&&dupes===0&&coverage;let file='';
 if(ok){file=path.join(outdir,`${safe(x.canonical_instrument+'|'+x.instrument_id)}.csv`);const f=fs.createWriteStream(file);f.write('timestamp,open,high,low,close,bid,ask,spread,volume,source,asset\n');for(const r of rows)f.write(r.join(',')+'\n');await new Promise(res=>f.end(res))}
 evidence.push({...x,rows:rows.length,invalid_ohlc:invalid,duplicate_timestamps:dupes,full_window_coverage:coverage,status:ok?'PASS':'FAIL',file,attempt,first:rows.length?rows[0][0]:'',last:rows.length?rows.at(-1)[0]:''});return
 }catch(e){last=String(e);await sleep(600*attempt)}}evidence.push({...x,rows:0,invalid_ohlc:0,duplicate_timestamps:0,full_window_coverage:false,status:'FAIL',file:'',attempt:3,error:last})}
async function pool(){let i=0;await Promise.all(Array.from({length:concurrency},async()=>{while(true){const j=i++;if(j>=work.length)return;await one(work[j])}}))}
(async()=>{await pool();fs.writeFileSync(path.join(outdir,'ACQUISITION_EVIDENCE.json'),JSON.stringify({window_start:START.toISOString(),window_end_exclusive:END.toISOString(),shard,shards,assets_scheduled:work.length,assets_pass:evidence.filter(x=>x.status==='PASS').length,evidence},null,2));console.log(JSON.stringify({shard,assets_scheduled:work.length,assets_pass:evidence.filter(x=>x.status==='PASS').length},null,2))})();
