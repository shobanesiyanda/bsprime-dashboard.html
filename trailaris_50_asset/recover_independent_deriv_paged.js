#!/usr/bin/env node
const fs=require('fs'),path=require('path');
const WebSocket=require('ws');
function arg(n,d=null){const i=process.argv.indexOf(n);return i>=0?process.argv[i+1]:d}
const START=new Date(arg('--start','2024-12-23T00:00:00Z'));
const END=new Date(arg('--end','2025-07-21T00:00:00Z'));
const OUT=arg('--outdir','raw_deriv5');
const PACE_MS=Number(arg('--pace-ms','2500'));
const PAGE_COUNT=Number(arg('--count','5000'));
const ITEMS=[['V75','R_75'],['V50','R_50'],['V100','R_100'],['V25','R_25'],['V10','R_10']];
const S=START.getTime(),E=END.getTime();fs.mkdirSync(OUT,{recursive:true});
if(!(S<E))throw new Error('invalid start/end');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function valid(o,h,l,c){return[o,h,l,c].every(Number.isFinite)&&h>=Math.max(o,c)&&l<=Math.min(o,c)&&h>=l}
async function openClient(){
 const ws=new WebSocket('wss://ws.binaryws.com/websockets/v3?app_id=1089');
 await new Promise((resolve,reject)=>{const t=setTimeout(()=>reject(new Error('open timeout')),20000);ws.once('open',()=>{clearTimeout(t);resolve()});ws.once('error',reject)});
 let rid=1;const pending=new Map();
 ws.on('message',m=>{let x;try{x=JSON.parse(m.toString())}catch{return};const p=pending.get(x.req_id);if(!p)return;pending.delete(x.req_id);x.error?p.reject(new Error(`${x.error.code||''}:${x.error.message||''}`)):p.resolve(x)});
 const req=o=>new Promise((resolve,reject)=>{const id=rid++;pending.set(id,{resolve,reject});ws.send(JSON.stringify({...o,req_id:id}));setTimeout(()=>{if(pending.has(id)){pending.delete(id);reject(new Error('request timeout'))}},45000)});
 return {ws,req};
}
async function requestPage(client,symbol,endEpoch,stats){
 let last='';
 for(let attempt=1;attempt<=12;attempt++){
  try{
   // Deriv rejects subscribe=0 for this historical request. Omit subscribe entirely.
   const x=await client.req({ticks_history:symbol,end:String(endEpoch),style:'candles',granularity:60,count:PAGE_COUNT,adjust_start_time:1});
   stats.requests++;await sleep(PACE_MS);return x;
  }catch(e){
   last=String(e);
   if(last.includes('RateLimit')||last.includes('TooManyRequests')){stats.rate_limit_backoffs++;const ms=Math.min(120000,15000*Math.max(1,stats.rate_limit_backoffs));console.error(JSON.stringify({symbol,event:'RATE_LIMIT_BACKOFF',attempt,ms,error:last}));await sleep(ms)}
   else{console.error(JSON.stringify({symbol,event:'REQUEST_RETRY',attempt,error:last}));await sleep(Math.min(15000,1200*attempt))}
  }
 }
 throw new Error(last);
}
async function one(asset,symbol){
 let client=await openClient();const map=new Map();const stats={requests:0,rate_limit_backoffs:0,reconnects:0,duplicates:0,conflicts:0,invalid:0,pages:0};
 try{
  let endEpoch=Math.floor((E-1)/1000),previousEarliest=null;
  while(true){
   let x;
   try{x=await requestPage(client,symbol,endEpoch,stats)}catch(e){
    try{client.ws.close()}catch{};stats.reconnects++;await sleep(8000);client=await openClient();x=await requestPage(client,symbol,endEpoch,stats);
   }
   const cs=x.candles||[];if(!cs.length)throw new Error(`empty page before requested start at end=${endEpoch}`);
   const epochs=cs.map(c=>+c.epoch).filter(Number.isFinite);if(!epochs.length)throw new Error('page has no finite epochs');
   const earliest=Math.min(...epochs);stats.pages++;
   for(const c of cs){
    const t=(+c.epoch)*1000;if(!Number.isFinite(t)||t<S||t>=E)continue;
    const o=+c.open,h=+c.high,l=+c.low,cl=+c.close;if(!valid(o,h,l,cl)){stats.invalid++;continue}
    const sig=[o,h,l,cl].join('|');if(map.has(t)){stats.duplicates++;if(map.get(t).sig!==sig)stats.conflicts++;continue}
    map.set(t,{sig,row:[new Date(t).toISOString(),o,h,l,cl,'','','',0,'DERIV_OFFICIAL_M1_CANDLES',asset]});
   }
   console.log(JSON.stringify({asset,page:stats.pages,rows:map.size,earliest:new Date(earliest*1000).toISOString(),end_cursor:new Date(endEpoch*1000).toISOString(),rate_limit_backoffs:stats.rate_limit_backoffs}));
   if(earliest*1000<=S)break;
   if(previousEarliest!==null&&earliest>=previousEarliest)throw new Error(`non-retreating history cursor ${earliest} >= ${previousEarliest}`);
   previousEarliest=earliest;endEpoch=earliest-1;
   if(stats.pages>100)throw new Error('page safety limit exceeded');
  }
  const rows=[...map.entries()].sort((a,b)=>a[0]-b[0]).map(v=>v[1].row);const first=rows.length?new Date(rows[0][0]):null,last=rows.length?new Date(rows.at(-1)[0]):null;
  const coverage=!!first&&!!last&&first<=new Date(S+60000)&&last>=new Date(E-2*60000);
  const ok=rows.length>=250000&&coverage&&stats.invalid===0&&stats.conflicts===0;
  if(ok){const f=fs.createWriteStream(path.join(OUT,asset+'_M1_normalized.csv'));f.write('timestamp,open,high,low,close,bid,ask,spread,volume,source,asset\n');for(const r of rows)f.write(r.join(',')+'\n');await new Promise(res=>f.end(res))}
  return {instrument_id:symbol,canonical_instrument:asset,provider_name:'DERIV_OFFICIAL_M1_CANDLES',rows:rows.length,first:first?first.toISOString():'',last:last?last.toISOString():'',full_window_coverage:coverage,status:ok?'PASS':'FAIL',file:ok?path.join(OUT,asset+'_M1_normalized.csv'):'',...stats,page_count:PAGE_COUNT,pace_ms:PACE_MS,request_semantics:'BACKWARD_COUNT_5000_CANDLE_PAGES__ONE_SYMBOL_AT_A_TIME__GLOBAL_PACING__RATE_LIMIT_BACKOFF__SUBSCRIBE_OMITTED'};
 }finally{try{client.ws.close()}catch{}}
}
(async()=>{
 const evidence=[];
 for(const [asset,symbol] of ITEMS){
  try{const r=await one(asset,symbol);evidence.push(r);console.log(JSON.stringify(r));}
  catch(e){const r={instrument_id:symbol,canonical_instrument:asset,provider_name:'DERIV_OFFICIAL_M1_CANDLES',rows:0,full_window_coverage:false,status:'FAIL',file:'',error:String(e)};evidence.push(r);console.error(JSON.stringify(r));}
  await sleep(8000);
 }
 const pass=evidence.filter(x=>x.status==='PASS').length;
 const out={state:'INDEPENDENT_DERIV_SYNTHETIC_RECOVERY_PAGED',start:START.toISOString(),end_exclusive:END.toISOString(),assets_scheduled:5,assets_pass:pass,evidence,execution_topology:'FIVE_EXACT_SYMBOLS_STRICTLY_SEQUENTIAL__PAGED_BACKWARD__PACED_AND_RATE_LIMIT_AWARE',data_reused_from_2026:false};
 fs.writeFileSync(path.join(OUT,'DERIV_RECOVERY_EVIDENCE.json'),JSON.stringify(out,null,2));console.log(JSON.stringify({assets_scheduled:5,assets_pass:pass},null,2));if(pass!==5)process.exitCode=2;
})().catch(e=>{console.error(e);process.exit(3)});
