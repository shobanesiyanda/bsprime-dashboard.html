const fs=require('fs');
const path=require('path');
const {getHistoricalRates}=require('dukascopy-node');
const WebSocket=require('ws');

const START_MS=Date.parse('2022-01-01T00:00:00Z');
const END_MS=Date.parse('2026-08-13T00:00:00Z')-1;
const RAW='trailaris_ephemeral/raw'; fs.mkdirSync(RAW,{recursive:true});
const duk={
 EURUSD:'eurusd',GBPUSD:'gbpusd',USDJPY:'usdjpy',AUDUSD:'audusd',USDCAD:'usdcad',USDCHF:'usdchf',NZDUSD:'nzdusd',
 EURJPY:'eurjpy',GBPJPY:'gbpjpy',EURGBP:'eurgbp',AUDJPY:'audjpy',CADJPY:'cadjpy',GBPCHF:'gbpchf',
 XAUUSD:'xauusd',XAGUSD:'xagusd',US100:'usatechidxusd',US500:'usa500idxusd',US30:'usa30idxusd',GER40:'deuidxeur',UK100:'gbridxgbp',JP225:'jpnidxjpy',
 WTI:'lightcmdusd',BRENT:'brentcmdusd',NATGAS:'gascmdusd'
};
const deriv={V75:'R_75',V50:'R_50',V100:'R_100',V25:'R_25',V10:'R_10'};
const coverage={};
function csvLine(r){return [r.timestamp,r.open,r.high,r.low,r.close,r.volume??0].join(',')+'\n';}
function existing(asset){try{const p=path.join(RAW,asset+'.csv'); const s=fs.statSync(p); return s.size>5000;}catch{return false;}}
async function getDuk(asset,id){
  if(existing(asset)){coverage[asset]={source:'CACHE',status:'OK'};return;}
  console.log('DUK',asset,id);
  try{
    const d=await getHistoricalRates({instrument:id,dates:{from:new Date(START_MS),to:new Date(END_MS+1)},timeframe:'m5',format:'json'});
    if(!Array.isArray(d)||!d.length) throw new Error('empty');
    const f=fs.createWriteStream(path.join(RAW,asset+'.csv')); f.write('timestamp,open,high,low,close,volume\n');
    let first=null,last=null,n=0;
    for(const r of d){const ts=+r.timestamp;if(!Number.isFinite(+r.close)||ts<START_MS||ts>END_MS)continue;if(first===null)first=ts;last=ts;n++;f.write(csvLine(r));}
    await new Promise(res=>f.end(res));coverage[asset]={source:'DUKASCOPY_RESEARCH_PROXY',rows:n,first,last,status:n>100?'OK':'SHORT'};
  }catch(e){coverage[asset]={source:'DUKASCOPY_RESEARCH_PROXY',rows:0,status:'FAIL',error:String(e)};console.error(asset,e);}
}
function sleep(ms){return new Promise(r=>setTimeout(r,ms));}
async function getDeriv(asset,symbol){
  if(existing(asset)){coverage[asset]={source:'CACHE',status:'OK'};return;}
  console.log('DERIV',asset,symbol);
  const rows=new Map(); let ws,seq=1,pending=new Map(),pages=0,errors=[];
  try{
    ws=new WebSocket('wss://ws.binaryws.com/websockets/v3?app_id=1089');
    await new Promise((resolve,reject)=>{const t=setTimeout(()=>reject(new Error('open timeout')),20000);ws.once('open',()=>{clearTimeout(t);resolve();});ws.once('error',reject);});
    ws.on('message',(m)=>{let x;try{x=JSON.parse(m.toString())}catch{return};const p=pending.get(x.req_id);if(p){pending.delete(x.req_id);x.error?p.reject(new Error(x.error.message)):p.resolve(x);}});
    function req(obj){return new Promise((resolve,reject)=>{const id=seq++;pending.set(id,{resolve,reject});ws.send(JSON.stringify({...obj,req_id:id}));setTimeout(()=>{if(pending.has(id)){pending.delete(id);reject(new Error('request timeout'));}},30000);});}
    let end='latest',lastEarliest=null;
    while(true){
      let x; try{x=await req({ticks_history:symbol,end,style:'candles',granularity:300,count:5000,adjust_start_time:1});}catch(e){errors.push(String(e));if(errors.length>=3)throw e;await sleep(1000);continue;}
      const cs=x.candles||[]; if(!cs.length)break; const eps=cs.map(c=>+c.epoch).filter(Number.isFinite);if(!eps.length)break;
      const earliest=Math.min(...eps); pages++;
      for(const c of cs){const ms=(+c.epoch)*1000;if(ms>=START_MS&&ms<=END_MS)rows.set(ms,[ms,c.open,c.high,c.low,c.close,0]);}
      if(earliest*1000<=START_MS)break;if(lastEarliest!==null&&earliest>=lastEarliest)throw new Error('pagination stalled');lastEarliest=earliest;end=earliest-1;
      if(pages%12===0)await sleep(300);
      if(pages>180)throw new Error('pagination safety limit');
    }
    const ordered=[...rows.values()].sort((a,b)=>a[0]-b[0]);const f=fs.createWriteStream(path.join(RAW,asset+'.csv'));f.write('timestamp,open,high,low,close,volume\n');for(const r of ordered)f.write(r.join(',')+'\n');await new Promise(res=>f.end(res));
    coverage[asset]={source:'DERIV_OFFICIAL',rows:ordered.length,first:ordered.length?ordered[0][0]:null,last:ordered.length?ordered.at(-1)[0]:null,pages,status:ordered.length>100?'OK':'SHORT',errors};
  }catch(e){coverage[asset]={source:'DERIV_OFFICIAL',rows:0,status:'FAIL',error:String(e),pages,errors};console.error(asset,e);} finally {try{ws&&ws.close()}catch{}}
}
async function pool(entries,fn,limit){let i=0;const workers=Array.from({length:limit},async()=>{while(true){const j=i++;if(j>=entries.length)return;await fn(...entries[j]);}});await Promise.all(workers);}
(async()=>{await Promise.all([pool(Object.entries(duk),getDuk,6),pool(Object.entries(deriv),getDeriv,5)]);fs.writeFileSync('trailaris_ephemeral/coverage_partial.json',JSON.stringify(coverage,null,2));console.log('ACQUIRE_FULL_V2_DONE',Object.keys(coverage).length);})();
