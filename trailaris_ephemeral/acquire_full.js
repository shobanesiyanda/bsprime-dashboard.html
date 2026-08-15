const fs=require('fs');
const path=require('path');
const {getHistoricalRates}=require('dukascopy-node');
const WebSocket=require('ws');

const START=new Date('2022-01-01T00:00:00Z');
const END=new Date('2026-08-13T00:00:00Z'); // exclusive; through 2026-08-12
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
async function getDuk(asset,id){
  console.log('DUK',asset,id);
  try{
    const d=await getHistoricalRates({instrument:id,dates:{from:START,to:END},timeframe:'m5',format:'json'});
    if(!Array.isArray(d)||!d.length) throw new Error('empty');
    const f=fs.createWriteStream(path.join(RAW,asset+'.csv')); f.write('timestamp,open,high,low,close,volume\n');
    let first=null,last=null,n=0;
    for(const r of d){ if(!Number.isFinite(+r.close))continue; const ts=+r.timestamp; if(first===null)first=ts; last=ts; n++; f.write(csvLine(r)); }
    await new Promise(res=>f.end(res));
    coverage[asset]={source:'DUKASCOPY_RESEARCH_PROXY',rows:n,first,last,status:n>100?'OK':'SHORT'};
  }catch(e){coverage[asset]={source:'DUKASCOPY_RESEARCH_PROXY',rows:0,status:'FAIL',error:String(e)}; console.error(asset,e);}
}
function sleep(ms){return new Promise(r=>setTimeout(r,ms));}
async function getDeriv(asset,symbol){
  console.log('DERIV',asset,symbol);
  const file=fs.createWriteStream(path.join(RAW,asset+'.csv')); file.write('timestamp,open,high,low,close,volume\n');
  let ws,seq=1,pending=new Map(),rows=0,first=null,last=null,seen=new Set();
  try{
    ws=new WebSocket('wss://ws.binaryws.com/websockets/v3?app_id=1089');
    await new Promise((resolve,reject)=>{const t=setTimeout(()=>reject(new Error('open timeout')),20000);ws.once('open',()=>{clearTimeout(t);resolve();});ws.once('error',reject);});
    ws.on('message',(m)=>{let x;try{x=JSON.parse(m.toString())}catch{return};const p=pending.get(x.req_id);if(p){pending.delete(x.req_id); if(x.error)p.reject(new Error(x.error.message));else p.resolve(x);}});
    function req(obj){return new Promise((resolve,reject)=>{const id=seq++;pending.set(id,{resolve,reject});ws.send(JSON.stringify({...obj,req_id:id}));setTimeout(()=>{if(pending.has(id)){pending.delete(id);reject(new Error('request timeout'));}},30000);});}
    const step=10*86400; let cur=Math.floor(START.getTime()/1000), endAll=Math.floor(END.getTime()/1000)-1;
    while(cur<=endAll){
      const e=Math.min(endAll,cur+step-1);
      try{
        const x=await req({ticks_history:symbol,start:cur,end:e,style:'candles',granularity:300,count:5000,adjust_start_time:1,subscribe:0});
        const cs=x.candles||[];
        for(const c of cs){const ep=+c.epoch;if(ep<cur||ep>e||seen.has(ep))continue;seen.add(ep);const ts=ep*1000;if(first===null)first=ts;last=ts;rows++;file.write([ts,c.open,c.high,c.low,c.close,0].join(',')+'\n');}
      }catch(e2){console.error('DERIV_CHUNK',asset,cur,String(e2));}
      cur=e+1; if(seq%30===0)await sleep(200);
    }
    coverage[asset]={source:'DERIV_OFFICIAL',rows,first,last,status:rows>100?'OK':'SHORT'};
  }catch(e){coverage[asset]={source:'DERIV_OFFICIAL',rows,status:'FAIL',error:String(e)};console.error(asset,e);} finally {await new Promise(res=>file.end(res));try{ws&&ws.close()}catch{}}
}
async function pool(entries,fn,limit=3){let i=0;const workers=Array.from({length:limit},async()=>{while(true){const j=i++;if(j>=entries.length)return;await fn(...entries[j]);}});await Promise.all(workers);}
(async()=>{
  await pool(Object.entries(duk),getDuk,3);
  await pool(Object.entries(deriv),getDeriv,2);
  fs.writeFileSync('trailaris_ephemeral/coverage_partial.json',JSON.stringify(coverage,null,2));
  console.log('ACQUIRE_FULL_JS_DONE',Object.keys(coverage).length);
})();
