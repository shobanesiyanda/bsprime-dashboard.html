#!/usr/bin/env node
const {getHistoricalRates}=require('dukascopy-node');
const fs=require('fs');
const START=new Date('2024-12-23T00:00:00Z');
const END=new Date('2025-07-21T00:00:00Z');
const ITEMS=[
  {instrument_id:'coffeecmdusx',canonical_instrument:'COFFEE.CMD',provider_name:'COFFEE.CMD/USX'},
  {instrument_id:'msftususd',canonical_instrument:'MSFT.US',provider_name:'MSFT.US/USD'}
];
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function valid(o,h,l,c){return[o,h,l,c].every(Number.isFinite)&&h>=Math.max(o,c)&&l<=Math.min(o,c)&&h>=l}
async function inspect(x){
 let lastErr='';
 for(let attempt=1;attempt<=4;attempt++){
  try{
   const d=await getHistoricalRates({instrument:x.instrument_id,dates:{from:START,to:END},timeframe:'m1',format:'json',priceType:'bid',batchSize:10,pauseBetweenBatchesMs:250});
   const seen=new Map();let invalid=0,duplicates=0,conflicts=0,outside=0;const samples=[];
   for(const b of d||[]){
    const t=+b.timestamp;if(!Number.isFinite(t)||t<START.getTime()||t>=END.getTime()){outside++;continue}
    const o=+b.open,h=+b.high,l=+b.low,c=+b.close;
    if(!valid(o,h,l,c)){invalid++;if(samples.length<8)samples.push({type:'invalid',timestamp:new Date(t).toISOString(),o,h,l,c});continue}
    const sig=[o,h,l,c,+b.volume||0].join('|');
    if(seen.has(t)){
      duplicates++;
      if(seen.get(t)!==sig){conflicts++;if(samples.length<8)samples.push({type:'duplicate_conflict',timestamp:new Date(t).toISOString(),first:seen.get(t),second:sig})}
      continue;
    }
    seen.set(t,sig);
   }
   const times=[...seen.keys()].sort((a,b)=>a-b);const first=times.length?new Date(times[0]):null,last=times.length?new Date(times.at(-1)):null;
   const gaps=[];for(let i=1;i<times.length;i++){const dt=(times[i]-times[i-1])/60000;if(dt>60&&gaps.length<20)gaps.push({from:new Date(times[i-1]).toISOString(),to:new Date(times[i]).toISOString(),minutes:dt})}
   return {...x,attempt,total_provider_rows:(d||[]).length,unique_valid_rows:times.length,invalid_ohlc:invalid,duplicate_timestamps:duplicates,conflicting_duplicates:conflicts,outside_window:outside,first:first?first.toISOString():null,last:last?last.toISOString():null,start_delay_hours:first?(first-START)/3600000:null,end_shortfall_hours:last?(END-last)/3600000:null,large_gaps_over_60m:gaps,samples};
  }catch(e){lastErr=String(e);await sleep(attempt*1000)}
 }
 return {...x,error:lastErr};
}
(async()=>{const out=[];for(const x of ITEMS)out.push(await inspect(x));fs.writeFileSync('MISSING2_DIAGNOSTIC.json',JSON.stringify(out,null,2));console.log(JSON.stringify(out,null,2));})();
