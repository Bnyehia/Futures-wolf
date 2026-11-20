"""Scheduled public-market snapshot scanner. No API keys or trading orders."""
import asyncio, json, math, os, time
from datetime import datetime, timezone
from pathlib import Path
import httpx
import numpy as np
import pandas as pd

BASE = 'https://fapi.binance.com'
OUTPUT = Path(__file__).resolve().parents[1] / 'site' / 'data' / 'latest.json'
MIN_VOLUME = float(os.getenv('MIN_QUOTE_VOLUME', '10000000'))
CONCURRENCY = int(os.getenv('CONCURRENCY', '4'))
LIMIT = int(os.getenv('RESULT_LIMIT', '20'))

async def fetch(client, path, params=None):
    for attempt in range(4):
        try:
            r = await client.get(BASE + path, params=params)
            if r.status_code in (418, 429, 500, 502, 503, 504):
                if attempt < 3:
                    await asyncio.sleep(min(2 ** attempt * 2, 12))
                    continue
            r.raise_for_status()
            return r.json()
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt == 3: raise
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError('Binance request failed')

def metrics(rows, now_ms):
    # Keep only confirmed closed candles, including when a delayed API response contains a still-open candle.
    closed = [r for r in rows if int(r[6]) < now_ms]
    if len(closed) < 100: return None
    df = pd.DataFrame(closed)
    close, high, low, vol = (df[i].astype(float) for i in (4,2,3,5))
    ema20 = close.ewm(span=20,adjust=False).mean()
    ema50 = close.ewm(span=50,adjust=False).mean()
    delta=close.diff()
    gains=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    losses=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rsi=100-100/(1+gains/(losses.replace(0,np.nan)))
    true_range=pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    atr=true_range.ewm(alpha=1/14,adjust=False).mean()
    macd=close.ewm(span=12,adjust=False).mean()-close.ewm(span=26,adjust=False).mean()
    hist=macd-macd.ewm(span=9,adjust=False).mean()
    return dict(price=float(close.iloc[-1]),ema20=float(ema20.iloc[-1]),ema50=float(ema50.iloc[-1]),
                rsi=float(rsi.iloc[-1]),atr=float(atr.iloc[-1]),hist=float(hist.iloc[-1]),
                volume_ratio=float(vol.iloc[-1]/max(vol.iloc[-21:-1].mean(),1e-12)),
                close_time=int(closed[-1][6]))

async def analyze(client, sem, symbol, now_ms):
    async with sem:
        try:
            candles=await asyncio.gather(*(fetch(client,'/fapi/v1/klines',{'symbol':symbol,'interval':tf,'limit':120}) for tf in ('1m','5m','15m')))
            a,b,c=(metrics(x,now_ms) for x in candles)
            if not all((a,b,c)): return None
            up=c['price']>c['ema20']>c['ema50'] and b['price']>b['ema20']
            down=c['price']<c['ema20']<c['ema50'] and b['price']<b['ema20']
            side='LONG' if up else 'SHORT' if down else None
            if side is None: return None
            momentum=(b['hist']>0 and 48<=b['rsi']<=72) if side=='LONG' else (b['hist']<0 and 28<=b['rsi']<=52)
            if not momentum: return None
            entry=a['price']; atr=b['atr']
            if not math.isfinite(atr) or atr<=0 or entry<=0: return None
            stop=entry-1.5*atr if side=='LONG' else entry+1.5*atr
            tp=entry+3*atr if side=='LONG' else entry-3*atr
            if stop<=0 or tp<=0: return None
            score=25+20+(20 if b['volume_ratio']>=1.2 else 8)+(15 if ((a['hist']>0)==(side=='LONG')) else 0)+(10 if .0008<=atr/entry<=.03 else 0)+10
            if score<75: return None
            return {'symbol':symbol,'side':side,'score':score,'entry':entry,'stop':stop,'takeProfit':tp,
                    'rrGross':2.0,'rsi':round(b['rsi'],1),'volumeRatio':round(b['volume_ratio'],2),
                    'candleClose':a['close_time'],'status':'candidate'}
        except (httpx.HTTPError,ValueError,TypeError,KeyError,IndexError,ZeroDivisionError):
            return None

async def main():
    now_ms=int(time.time()*1000)
    async with httpx.AsyncClient(timeout=18,limits=httpx.Limits(max_connections=10),headers={'User-Agent':'FuturesAIScanner-Research/1.0'}) as client:
        info, tickers = await asyncio.gather(fetch(client,'/fapi/v1/exchangeInfo'),fetch(client,'/fapi/v1/ticker/24hr'))
        allowed={s['symbol'] for s in info['symbols'] if s.get('status')=='TRADING' and s.get('contractType')=='PERPETUAL' and s.get('quoteAsset')=='USDT'}
        eligible=sorted((t for t in tickers if t['symbol'] in allowed and float(t.get('quoteVolume',0))>=MIN_VOLUME),key=lambda x:float(x['quoteVolume']),reverse=True)
        sem=asyncio.Semaphore(CONCURRENCY)
        # Scan every liquid eligible contract, not just the top N by volume.
        outcomes=await asyncio.gather(*(analyze(client,sem,t['symbol'],now_ms) for t in eligible))
        opportunities=sorted((x for x in outcomes if x),key=lambda x:(x['score'],x['volumeRatio']),reverse=True)
        payload={'generatedAt':datetime.now(timezone.utc).isoformat(),'scanned':len(eligible),
                 'matched':len(opportunities),'opportunities':opportunities[:LIMIT],
                 'market':'Binance USDT-M Perpetual','strategy':'Scalping 1m/5m/15m',
                 'note':'Indicative setups only. Not live, no execution, no fees/spread/slippage/funding modeled. Not backtested.'}
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        tmp=OUTPUT.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        tmp.replace(OUTPUT)
        print(f"Scanned {len(eligible)} contracts; matched {len(opportunities)}; saved {OUTPUT}")

if __name__=='__main__':
    asyncio.run(main())
