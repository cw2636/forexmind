import os
os.environ['PYTHONPATH'] = '/home/wilson/Forex'
from forexmind.config.settings import get_settings
import oandapyV20
import oandapyV20.endpoints.instruments as _instruments
import pandas as pd
from forexmind.indicators.engine import IndicatorEngine

TRAIN_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "XAU_USD"]
TRAIN_BARS = 10000
MAX_OANDA_COUNT = 5000
TRAIN_TF = 'H1'

cfg = get_settings().oanda
api = oandapyV20.API(access_token=cfg.api_key, environment=cfg.environment)
engine = IndicatorEngine()
frames = []

for pair in TRAIN_PAIRS:
    try:
        print(f'Fetching for {pair}...')
        total_rows = []
        to_time = None
        while len(total_rows) < TRAIN_BARS:
            need = min(MAX_OANDA_COUNT, TRAIN_BARS - len(total_rows))
            params = {"granularity": TRAIN_TF, "price": "M", "count": str(need)}
            if to_time is not None:
                params['to'] = to_time
            req = _instruments.InstrumentsCandles(pair, params=params)
            data = api.request(req)
            batch = []
            for candle in data.get('candles', []):
                if not candle.get('complete', True):
                    continue
                mid = candle['mid']
                batch.append({
                    'time': pd.Timestamp(candle['time']).tz_convert('UTC'),
                    'open': float(mid['o']),
                    'high': float(mid['h']),
                    'low': float(mid['l']),
                    'close': float(mid['c']),
                    'volume': int(candle.get('volume',0)),
                })
            if not batch:
                break
            total_rows.extend(batch)
            earliest = pd.Timestamp(batch[0]['time'])
            to_time = (earliest - pd.Timedelta(microseconds=1)).isoformat()
            print(f'  fetched batch {len(batch)} -> total {len(total_rows)}')
            if len(batch) < need:
                break
        if not total_rows:
            print(f'  no data for {pair}, skipping')
            continue
        raw = pd.DataFrame(total_rows).set_index('time').sort_index().iloc[-TRAIN_BARS:]
        ind_df = engine.compute(raw)
        ind_df['instrument'] = pair
        frames.append(ind_df)
        print(f'  done: {len(ind_df)} rows for {pair}')
    except Exception as e:
        print('  EXCEPTION for', pair, e)

if not frames:
    print('No frames fetched, aborting')
    raise SystemExit(1)

train_df = pd.concat(frames)
print('Total training rows:', len(train_df))

# LightGBM training
try:
    from forexmind.strategy.ml_strategy import LightGBMStrategy
    from forexmind.strategy.feature_engineering import build_feature_matrix
    lgbm = LightGBMStrategy()
    feat_df = build_feature_matrix(train_df, add_target=True)
    print('Feature matrix rows:', len(feat_df))
    lgbm_result = lgbm.train(feat_df)
    print('LightGBM result:', lgbm_result)
except Exception as e:
    print('LightGBM training failed:', e)

# LSTM training (may be slow)
try:
    from forexmind.strategy.ml_strategy import LSTMStrategy
    lstm = LSTMStrategy()
    lstm_result = lstm.train(train_df)
    print('LSTM result:', lstm_result)
except Exception as e:
    print('LSTM training failed:', e)
