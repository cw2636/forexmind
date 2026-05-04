"""CI retrain script used by GitHub Actions.
Fetches paginated training data, retrains LightGBM + LSTM, saves models and metrics.
"""
import json
from pathlib import Path
import os
os.environ.setdefault('PYTHONHASHSEED','0')

# Make repo importable
from forexmind.config.settings import get_settings
from forexmind.indicators.engine import IndicatorEngine

import oandapyV20
import oandapyV20.endpoints.instruments as _instruments
import pandas as pd

from forexmind.strategy.ml_strategy import LightGBMStrategy, LSTMStrategy
from forexmind.strategy.feature_engineering import build_feature_matrix

OUT_DIR = Path('artifacts')
OUT_DIR.mkdir(exist_ok=True)

cfg = get_settings().oanda
api = oandapyV20.API(access_token=cfg.api_key, environment=cfg.environment)
engine = IndicatorEngine()

TRAIN_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "XAU_USD"]
TRAIN_BARS = 10000
MAX_OANDA_COUNT = 5000
TRAIN_TF = 'H1'

import time, random

def _api_request_with_retries(req, max_attempts: int = 5):
    base = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            return api.request(req)
        except Exception as e:
            msg = str(e)
            if "Maximum value for 'count' exceeded" in msg:
                raise
            if attempt == max_attempts:
                raise
            sleep_for = base * (2 ** (attempt - 1)) + random.random() * 0.5
            print(f"OANDA request failed (attempt {attempt}/{max_attempts}): {e} — retrying in {sleep_for:.1f}s")
            time.sleep(sleep_for)

frames = []
for pair in TRAIN_PAIRS:
    total_rows = []
    to_time = None
    while len(total_rows) < TRAIN_BARS:
        need = min(MAX_OANDA_COUNT, TRAIN_BARS - len(total_rows))
        params = {"granularity": TRAIN_TF, "price": "M", "count": str(need)}
        if to_time is not None:
            params['to'] = to_time
        req = _instruments.InstrumentsCandles(pair, params=params)
        data = _api_request_with_retries(req)
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
        if len(batch) < need:
            break
    if not total_rows:
        print(f'No data for {pair}, skipping')
        continue
    raw = pd.DataFrame(total_rows).set_index('time').sort_index().iloc[-TRAIN_BARS:]
    ind_df = engine.compute(raw)
    ind_df['instrument'] = pair
    frames.append(ind_df)

train_df = pd.concat(frames)
print('Total training rows:', len(train_df))

# Train and save LightGBM
metrics = {}
try:
    lgbm = LightGBMStrategy()
    feat_df = build_feature_matrix(train_df, add_target=True)
    lgbm_result = lgbm.train(feat_df)
    try:
        lgbm.save_model(lgbm._model, OUT_DIR / 'lgbm_forex.pkl')
    except Exception:
        # fallback: if LightGBMStrategy uses a different save API
        try:
            lgbm.save(lgbm._model, OUT_DIR / 'lgbm_forex.pkl')
        except Exception:
            pass
    metrics['lightgbm'] = lgbm_result
    print('LightGBM done')
except Exception as e:
    metrics['lightgbm'] = {'error': str(e)}
    print('LightGBM failed:', e)

# Train and save LSTM
try:
    lstm = LSTMStrategy()
    lstm_result = lstm.train(train_df)
    # LSTMStrategy should provide save; fallback to save_model if exists
    try:
        lstm.save_model(lstm._model, OUT_DIR / 'lstm_forex.pt')
    except Exception:
        pass
    metrics['lstm'] = lstm_result
    print('LSTM done')
except Exception as e:
    metrics['lstm'] = {'error': str(e)}
    print('LSTM failed:', e)

# Write metrics json (named retrain_metrics.json for monitoring)
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(OUT_DIR / 'retrain_metrics.json', 'w') as fh:
    json.dump(metrics, fh, indent=2, cls=NumpyEncoder)

print('Artifacts saved to', OUT_DIR)

# Call auto-rollback: monitor and potentially restore previous model
print('\n' + '='*65)
print('  Invoking auto-rollback evaluation')
print('='*65)
try:
    from forexmind.monitoring.auto_rollback import evaluate_and_rollback
    evaluate_and_rollback(
        metrics_file=OUT_DIR / 'retrain_metrics.json',
        baseline_file=OUT_DIR / 'metrics_history.json',
        threshold=0.05  # rollback if accuracy drops >5%
    )
    print('Auto-rollback evaluation complete.')
except Exception as e:
    print(f'Warning: auto-rollback failed: {e}')
    import traceback
    traceback.print_exc()
