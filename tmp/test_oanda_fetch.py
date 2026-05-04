from forexmind.config.settings import get_settings
import oandapyV20
import oandapyV20.endpoints.instruments as instruments

cfg = get_settings().oanda
print('OANDA configured:', cfg.is_configured)
print('Environment:', cfg.environment)
print('Account ID present:', bool(cfg.account_id))

api = oandapyV20.API(access_token=cfg.api_key, environment=cfg.environment)
req = instruments.InstrumentsCandles('EUR_USD', params={'granularity':'H1','price':'M','count':'100'})
try:
    resp = api.request(req)
    print('Response keys:', list(resp.keys()))
    candles = resp.get('candles', [])
    print('Candles returned:', len(candles))
    if candles:
        print('First candle time/complete:', candles[0].get('time'), candles[0].get('complete'))
except Exception as e:
    import traceback
    print('Exception:', type(e), e)
    traceback.print_exc()
