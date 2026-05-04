import os
os.environ['PYTHONPATH'] = '/home/wilson/Forex'
import asyncio
from forexmind.config.settings import get_settings
import oandapyV20
import oandapyV20.endpoints.trades as trades
from forexmind.risk.manager import get_risk_manager

cfg = get_settings().oanda
api = oandapyV20.API(access_token=cfg.api_key, environment=cfg.environment)
req = trades.TradesList(accountID=cfg.account_id)
try:
    resp = api.request(req)
    open_trades = resp.get('trades', [])
    oanda_ids = {str(t['id']) for t in open_trades}
    print('OANDA open trades count:', len(oanda_ids))
    print('OANDA trade ids sample:', list(oanda_ids)[:10])

    rm = get_risk_manager()
    # run the async sync method
    asyncio.run(rm.sync_open_trades(oanda_ids))
    print('Risk manager open_trade_count after sync:', rm.open_trade_count)
except Exception as e:
    import traceback
    print('Exception:', type(e), e)
    traceback.print_exc()
