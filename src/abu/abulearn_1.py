from abupy import AbuFactorBuyXD, BuyCallMixin, AbuBenchmark, AbuCapital, ABuPickTimeExecute, EMarketSourceType
import abupy
import pandas as pd

# pandas.core.window.ewm


class AbuFactorBuyBreak(AbuFactorBuyXD, BuyCallMixin):
    # 买入因子需要继承AbuFactorBuyXD或者更复杂的策略继承AbuFactorBuyBase
    # 买入因子混入BuyCallMixin，即做为正向策略，股票相关的策略全部是正向策略，即买涨，后续章节示例期货，期权会使用BuyPutMixin
    # 买入因子需要实现fit_day，即每一个交易日如何执行交易策略，当符合买入条件后，使用buy_tomorrow或者buy_today生成订单
    def fit_day(self, today):
        if today.close == self.xd_kl.close.max():
            return self.buy_tomorrow()
        return None

abupy.env.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx
buf_factors = [{'xd': 60, 'class': AbuFactorBuyBreak}, {'xd': 42, 'class': AbuFactorBuyBreak}]
benchmark = AbuBenchmark()
captial = AbuCapital(1000000, benchmark)

orders_pd, actions_pd, _ = ABuPickTimeExecute.do_symbols_with_same_factors(['usTSLA'], benchmark, buf_factors, None, captial, show=True)

