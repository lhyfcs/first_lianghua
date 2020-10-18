from abupy import AbuFactorBuyXD, BuyCallMixin, AbuBenchmark, \
    AbuCapital, ABuPickTimeExecute, EMarketSourceType, \
    AbuFactorSellXD, ESupportDirection, AbuFactorAtrNStop, AbuFactorPreAtrNStop, \
    AbuFactorCloseAtrNStop
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

class AbuFactorSellBreak(AbuFactorSellXD):
    # 1. 卖出因子需要继承AbuFactorSellXD或者更复杂的策略继承AbuFactorSellBase
    # 2. 卖出因子需要实现support_direction方法，确定策略支持的买入策略方向，本例中[ESupportDirection.DIRECTION_CAll.value]即只支持正向买入策略，即买涨
    # 3. 卖出因子需要实现fit_day，看有没有符合卖出条件的交易单子

    def support_direction(self):
        return [ESupportDirection.DIRECTION_CAll.value]

    def fit_day(self, today, orders):
        if today.close == self.xd_kl.close.min:
            for order in orders:
                self.sell_tomorrow(order)
# AbuFactorBuyBreak 突破买入策略
# AbuFactorSellBreak 突破卖出策略
# AbuFactorAtrNStop 止盈止损策略
# AbuFactorPreAtrNStop 暴跌止损因子
# AbuFactorCloseAtrNStop 移动止盈策略
abupy.env.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx
buf_factors = [{'xd': 60, 'class': AbuFactorBuyBreak}, {'xd': 42, 'class': AbuFactorBuyBreak}]
sellfactors = [{'xd': 120, 'class': AbuFactorSellBreak},
               {'stop_loss_n': 0.5, 'stop_win_n': 3, 'class': AbuFactorAtrNStop},
               {'class': AbuFactorPreAtrNStop, 'pre_atr_n': 1.0},
               {'class': AbuFactorCloseAtrNStop, 'close_atr_n': 1.5}]
benchmark = AbuBenchmark()
captial = AbuCapital(1000000, benchmark)

orders_pd, actions_pd, _ = ABuPickTimeExecute.do_symbols_with_same_factors(['usTSLA'], benchmark, buf_factors, sell_factors=sellfactors, capital=captial, show=True)

