import pandas as pd
import numpy as np
import os
import getstockid
import stockutils


stockids = getstockid.readstockids()
rootpath = getstockid.getrootpath()


# 检测方法，1，2020-08-13，天山生物，
#               1. 近期突破80日高点，也就是4个月，
#               2. 60日线比较平，有一个大体的U型，
#               3. 日最大涨幅都不是很大，基本没有涨停
#               4. 至少一个一个3日的中枢，中枢日内收盘价高于5日均线
#               5. 被突破之后有一个上影线，因为大盘跌造成的
def buymethod1(data, endDate):
    subdata = data[:endDate]
    fitMax = stockutils.breakupMax(subdata, 80)
    ustyle = stockutils.averagelineUstyle(subdata, 'ma60', 80)
    raiseCheck = stockutils.raiseanddowncheck(subdata, 6.0, 80)
    daycenter, _ = stockutils.centerCheck(subdata)
    topline = stockutils.daytopsingleline(subdata, chechrate=80)
    if fitMax and ustyle and raiseCheck and daycenter and topline:
        return True
    return False


# 检测方法，2，2020-09-08，任子行，
#               1. 突破60日最高点，
#               2. 60日均线有一个下落，平缓，再试图拉起的过程，
#               3. 之前有连续的2个日线中枢，3到5日都有可能，不超过5日。
#               4.近期单日涨幅不大，不超过6%，没有涨停。
#               5.当日指数大跌，有上影线，并且较长。
#               6, 当日最低不低于5日线均线,
#               7.在做中枢的区间里，最低值没有低于10日线，碰10日线立即拉升形成中枢,
#               8. 中枢形态是两边高，中间低，中间，中间的振幅小
def buymethod2(data, endDate):
    subdata = data[:endDate]
    fitMax = stockutils.breakupMax(subdata)
    ustyle = stockutils.averagelineUstyle(subdata, 'ma60')
    raiseCheck = stockutils.raiseanddowncheck(subdata, 6.0)
    topline = stockutils.daytopsingleline(subdata, chechrate=80)



# 检测方法，3，200821-200822，东方热电，
#               1. 近期突破60日高点。
#               2，回调过程中连续两日出现下影线，10日或者20日均线相交，
#               3。连续两日均是收盘高于开盘，涨幅不大，不超过3%，
#               4，今日最低点高于前一日最低点，
#               5.距离60日最低点涨幅不超过50%
#               6. 5日内出现过超过5%的大跌
def buymethod3():
    pass


# 急剧震荡型，4。200810-200817。露笑科技
#               1. 近期10日，内突破60日内最高点。
#               2. 60日均线呈现出一个U型，
#               3。近期出现过涨停，然后涨停之后急剧下跌，然后又拉起，
#               4，连续2日或3日下跌，跌幅一共至少超过6%
def buymethod4():
    pass


# 高位涨停，跌停交替型，300185，通裕重工，珈伟新能，300317，时间2020-09-14
#         1. 非创业板，涨幅10，创业板20，10个交易日能出现涨停，
#         2. 2日内出现大幅下跌，要么跌停，要么跌幅每日跌幅都超过5%或者10%


# 大盘下跌日，高位出现上影线，天能重工300569，2020-09-10
#       1. 突破高位  2. 最近没有涨停  3. 连续两日出现上影线，这两日的最高点超过之前60日最高点





stock_list1 = [] # 突破到至少60个交易日的最高点
stock_list2 = [] # macd日线背驰
stock_list3 = [] # 连续3日持续下跌
for id in stockids:
    id = '300313.SZ'
    idfile = os.path.join(rootpath, id + '.csv')
    iddata = pd.read_csv(idfile, index_col='date', parse_dates=['date'])
    macd,  _, _ = stockutils.calculateMACD(iddata['close'])
    iddata['macd'] = macd
    iddata['ma5'] = iddata['close'].rolling(5).mean()
    iddata['ma10'] = iddata['close'].rolling(5).mean()
    iddata['ma20'] = iddata['close'].rolling(5).mean()
    iddata['ma30'] = iddata['close'].rolling(5).mean()
    iddata['ma60'] = iddata['close'].rolling(5).mean()
    iddata['ma250'] = iddata['close'].rolling(5).mean()
    subdata = iddata[:'2020-08-18']
    fitMax = stockutils.breakupMax(subdata)
    ustyle = stockutils.averagelineUstyle(subdata, 'ma60', 80)
    raiseCheck = stockutils.raiseanddowncheck(subdata, 6.0, 80)
    topline = stockutils.daytopsingleline(subdata)
    midcenter = stockutils.centerCheck(subdata)
    print(macd)
