import pandas as pd
import numpy as np
import os
import getstockid
import stockutils
import updatestocksdata
import baostock as bs
import datetime
from drawkline import InterCandle
import matplotlib.pyplot as plt
import pickle5

# install conda & pytorch in mac
# https://zwarrior.medium.com/configure-pytorch-for-pycharm-using-conda-in-macos-catalina-5bc6f2353c90
rootpath = getstockid.getrootpath()

# 检测方法，1，2020-08-13，天山生物，
#               1. 近期突破80日高点，也就是4个月，
#               2. 60日线比较平，有一个大体的U型，
#               3. 日最大涨幅都不是很大，基本没有涨停
#               4. 至少一个一个3日的中枢，中枢日内收盘价高于5日均线
#               5. 被突破之后有一个上影线，因为大盘跌造成的
def buymethod1(data, endDate, cdoe):
    subdata = data[:endDate]
    fitMax = stockutils.breakupMax(subdata, 80)
    ustyle = stockutils.averagelineUstyle(subdata, 'ma60', 80)
    raiseCheck = stockutils.raiseanddowncheck(subdata, 6.0, 80)
    daycenter, _ = stockutils.centerCheck(subdata[:-1])
    topline = stockutils.daytopsingleline(subdata, chechrate=4)
    if fitMax and ustyle and raiseCheck and daycenter and topline:
        return True
    return False


# 检测方法，2，2020-09-07，任子行，300311.SZ
#               1. 突破60日最高点，
#               2. 60日均线有一个下落，平缓，再试图拉起的过程，
#               3. 之前有连续的2个日线中枢，3到5日都有可Î能，不超过5日。
#               4.近期单日涨幅不大，不超过6%，没有涨停。
#               5.当日大盘指数大跌，有上影线，并且较长。
#               6, 当日最低不低于5日线均线,
#               7.在做中枢的区间里，最低值没有低于10日线，碰10日线立即拉升形成中枢,
#               8. 中枢形态是两边高，中间低，中间的振幅小
def buymethod2(data, endDate, cdoe):
    subdata = data[:endDate]
    check1 = stockutils.breakupMax(subdata)
    check2 = stockutils.averagelineUstyle(subdata, 'ma60')
    check3, days = stockutils.centerCheck(subdata[:-1], 'ma10')
    if not check3:
        return False
    check3, _ = stockutils.centerCheck(subdata[:-1-days], 'ma10')
    if not check3:
        return False
    check4 = stockutils.raiseanddowncheck(subdata, 6.0)

    check5 = stockutils.daytopsingleline(subdata, chechrate=3)
    check6 = subdata['low'][-1] > subdata['ma5'][-1]
    check7 = stockutils.midcentercheck(subdata[:-1])
    return check1 and check2 and check3 and check4 and check5 and check6 and check7



# 检测方法，3，200821-200824，东方热电，id = '300217.SZ'
#               1. 近期突破60日高点，至少站上20日均线, 距离60日最低点涨幅不超过50%
#               2，回调过程中连续两日出现下影线，10日或者20日均线相交，
#               3。连续两日均是收盘高于开盘，涨幅不大，不超过3%，
#               4，今日最低点高于前一日最低点，
#               5. 5日内出现过超过5%的大跌
#               6. 均线呈现 5 > 10 > 20 > 60
def buymethod3(data, enddate, cdoe):
    subdata = data[:enddate]
    check1 = stockutils.breakupmaxinperday(subdata) and subdata['low'][-1] > subdata['ma20'][-1]
    check2 = stockutils.continuefitfunction(subdata, stockutils.bottomsingleline(2.5))
    check3 = (3 > subdata['pctChg'][-1] > 0) and (3 > subdata['pctChg'][-2] > 0)
    check4 = subdata['low'][-1] > subdata['low'][-2]
    check5 = subdata.iloc[-5:, -12].min() < -5  # 加上了均值线
    check6 = stockutils.continuefitfunction(subdata, stockutils.averagelineorder, continuedata=5)
    return check1 and check2 and check3 and check4 and check5 and check6


# 急剧震荡型，4。200810-200819。露笑科技, 002617.SZ
#               1. 近期10日，内突破60日内最高点。
#               2. 60日均线呈现出一个U型，
#               3。近期出现过涨停，然后涨停之后急剧下跌，然后又拉起，
#               4，连续2日或3日下跌，跌幅一共至少超过6%,
#               5. 30个交易日内最低点到最高点涨幅不超过50%
#               6. 每日高点到低点的降幅不超过15， 30
def buymethod4(data, endDate, code):
    subdata = data[:endDate]
    check1 = stockutils.breakupinrange(subdata)
    check2 = stockutils.averagelineUstyle(subdata, 'ma60')
    check3 = stockutils.zhengdangtype(subdata)
    check4 = stockutils.daydownfind(subdata, checkrate=8.0)
    check5 = stockutils.buttomtotopcheck(subdata)
    check6 = stockutils.toptobottomcheck(subdata, code)
    return check1 and check2 and check3 and check4 and check5 and check6


# 高位涨停，跌停交替型，珈伟新能，300317，时间2020-09-10
#         1. 非创业板，涨幅10，创业板20，10个交易日能出现涨停，
#         2. 2日内出现大幅下跌，要么跌停，要么跌幅每日跌幅都超过5%或者10%
def buymethod5(data, endData, code):
    chuangyeban = '300' in code
    # check1 = 0
    # check2 = 0
    subdata = data[:endData]
    if chuangyeban:
        check1 = stockutils.largestepdaycount(data, checkrate=19.5)
    else:
        check1 = stockutils.largestepdaycount(data, checkrate=9.5)

    if chuangyeban:
        check2 = stockutils.largestepdaycount(data, checkrate=-10.0, large=False)
    else:
        check2 = stockutils.largestepdaycount(data, checkrate=-5, large=False)
    check3 = stockutils.buttomtotopcheck(subdata)
    return check1 >= 1 and check2 >= 2 and check3


# 大盘下跌日，高位出现上影线，天能重工300569.SZ，2020-09-10
#       1. 突破高位
#       2. 最近没有涨停
#       3. 连续两日出现上影线，这两日的最高点超过之前60日最高点
#       4. 之前有两个波动，第一个波动之前已经破新高，这个算法比较复杂
def bugmethod6(data, endDate, code):
    subdata = data[:endDate]
    check1 = stockutils.breakupMax(subdata) or stockutils.breakupMax(subdata[:-1])
    chuangyeban = '300' in code
    raiserate = 2.5
    if chuangyeban:
        raiserate = 5
        check2 = stockutils.raiseanddowncheck(subdata, 15.0, 60, 0)
    else:
        check2 = stockutils.raiseanddowncheck(subdata, 8.0, 60)
    check3 = stockutils.daytopsingleline(data, chechrate=4) and stockutils.daytopsingleline(data[: -1], chechrate=4) and subdata['pctChg'][-1] < raiserate and subdata['pctChg'][-2] < raiserate
    check4 = stockutils.mergeklineandcheckcenter(subdata[-12:-2])
    return check1 and check2 and check3 and check4


# 连续5日下跌，跌幅均不超过1%，每日均是绿盘， 000728.SZ 国元证券,2020-09-17
#       1. 连续5日下跌，
#       2. 连续5日绿色
#       3. 每日跌幅均不超过1%
#       4. 今日最低是30交易日内最低点
#       5. macd往前推2红2绿，红面积极其小于绿面积
#
def buymethod7(data, endData, code):
    subdata = data[:endData]
    fivedata = subdata[-5:]
    check1 = len(fivedata[fivedata['pctChg'] < 0]) == 5
    check2 = len(fivedata[fivedata['open'] >= fivedata['close']]) == 5
    check3 = len(fivedata[fivedata['pctChg'] > -1]) == 5
    sub1 = subdata[-30:]
    check4 = sub1['low'][-1] == sub1['low'].min()
    check5 = stockutils.macdstatuscheck(subdata)
    return check1 and check2 and check3 and check4 and check5

# 连续5日上涨之后接连两天下跌。曙光股份，600303.SH，2020-09-24
#           1. 之前连续5天上涨，每天上涨幅度均不超过4%，
#           2. 连续2天下跌
#           3. 日线价格高于60日均线

def buymethod8(data, endData, code):
    subdata = data[:endData]
    def raisecheck(d):
        return 0 < d['pctChg'][-1] < 4
    check1 = stockutils.continuefitfunction(subdata[:-2], raisecheck, 5)
    def reduceCheck(d):
        return -5 < d['pctChg'][-1] < 0
    check2 = stockutils.continuefitfunction(subdata, reduceCheck, 2)
    check3 = subdata['close'][-1] > subdata['ma60'][-1]
    return check1 and check2 and check3


# 涨停之后，连续下跌，拓日新能 --002218，2020-09-29，亿晶光电--600537
#           1. 涨停，  
#           2.连续3日或者4日绿柱子，也就是close < open  
#           3. 超过60日线

def buymethod9(data, endDate, code) -> bool:
    subdata = data[:endDate]
    def greencheck(d):
        return d['close'][-1] < d['open'][-1]
    continueday = 4
    check2 = stockutils.continuefitfunction(subdata, greencheck, 4)
    if not check2:
        check2 = stockutils.continuefitfunction(subdata, greencheck, 3)
        continueday = 3
    check1 = subdata['pctChg'][-(continueday + 1)] > 9.9
    check3 = subdata['close'][-1] > subdata['ma60'][-1]
    return check1 and check2 and check3


# 恒星科技, 002132，2020-09-25
            # 1. 一个至少超过3%的下跌，
            # 2. 之前连续两日有上影线，
            # 3. 都是上涨
            # 4. 60日线
def buymethod10(data, endDate, code):
    subdata = data[:endDate]
    check1 = subdata['pctChg'][-1] < -3
    predata = subdata[:-1]

    def toplinefun(rate):
        def funinner(d):
            return stockutils.daytopsingleline(d, rate, False)
        return funinner
    check2 = stockutils.continuefitfunction(predata, toplinefun(2), 2)
    def raisecheck(d):
        return d['pctChg'][-1] > 0
    check3 = stockutils.continuefitfunction(predata, raisecheck)
    check4 = subdata['close'][-1] > subdata['ma60'][-1]
    return check1 and check2 and check3 and check4



# 下跌背驰规则 , 中鼎股份 000887，深南股份 002417
#           1. macd 线处于绿色，且背驰
#           2. 低于5日局限，10日均线， 60日均线
#           3. 近期下跌幅度大，至少30%
#           4. 当日是近期30日最低点
#           5. 连续3天下跌
def buymethod11(data, endDate, code):
    subdata = data[:endDate]
    check1 = stockutils.macdstatuscheck(subdata, False)
    close = subdata['close'][-1]
    check2 = close < subdata['ma5'][-1] and close < subdata['ma10'][-1] and close < subdata['ma60'][-1]
    thty = subdata[-30: ]
    check3 = (thty['high'].max() - thty['low'].min()) / thty['low'].min() > 0.28
    check4 = subdata['low'][-1] == thty['low'].min()

    reduceCheck = lambda d: d['pctChg'][-1] <= 0
    check5 = stockutils.continuefitfunction(subdata, reduceCheck, 3)
    return check1 and check2 and check3 and check4 and check5 and subdata['pctChg'][-1] < -1.5

def searchIncreaseFast(data, endDate, code):
    startPeriod = 300
    subdata = data[-startPeriod:]
    searchPeriod = 25
    for i in range(0, startPeriod - searchPeriod):
        period = subdata[i: i + searchPeriod]
        low = period['low'].min()
        high = period['high'].max()
        if (high - low) / low > 1:
            return True
    return False
# getids

def feature_normalize(data):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mu)/std

if __name__ == '__main__':
    bs.login()
    idCollect = getstockid.GetStockId()
    idCollect.downloadStockIdByDate()

    # download data
    dataUpdater = updatestocksdata.UpdateSocketData()
    dataUpdater.update()
    bs.logout()
    stockids = getstockid.readstockids()
    funarr = [buymethod1, buymethod2, buymethod3, buymethod4, buymethod5, bugmethod6, buymethod7, buymethod8, buymethod9, buymethod10, buymethod11]
    # funarr = [searchIncreaseFast]
    result = []
    for i in range(len(funarr)):
        result.append([])

    curtime = datetime.datetime.now()
    findStock = []
    for index, id in enumerate(stockids):
#        id = '603518.SH'
        idfile = os.path.join(rootpath, id + '.csv')
        iddata = pd.read_csv(idfile, index_col='date', parse_dates=['date'])
        macd,  _, _ = stockutils.calculateMACD(iddata['close'])
        iddata['macd'] = macd
        iddata['ma5'] = iddata['close'].rolling(5).mean()
        iddata['ma10'] = iddata['close'].rolling(10).mean()
        iddata['ma20'] = iddata['close'].rolling(20).mean()
        iddata['ma30'] = iddata['close'].rolling(30).mean()
        iddata['ma60'] = iddata['close'].rolling(60).mean()
        iddata['ma250'] = iddata['close'].rolling(250).mean()
        # method8 = buymethod8(iddata, '2020-09-24', '600303')
        # method9 = buymethod9(iddata, '2020-09-29', '002218')
        # method10 = buymethod10(iddata, '2020-09-25', '002132')
        # 市盈率检测
        if not stockutils.pecheck(iddata):
            continue
        # subdata = iddata[:'2020-08-18']
        # method1 = buymethod1(iddata, '2020-08-18')
        # method2 = buymethod2(iddata, '2020-09-07')
        # method3 = buymethod3(iddata, '2020-08-24')
        # method4 = buymethod4(iddata, '2020-08-17')
        # method5 = buymethod5(iddata, '2020-09-14', '300317')
        # method6 = bugmethod6(iddata, '2020-09-10', '300569')
        ids = id.split('.')
        # print(ids[0])
        # normal = feature_normalize(iddata['preclose'])
        # topCount = stockutils.findBreakCountInLatestTime(iddata, 60, 200)
        # btmCount = stockutils.findBreakCountInLatestTime(iddata, 60, 200, False)
        for i in range(len(funarr)):
            if funarr[i](iddata, '2021-04-30', ids[0]):
                result[i].append(ids[0])
                findStock.append(id)
        if index % 50 == 0:
            tmpCur = datetime.datetime.now()
            print('time stamp', (tmpCur - curtime).seconds)
            curtime = tmpCur
            print('solve count', index)

    for i in range(len(result)):
        print('#########################  method ', i, "############################")
        print('find count: ', len(result[i]))
        print(result[i])
    findStock = list(set(findStock))
    print(findStock)
    with open(os.path.join(rootpath, 'findstock.txt'), "wb") as fp:  # Pickling
        pickle5.dump(findStock, fp)
    # with open("test.txt", "rb") as fp:  # Unpickling
    #     findStock = pickle5.load(fp)
    candle = InterCandle(findStock)
    candle.refresh_texts(candle.df.iloc[candle.idx_start + candle.idx_range])
    candle.refresh_plot(candle.idx_start, candle.idx_range)
    plt.show()
    print('complete')
