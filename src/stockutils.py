import numpy as np
import pandas as pd
import datetime


def calculateEMA(period, closeArray, emaArray=[]):
    """计算指数移动平均"""
    length = len(closeArray)
    nanCounter = np.count_nonzero(np.isnan(closeArray))
    if not emaArray:
        emaArray.extend(np.tile([np.nan], (nanCounter + period - 1)))
        firstema = np.mean(closeArray[nanCounter:nanCounter + period - 1])
        emaArray.append(firstema)
        for i in range(nanCounter + period, length):
            ema = (2 * closeArray[i] + (period - 1) * emaArray[-1]) / (period + 1)
            emaArray.append(ema)
    return np.array(emaArray)


def calculateMACD(closeArray, shortPeriod=12, longPeriod=26, signalPeriod=9):
    ema12 = calculateEMA(shortPeriod, closeArray, [])
    ema26 = calculateEMA(longPeriod, closeArray, [])
    diff = ema12 - ema26
    dea = calculateEMA(signalPeriod, diff, [])
    macd = 2 * (diff - dea)
    return macd, diff, dea


def breakupMax(data, checkrange=60, bottomrise=1.5):
    closerange = data['close'][-checkrange:]
    minclose = closerange.min()
    if closerange[-1] == closerange.max() and (closerange[-1] - minclose) / minclose < bottomrise:
        return True
    return False


def breakupinrange(data, breakrange=10, checkrange=60):
    subbreak = data[-breakrange:]
    subdata = data[-checkrange:]
    return subbreak['high'].max() == subdata['high'].max()


def breakupmaxinperday(data, predayrange=5, checkrange=60):
    predata = data['high'][-predayrange:]
    datarange = data['high'][-checkrange:]
    return predata.max() == datarange.max()


def findvaluepos(list, value):
    l = len(list)
    for i in range(int(l / 2)):
        if value == list[i]:
            return i
        elif value == list[-i]:
            return l - i

def findmaxandmindis(list):
    bottom=[]
    top=[]
    findfirst = False
    findTop = True
    for i in range(2, len(list) - 2):
        if not findfirst:
            if list[i - 2] < list[i - 1] < list[i] > list[i + 1] > list[i + 2]:
                top.append(i)
                findfirst = True
                findTop = False
            if list[i - 2] < list[i - 1] > list[i] < list[i + 1] < list[i + 2]:
                bottom.append(i)
                findfirst = True
                findTop = True
        else:
            if findTop and list[i - 2] < list[i - 1] < list[i] > list[i + 1] > list[i + 2]:
                top.append(i)
                findTop = False
            elif not findTop and list[i - 2] < list[i - 1] > list[i] < list[i + 1] < list[i + 2]:
                bottom.append(i)
                findTop = True

    if len(top) <= 0:
        return 0, 0, False
    mergetop = [top[0]]
    for i in range(1, len(bottom) - 1):
        if list[top[i-1]] > list[top[i]] and list[bottom[i - 1]] > list[bottom[i]]:
            pass
        elif list[top[i - 1]] < list[top[i]] and list[bottom[i - 1]] < list[bottom[i]]:
            pass
        else:
            mergetop.append(top[i])
    if len(mergetop) <= 1:
        return 0, len(list) - 1, True
    maxpos = 0
    maxlen = -1
    for i in range(1, len(mergetop)):
        if maxlen < (mergetop[i] - mergetop[i - 1]):
            maxlen = mergetop[i] - mergetop[i - 1]
            maxpos = i

    return mergetop[maxpos - 1], mergetop[maxpos], True



def averagelineUstyle(data, linename, checkrange=60):
    linedata = data[linename][-checkrange:]
    # 检测算法，1. 两边高，中间低
    # 2. 低点在中间位置
    # 3. 高点在两边
    left = linedata[0]
    right = linedata[-1]
    mid = linedata[int(checkrange/2)]
    values = linedata.values
    maxleft, maxright, find = findmaxandmindis(values)
    if not find:
        return False
    return values[maxright] <= right and (maxright - maxleft) > (checkrange / 10 * 3) and linedata.max() / linedata.min() < 1.5

def raiseanddowncheck(data, maxrate ,checkrange=60, times=2):
    linedata = data['pctChg'][-checkrange:]
    return len(linedata[abs(linedata) > maxrate]) <= times;


def daytopsingleline(data, chechrate=2.5, checkmax=True, checkrange=60):
    high = data['high'][-checkrange:]
    if checkmax and high[-1] != high.max():
        return False
    last = data.iloc[-1]
    top = max(last.close, last.open)
    if (last.high - top) / top * 100 > chechrate:
        return True
    return False


def bottomsingleline(checkrate=2):
    def cal(data):
        last = data.iloc[-1]
        linerate = (last.open - last.low)/last.low * 100
        return linerate > checkrate
    return cal


def averagelineorder(data):
    last = data.iloc[-1]
    return last['ma5'] > last['ma10'] > last['ma20'] > last['ma60']


def centerCheck(data, baseline='ma5', checkbaseline = True):
    # center check, 5, 4, 3 day
    #       1. side is raise
    #       2. left low is the lowest
    for i in range(3, 5):
        try:
            subdata = data[-i:]
        except:
            return False, 0
        left = subdata.iloc[0]
        right = subdata.iloc[-1]
        rightPart = subdata[1:]
        if left.low != subdata['low'].min():
            continue
        if right.low != rightPart['low'].min():
            continue
        if left.open > left.close or right.open > right.close:
            continue
        if checkbaseline and len(subdata[subdata['close'] < subdata[baseline]]) > 0:
            continue
        return True, i
    return False, 0


def midcentercheck(data, mixrate = 1.5):
    subdata = data[-3:]
    left = subdata.iloc[0]
    right = subdata.iloc[2]
    mid = subdata.iloc[1]
    midrate = (mid.close - mid.open) / mid.close * 100.0
    return left.open < left.close and right.open < right.close and \
           mid.low > left.open and mid.low > right.open and \
           mid.high < left.close and mid.high < right.close and 0 > midrate > -mixrate * 100


def pecheck(data):
    pe = data['peTTM'][-1]
    return pe > 0 or pe < -30


def continuefitfunction(data, fun, continuedata=2):
    for i in range(continuedata):
        if i == 0:
            subdata = data
        else:
            subdata = data[:-i]
        if not fun(subdata):
            return False
    return True


def predate(cur, daymun):
    curtime = datetime.datetime.strptime(cur, '%Y-%m-%d')
    pretime = (curtime - datetime.timedelta(days=daymun)).strftime('%Y-%m-%d')
    return pretime


def zhengdangtype(data, checkrange=10):
    subdata = data[-checkrange:]
    for i in range(checkrange - 1):
        if abs(subdata['pctChg'][i] - 10) < 0.2 and subdata['pctChg'][i + 1] < -5.0:
            return True
    return False


def largestepdaycount(data, checkrange=10, checkrate=10, large=True):
    subdata = data[-checkrange:]
    if large:
        return len(subdata[subdata['pctChg'] > checkrate])
    else:
        return len(subdata[subdata['pctChg'] < checkrate])



def continuestylecheck(data, israise=True, continueday=2, checkrate=6.0):
    checkrange = len(data)
    for i in range(checkrange - continueday):
        total = 0
        nocontinue = False
        for j in range(continueday):
            if (israise and data['pctChg'][i + j] > 0) or (not israise and data['pctChg'][i + j] < 0):
                total += data['pctChg'][i]
            else:
                nocontinue = True
                break
        if (not nocontinue) and (total < checkrate or total > checkrate):
            return True
    return False


def daydownfind(data, aboveline='ma20', checkrange=6, checkrate=-6.0):
    subdata = data[-checkrange:]
    return (continuestylecheck(subdata, False, 2, checkrate) or continuestylecheck(subdata, False, 3, checkrate)) and subdata['low'][-1] > subdata[aboveline][-1]


def buttomtotopcheck(data, checkrange=30, checkrate=1.5):
    subdata = data[-checkrange:]
    return subdata['low'].min() * checkrate > subdata['high'].max()


def toptobottomcheck(data, code, checkrange=20, checkrate=0.15):
    subdata = data[-checkrange:]
    if '300' in code:
        checkrate *= 2
    falldata = subdata[subdata['high'] * (1 - checkrate) > subdata['low']]
    return len(falldata) <= 0


def mergeklineandcheckcenter(data):
    startitem = 0
    i = 0
    newData = []
    low = []
    high = []
    open = []
    close = []
    def appendData(start, end):
        low.append(data['low'][start])
        high.append(data['high'][end])
        open.append(data['open'][start])
        close.append(data['close'][end])

    while i < len(data) - 1:
        i += 1
        if data['pctChg'][startitem] * data['pctChg'][i] > 0:
            continue
        else:
            appendData(startitem, i - 1)
            # newData.append({ 'low': data['low'][startitem], 'high': data['high'][i - 1], 'open': data['open'][startitem], 'close': data['close'][i - 1] })
            startitem = i
    appendData(startitem, i - 1)
    newSeries = pd.DataFrame({'low': low, 'open': open, 'high': high, 'close': close})
    for i in range(len(newSeries) - 2):
        find, _ = centerCheck(newSeries[i:], checkbaseline=False)
        if find:
            return True
    return False