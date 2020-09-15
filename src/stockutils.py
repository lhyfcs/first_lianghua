import numpy as np


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


def findvaluepos(list, value):
    l = len(list)
    for i in range(int(l / 2)):
        if value == list[i]:
            return i
        elif value == list[-i]:
            return l - i


def averagelineUstyle(data, linename, checkrange=60):
    linedata = data[linename][-checkrange:]
    # 检测算法，1. 两边高，中间低
    # 2. 低点在中间位置
    # 3. 高点在两边
    left = linedata[0]
    right = linedata[-1]
    print(checkrange/2)
    mid = linedata[int(checkrange/2)]
    if left < mid or right < mid:
        return False
    values = linedata.values
    posMax = findvaluepos(values, linedata.max())
    posMin = findvaluepos(values, linedata.min())
    checkvalue = checkrange / 10
    if abs(posMin - 0) < checkvalue * 2 or abs(posMin - checkrange) < checkvalue * 2:
        return False
    if abs(posMax - 0) > checkvalue and abs(posMax - checkrange) > checkvalue:
        return False
    return True


def raiseanddowncheck(data, maxrate ,checkrange=60):
    linedata = data['pctChg'][-checkrange:]
    print(len(linedata[abs(linedata) > maxrate]))
    return len(linedata[abs(linedata) > maxrate]) <= 2;


def daytopsingleline(data, chechrate=2.5, checkmax=True, checkrange=60):
    high = data['high'][-checkrange:]
    if checkmax and high[-1] != high.max():
        return False
    last = data.iloc[-1]
    top = max(last.close, last.open)
    if (last.high - top) / top * 100 > chechrate:
        return True
    return False


def centerCheck(data, baseline='ma5'):
    # center check, 5, 4, 3 day
    #       1. side is raise
    #       2. left low is the lowest
    for i in range(3, 5):
        subdata = data[-i:]
        left = subdata.iloc[0]
        right = subdata.iloc[-1]
        if left.low != subdata['low'].min():
            continue
        if left.open > left.close or right.open > right.close:
            continue
        if len(subdata[subdata['close'] < subdata[baseline]]) > 0:
            continue
        return True, i
    return False


def pecheck(data):
    pe = data['peTTM'][-1]
    return pe > 0 or pe < -50


def continuefitfunction(data, fun, continuedata=2):
    for i in range(continuedata):
        subdata = data[:-i]
        if not fun(subdata):
            return False
    return True




