import pandas as pd
import numpy as np



def boll_bands(data, ndays):
    """
    : param data 计算数据
    : param ndays 计算使用的简单移动均线周期
    """
    # pandas.std() 默认除以n-1，即是无偏的，如果想和numpy.std()一样有偏，需要加上参数ddof=0
    # 此处添加ddof的原因是wind和yahoo的计算均采用有偏值进行计算
    ma = pd.Series(np.round(data['close'].rolling(ndays).mean(), 2), name='MA%s' % ndays)
    # 计算nday标准差，有偏
    std = pd.Series(np.round(data['close'].rolling(ndays).std(ddof=0), 2))
    b1 = ma + (2 * std)
    data['bb-u'] = b1
    # B1 = pd.Series(b1, name='UpperBollingerBand')
    # data = data.join(ma)
    # data = data.join(B1)
    b2 = ma - (2 * std)
    data['bb-l'] = b2
    data['bb-m'] = ma
    # B2 = pd.Series(b2, name='LowerBollingerBand')
    # data = data.join(B2)


# RSI = N日内收盘价涨数的均值/N日内收盘价涨和跌的均值 * 100
def rsi_cal(data, periods = 20):
    datalist = (data['preclose'] == 0).to_list()
    # data.iloc[(data['preclose'] == 0).to_list(), 'change'] = 0 #如果是首日，change记为0
    data['x'] = data['change'].apply(lambda x: max(x, 0))
    data['rsi'] = (data['x'].ewm(alpha=1/periods, adjust=False).mean() / (np.abs(data['change'])).ewm(alpha=1/periods, adjust=False).mean()) * 100


# 原理： 计算N日RSV=（今日收盘-N日最低）/（N日最高-N日最低） * 100
# K = （l-（l/M1））* 前一日K值 + l/M1 * RSV
# D = （l-（l/M1））*前一日D值 + l/M1 * K值
# J = 3 * K - 2 * D
def DKJ(data, N, M1, M2):
    lowlist = data['low'].rolling(N).min()
    lowlist.fillna(value=data['low'].expanding().min(), inplace=True)
    highlist = data['high'].rolling(N).max()
    highlist.fillna(value=data['high'].expanding().max(), inplace=True)
    rsv = (data['close'] - lowlist) / (highlist - lowlist) * 100
    data['kdj_k'] = rsv.ewm(alpha=1/M1, adjust=False).mean()
    data['kdj_d'] = data['kdj_k'].ewm(alpha=1/M2, adjust=False).mean()
    data['kdj_j'] = 3.0 * data['kdj_k'] - 2.0 * data['kdj_d']


# DMA 计算指标
# 计算DIF：close的N1日移动平均- close的N2日移动平均
# 计算AMA：DIF的M日移动平均
def DMA(data, N1, N2, M):
     data['DIF'] = data['close'].rolling(N1).mean() - data['close'].rolling(N2).mean()
     data['AMA'] = data['close'].rolling(M).mean()


# 计算均线
def mavaluecalculate(data, avgs = [5, 10, 20, 30, 60, 120, 250]):
    for avg in avgs:
        data[f'ma{avg}'] = data['close'].rolling(avg).mean()
