import pandas as pd
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch
import numpy as np
import os
import getstockid
import stockutils
from train_data_set import SocketTrainDataLoader

rootpath = getstockid.getrootpath()

# 参数名称	参数描述	说明
# date	交易所行情日期	格式：YYYY-MM-DD
# code	证券代码	格式：sh.600000。sh：上海，sz：深圳
# open	今开盘价格	精度：小数点后4位；单位：人民币元
# high	最高价	精度：小数点后4位；单位：人民币元
# low	最低价	精度：小数点后4位；单位：人民币元
# close	今收盘价	精度：小数点后4位；单位：人民币元
# preclose	昨日收盘价	精度：小数点后4位；单位：人民币元
# volume	成交数量	单位：股
# amount	成交金额	精度：小数点后4位；单位：人民币元
# adjustflag	复权状态	不复权、前复权、后复权
# turn	换手率	精度：小数点后6位；单位：%
# tradestatus	交易状态	1：正常交易 0：停牌
# pctChg	涨跌幅（百分比）	精度：小数点后6位
# peTTM	滚动市盈率	精度：小数点后6位
# psTTM	滚动市销率	精度：小数点后6位
# pcfNcfTTM	滚动市现率	精度：小数点后6位
# pbMRQ	市净率	精度：小数点后6位
# isST	是否ST	1是，0否

def feature_normalize(data):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mu)/std
root = '../training/data/'

def normalizeCol(data, cols):
    maxValue = -1
    for col in cols:
        maxCol = data[col].max()
        if maxValue < maxCol:
            maxValue = maxCol
    for col in cols:
        data[col] = data[col] / maxValue

def readconverttodataset():
    stockids = getstockid.readstockids()
    dataRowNum = 30
    # train_data
    train_data = np.array([])
    target_train = np.array([])
    target_test = np.array([])
    test_data = np.array([])
    for index, id in enumerate(stockids):
        idfile = os.path.join(rootpath, id + '.csv')
        iddata = pd.read_csv(idfile, index_col='date', parse_dates=['date'])
        macd,  _, _ = stockutils.calculateMACD(iddata['close'])
        iddata['macd'] = macd
        iddata['ma5'] = iddata['close'].rolling(5).mean()
        iddata['ma10'] = iddata['close'].rolling(10).mean()
        iddata['ma20'] = iddata['close'].rolling(20).mean()
        iddata['ma30'] = iddata['close'].rolling(30).mean()
        iddata['ma60'] = iddata['close'].rolling(60).mean()
        iddata['ma120'] = iddata['close'].rolling(120).mean()
        iddata['ma250'] = iddata['close'].rolling(250).mean()
        if not stockutils.pecheck(iddata, 100):
            continue

        preLen = len(iddata)
        # remove1 = iddata['tradestatus']==0
        # remove2 = iddata['isST']==1
        iddata.drop(iddata[iddata['tradestatus']==0].index, inplace=True)
        latlen = len(iddata)

        # if preLen != latlen:
        #     print('code:', iddata['code'][0])
        # print(iddata['open', 'high', 'low', 'preclose', 'adjustflag', 'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ', 'macd', 'ma5', 'ma10', 'ma20', 'ma30', 'ma60', 'ma250'][-10:].to_numpy())
        basedata = iddata.drop(['Unnamed: 0', 'code', 'volume', 'amount', 'tradestatus', 'isST', 'pctChg'], axis=1)
        normalizeCol(basedata, ['open', 'high', 'low', 'close', 'preclose', 'ma5', 'ma10', 'ma20', 'ma30', 'ma60', 'ma120', 'ma250'])
        # print(iddata.drop(['Unnamed: 0', 'code', 'volume', 'amount', 'tradestatus', 'isST'], axis=1)[-10:].to_numpy())
        socket_data = np.array([basedata[i: i + dataRowNum].to_numpy() for i in range(latlen - dataRowNum - 1)])
        # print(np.shape(socket_data))
        target_data = np.array([int(iddata['pctChg'][i + dataRowNum] * 10) for i in range(0, latlen - dataRowNum - 1)])

        keep_index = np.random.choice(np.arange(latlen - dataRowNum - 1), int(latlen * 0.5), False)
        socket_data = socket_data[keep_index]
        target_data = target_data[keep_index]
        keep_len = len(socket_data);
        test_index = np.random.choice(np.arange(keep_len), int(latlen * 0.2), False)
        train_index = np.delete(np.arange(keep_len), test_index)
        if test_data.any():
            test_data = np.concatenate((test_data, socket_data[test_index]))
        else:
            test_data = socket_data[test_index]
        print('test_data size:', np.shape(test_data)[0])
        if target_test.any():
            target_test = np.concatenate((target_test, target_data[test_index]))
        else:
            target_test = target_data[test_index]

        if train_data.any():
            train_data = np.concatenate((train_data, socket_data[train_index]))
        else:
            train_data = socket_data[train_index]
        print('train_data size:', np.shape(train_data)[0])
        if target_train.any():
            target_train = np.concatenate((target_train, target_data[train_index]))
        else:
            target_train = target_data[train_index]

        if (index + 1) % 100 == 0:
            np.save(os.path.join(root, 'train.npy'), arr=train_data)
            np.save(os.path.join(root, 'test.npy'), arr=test_data)
            np.savetxt(os.path.join(root, 'target_train.csv'), X=target_train, fmt="%d", delimiter=',')
            np.savetxt(os.path.join(root, 'target_test.csv'), X=target_test, fmt="%d", delimiter=',')
        print('Index ', index, ' complete.')

    np.save(os.path.join(root, 'train.npy'), arr=train_data)
    np.save(os.path.join(root, 'test.npy'), arr=test_data)
    np.savetxt(os.path.join(root, 'target_train.csv'), X=target_train, fmt="%d", delimiter=',')
    np.savetxt(os.path.join(root, 'target_test.csv'), X=target_test, fmt="%d", delimiter=',')
    print('Save data complete')
    # 每连续50组数据作为一个train data, dataRowNum = 50，下一个交易日的涨幅作为target，分类-200到200
    # 保存traindataset
    # 获取部分生成testdata


# 使用单纯的线性回归模型
# 使用CNN模型
# 使用GAN模型
# LSTM




if __name__ == '__main__':
    readconverttodataset()
    # dataloader = SocketTrainDataLoader(root)
    # for _, (data, target) in enumerate(dataloader):
    #     data = data.squeeze(0)
    #     print(np.shape(data))
    #     print(target)
