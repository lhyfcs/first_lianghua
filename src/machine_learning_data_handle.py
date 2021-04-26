import pandas as pd
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch
import numpy as np
import os
import getstockid
import stockutils

rootpath = getstockid.getrootpath()

def feature_normalize(data):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mu)/std

def readconverttodataset():
    stockids = getstockid.readstockids()
    for index, id in enumerate(stockids):
        # id = '002132.SZ'
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
        normal1 = torch.tensor(iddata['close'].values.astype(np.float32))
        normal1 = F.normalize(normal1, p=2, dim=0)
        # normal = feature_normalize(iddata['close'])
        normal = normal1.numpy()
        print('max', normal.max(), 'min', normal.min())



if __name__ == '__main__':
    readconverttodataset()
