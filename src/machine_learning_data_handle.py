import pandas as pd
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import os
import getstockid
import stockutils
import torchvision.transforms as transforms
import torchvision.datasets as dset
from train_data_set import SocketTrainDataLoader
from net_classes import LinearNet
from net_classes import CnnNet
from net_classes import LSTMNet
import shutil
from datetime import datetime

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

def readconverttodataset(savebyid):
    stockids = getstockid.readstockids()
    dataRowNum = 48
    # train_data
    train_data = np.array([])
    target_train = np.array([])
    target_test = np.array([])
    test_data = np.array([])
    total = len(stockids)
    for index, id in enumerate(stockids):
        if savebyid:
            folder = os.path.join(root, id);
            if not os.path.exists(folder):
                os.mkdir(folder)
            else:
                continue
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
        iddata = iddata[250:]
        # if not stockutils.pecheck(iddata, 100):
        #     continue

        preLen = len(iddata)
        # remove1 = iddata['tradestatus']==0
        # remove2 = iddata['isST']==1
        iddata.drop(iddata[iddata['tradestatus']==0].index, inplace=True)
        latlen = len(iddata)

        # if preLen != latlen:
        #     print('code:', iddata['code'][0])
        # print(iddata['open', 'high', 'low', 'preclose', 'adjustflag', 'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ', 'macd', 'ma5', 'ma10', 'ma20', 'ma30', 'ma60', 'ma250'][-10:].to_numpy())
        basedata = iddata.drop(['Unnamed: 0', 'code', 'volume', 'amount', 'adjustflag', 'tradestatus', 'isST', 'pctChg', 'pbMRQ'], axis=1)
        normalizeCol(basedata, ['open', 'high', 'low', 'close', 'preclose', 'ma5', 'ma10', 'ma20', 'ma30', 'ma60', 'ma120', 'ma250'])
        # print(iddata.drop(['Unnamed: 0', 'code', 'volume', 'amount', 'tradestatus', 'isST'], axis=1)[-10:].to_numpy())
        socket_data = np.array([basedata[i: i + dataRowNum].to_numpy() for i in range(latlen - dataRowNum - 1)])
        # print(np.shape(socket_data))
        target_data = np.array([round(iddata['pctChg'][i + dataRowNum] * 10) for i in range(0, latlen - dataRowNum - 1)])
        # 给每一个计算数据，则不需要进行裁剪
        # 只保留数据的50%，从中选择测试数据和训练数据
        if not savebyid:
            keep_index = np.random.choice(np.arange(latlen - dataRowNum - 1), int(latlen * 0.5), False)
            socket_data = socket_data[keep_index]
            target_data = target_data[keep_index]
        keep_len = len(socket_data)
        test_index = np.random.choice(np.arange(keep_len), int(keep_len * 0.2), False)
        train_index = np.delete(np.arange(keep_len), test_index)
        # for lstm generate data by date order, do not shuffle
        # np.random.shuffle(train_index)
        if test_data.any() and not savebyid:
            test_data = np.concatenate((test_data, socket_data[test_index]))
        else:
            test_data = socket_data[test_index]
        print('test_data size:', np.shape(test_data)[0])
        if target_test.any() and not savebyid:
            target_test = np.concatenate((target_test, target_data[test_index]))
        else:
            target_test = target_data[test_index]

        if train_data.any() and not savebyid:
            train_data = np.concatenate((train_data, socket_data[train_index]))
        else:
            train_data = socket_data[train_index]
        print('train_data size:', np.shape(train_data)[0])
        if target_train.any() and not savebyid:
            target_train = np.concatenate((target_train, target_data[train_index]))
        else:
            target_train = target_data[train_index]

        if savebyid:

            np.save(os.path.join(root, id, 'train.npy'), arr=train_data)
            np.save(os.path.join(root, id, 'test.npy'), arr=test_data)
            np.savetxt(os.path.join(root, id, 'target_train.csv'), X=target_train, fmt="%d", delimiter=',')
            np.savetxt(os.path.join(root, id, 'target_test.csv'), X=target_test, fmt="%d", delimiter=',')
        else:
            if (index + 1) % 100 == 0:
                np.save(os.path.join(root, 'train.npy'), arr=train_data)
                np.save(os.path.join(root, 'test.npy'), arr=test_data)
                np.savetxt(os.path.join(root, 'target_train.csv'), X=target_train, fmt="%d", delimiter=',')
                np.savetxt(os.path.join(root, 'target_test.csv'), X=target_test, fmt="%d", delimiter=',')
        print('Index {} complete. total is {} to {:.2%}'.format(index + 1, total, (index + 1) / total))

    # np.save(os.path.join(root, 'train.npy'), arr=train_data)
    # np.save(os.path.join(root, 'test.npy'), arr=test_data)
    # np.savetxt(os.path.join(root, 'target_train.csv'), X=target_train, fmt="%d", delimiter=',')
    # np.savetxt(os.path.join(root, 'target_test.csv'), X=target_test, fmt="%d", delimiter=',')
    print('Save data complete')
    # 每连续48组数据作为一个train data, dataRowNum = 50，下一个交易日的涨幅作为target，分类-200到200
    # 保存traindataset
    # 获取部分生成testdata


# 使用单纯的线性回归模型
# 使用CNN模型
# 使用GAN模型
# LSTM

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def linearTrain(id):
    train_data = SocketTrainDataLoader(os.path.join(root, id))
    train_loader = data_utils.DataLoader(dataset=train_data, shuffle=True, num_workers=2)
    LR = 0.002
    # for linear net type 1
    # linearNet = LinearNet(50 * 17)
    # optimizer = torch.optim.Adam(linearNet.parameters(), lr=LR)
    # for cnn net typpe 2
    # cnnTrainer = CnnNet()
    # print(cnnTrainer)
    # optimizer = torch.optim.Adam(cnnTrainer.parameters(), lr=LR)
    lstmnet = LSTMNet(16, 2048, 1, 200)
    hidden = lstmnet.init_hidden(1)
    optimizer = torch.optim.Adam(lstmnet.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()  #  该函数内包含softmax，所以训练函数内最后一步不需要再加softmax的激励函数
    EPOCHS = 20
    test_data = SocketTrainDataLoader(os.path.join(root, id))
    test_loader = data_utils.DataLoader(dataset=test_data, shuffle=True, num_workers=2)
    for epoch in range(EPOCHS):
        start = datetime.now()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            try:
                # linear net type 1
                # batch_x = batch_x.view(-1, 50 * 17)
                # output = linearNet(batch_x.float())

                # CNN net type 2
                # batch_x = torch.unsqueeze(batch_x, 1)
                # output = cnnTrainer(batch_x.float())
                output, hidden = lstmnet(batch_x.float(), hidden)
                hidden = repackage_hidden(hidden)
                loss = loss_func(output, batch_y)
                # print('train rate: %.6f' % loss.data)
                optimizer.zero_grad()
                loss.backward()
                # for name, parms in linearNet.named_parameters():
                #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
                optimizer.step()
            except Exception as err:
                print(err)
        print('epoch training time: %d in %d' % ((datetime.now()-start).seconds, epoch))
        # validate with test
        total = .0
        start = datetime.now()
        torch.save(lstmnet.state_dict(), '../model/lstm.pkl')
        hidden = lstmnet.init_hidden(1)
        with torch.no_grad():
            for step, (test_x, test_y) in enumerate(test_loader):
                # linear net type 1
                # test_x = test_x.view(-1, 50 * 17)
                # pred_y = linearNet(test_x.float())

                # CNN net type 2
                # test_x = torch.unsqueeze(test_x, 1)
                # pred_y = cnnTrainer(test_x.float())

                pred_y, hidden = lstmnet(test_x.float(), hidden)
                hidden = repackage_hidden(hidden)
                # pred_y = torch.max(pred_y, 1)
                loss = loss_func(pred_y, test_y)
                # _, pred_label = torch.max(pred_y, 1)
                total += loss.item()
                # for LSTM best result accuracy = 4.03
        accuracy = total / len(test_data)
        print('Accuracy=%.2f' % accuracy)
        print('epoch test time: %d in %d' % ((datetime.now() - start).seconds, epoch))


def clearOldData():

    for _, dirs, files in os.walk(root):
        for dir in dirs:
            if dir == '.' or dir == '..':
                continue
            shutil.rmtree(os.path.join(root, dir))

if __name__ == '__main__':
    # clearOldData()
    # readconverttodataset(True)
    linearTrain('000001.SZ')


    # for _, (data, target) in enumerate(dataloader):
    #     data = data.squeeze(0)
    #     print(np.shape(data))
    #     print(target)
