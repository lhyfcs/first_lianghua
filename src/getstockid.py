import baostock as bs
import pandas as pd
import tushare as ts
import datetime
import dateutil
import os
pwd = os.getcwd()
idsCsvPath=os.path.join(os.path.abspath(os.path.dirname(pwd)+os.path.sep+"."), 'data', 'ids.csv')

basePath = '/User/liujinf/'

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
def readstockids():
    rootfolder = getrootpath()
    idscsvpath = os.path.join(rootfolder, 'ids.csv')
    pdIds = pd.read_csv(idscsvpath)['ts_code']
    return pdIds

def getrootpath():
    return os.path.join(os.path.abspath(os.path.dirname(pwd) + os.path.sep + "."), 'data')


def todaydate():
    today = datetime.datetime.today()
    searchday = (today - datetime.timedelta(weeks=4 * 52)).strftime('%Y-%m-%d')
    print('Get day:', searchday)
    start_date = (today - datetime.timedelta(weeks=4 * 52 + 1)).strftime('%Y-%m-%d')
    end_date = (today - datetime.timedelta(weeks=4 * 52 - 1)).strftime('%Y-%m-%d')

# 1. use baostock
    rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
    while (rs.error_code == '0') & rs.next():
        day = rs.get_row_data()
        if day[1]:
            searchday = day[0]
            break
    return searchday

class GetStockId:
    def __init__(self):
        pass

    def downloadStockIdByDate(self):
        checkday = todaydate()
        # pro = ts.pro_api()
        pro = ts.pro_api('dec07c5bd502e7b1ee2d5a9f105ea04f62552df07b3c8ab6551dc7d0')
        data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date, list_status, market')
        print(len(data))
        data = data[(data['list_date'] < checkday) & (~data['name'].str.contains('ST'))]
        print(len(data))
        data.to_csv(idsCsvPath)
# rs = bs.query_all_stock(checkday)
# data_list = []
# while (rs.error_code == '0') & rs.next():
#     data_list.append(rs.get_row_data())
#
# result = pd.DataFrame(data_list, columns=rs.fields)
# result.to_csv('./data/ids.csv')

