import pandas as pd
import numpy as np
import tushare as ts
import os
import baostock as bs
import getstockid


pwd = os.getcwd()
rootFolder = os.path.join(os.path.abspath(os.path.dirname(pwd)+os.path.sep+"."), 'data')
idsCsvPath=os.path.join(rootFolder, 'ids.csv')

pdIds = pd.read_csv(idsCsvPath)['ts_code']
lg = bs.login()
# pro = ts.pro_api()
# ts.pro_api('dec07c5bd502e7b1ee2d5a9f105ea04f62552df07b3c8ab6551dc7d0')
checkday = getstockid.todaydate()
for id in pdIds:
    # idData = ts.get_hist_data(id.split('.')[0])
    # idPath = os.path.join(rootFolder, id + '.csv')
    # idData.to_csv(idPath)
    ids = id.split('.')
    print(ids[1].lower() + '.' + ids[0])
    rs = bs.query_history_k_data_plus(ids[1].lower() + '.' + ids[0],
                                      "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,isST",
                                      start_date=checkday, end_date='',
                                      frequency="d", adjustflag="3")
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    data = pd.DataFrame(data_list, columns=rs.fields)
    data['ma5'] = data['close'].rolling(5).mean()
    data['ma10'] = data['close'].rolling(5).mean()
    data['ma20'] = data['close'].rolling(5).mean()
    data['ma30'] = data['close'].rolling(5).mean()
    data['ma60'] = data['close'].rolling(5).mean()
    data['ma250'] = data['close'].rolling(5).mean()
    data = data.dropna(how='any')
    data.to_csv(os.path.join(rootFolder, id + '.csv'))
bs.logout()

