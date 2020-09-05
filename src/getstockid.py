import baostock as bs
import pandas as pd
import tushare as ts
import datetime
import dateutil
import os
pwd = os.getcwd()
idsCsvPath=os.path.join(os.path.abspath(os.path.dirname(pwd)+os.path.sep+"."), 'data', 'ids.csv')

basePath = '/User/liujinf/'
today = datetime.datetime.today()

checkday = (today - datetime.timedelta(weeks=4 * 52)).strftime('%Y-%m-%d')
print('Get day:', checkday);
start_date = (today - datetime.timedelta(weeks=4 * 52 + 1)).strftime('%Y-%m-%d')
end_date = (today - datetime.timedelta(weeks=4 * 52 - 1)).strftime('%Y-%m-%d')

# 1. use baostock
lg = bs.login()
rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
while (rs.error_code == '0') & rs.next():
    day = rs.get_row_data()
    if day[1]:
        checkday = day[0]
        break
pro = ts.pro_api()
ts.pro_api('dec07c5bd502e7b1ee2d5a9f105ea04f62552df07b3c8ab6551dc7d0')
data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date, list_status, market')
print(len(data))
d1 = data['name']
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
bs.logout()
