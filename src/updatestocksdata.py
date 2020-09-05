import pandas as pd
import numpy as np
import tushare as ts
import os
import baostock as bs

pwd = os.getcwd()
rootFolder = os.path.join(os.path.abspath(os.path.dirname(pwd)+os.path.sep+"."), 'data')
idsCsvPath=os.path.join(rootFolder, 'ids.csv')

pdIds = pd.read_csv(idsCsvPath)['ts_code']

# pro = ts.pro_api()
# ts.pro_api('dec07c5bd502e7b1ee2d5a9f105ea04f62552df07b3c8ab6551dc7d0')

for id in pdIds:

#    idData = pro.daily(td_code=id)
#    idPath = os.path.join(rootFolder, id + '.csv')
#    idData.to_csv(idPath)


