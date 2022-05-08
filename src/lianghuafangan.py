

#
def methoddownbacktoyearline():
    pass

from enum import Enum

# stock status struct
# stockid
# count
# buyprice
# buy date

MAX_BUY_DATA = 100
class stock:
    def __init__(self, id, count, buyprice, buydate):
        # when id is not none, hold money should be 0
        self.id = id
        self.count = count
        self.buyprice = buyprice
        self.buydate = buydate
        self.holdmoney = 0

    def reset(self):
        self.id = None
        self.count = 0
        self.buyprice: float = 0.0
        self.buydate = ''

    def buyorder(self, id, money, price, date):
        self.id = id
        self.buyprice = price
        self.buydate = date
        self.count = int(money / price * 100)
        return money - self.count * price * 100


    def setholdmoney(self, holdmoney):
        self.holdmoney = holdmoney

    def istakemoney(self):
        return self.id is None


class Ordertype(Enum):
    SALE = 0
    BUY = 1


class action:
    def __init__(self, ordertype, count, price, date, name):
        self.ordertype = ordertype
        self.ordercount = count
        self.orderprice = price
        self.date = date
        self.name = name

# action struct
# order

class buydata:
    def __init__(self, id, price, weight, name):
        self.id = id
        self.price = price
        self.weight = weight
        self.name = name


class lianghuasystem:
    def __init__(self, moneyDivide, rundays, buyRules, saleRules, total, stocksdata):
        self.divide = moneyDivide
        self.rundays = rundays
        self.buyRules = buyRules
        self.saleRules = saleRules
        self.total = total
        self.stockStatus = []
        self.preStatus = []
        self.actions = []
        self.stockdata = stocksdata
        self.curdateindex
        self.operateActions = []

    def updateStauts(self, dateindex, salerules, total=0.0):
        # check if need sale
        # 1. loop all stocks, check if self current stock
        # 2. average money
        # 3. check if need by new stock
        curdate = self.totaldata[dateindex].date
        for index in range(self.stockStatus):
            stock = self.stockStatus[index]
            if not stock.stockid:
                total += stock.money
            else:
                totaldata = self.stocksdata[stock.stockid]
                for salerule in salerules:
                    today = self.totaldata[dateindex]
                    sale, saleprice = salerule(today)
                    if sale:
                        total += stock.count * saleprice
                        self.operateActions.append(action(Ordertype.SALE, stock.count, saleprice, curdate))
                        stock.reset()

        moneystocks = list([lambda s: s.istakemoney(), self.stockstatus])
        moneyforone = total / len(moneystocks)
        buys = []

        for buyrule in self.buyrules:
            for data in self.stockdata:
                buy, buyprice, buyweight, name = buyrule(data, dateindex)
                if buy:
                    buys.append(buydata(data.id, buyprice, buyweight, name))
                if len(buys) >= MAX_BUY_DATA:
                    break
            if len(buys) >= MAX_BUY_DATA:
                break
        sorted(buys, lambda x: x.buyweight, reverse=True)
        buyindex = 0
        stockindex = 0

        while buyindex < len(buys) and stockindex < len(moneystocks):
            buy = buys[buyindex]
            # 够一手
            if moneyforone > buy.price * 100:
                stock = moneystocks[stockindex]
                leave = stock.buyorder(id, moneyforone, buy.price, curdate)
                total += leave - moneyforone
                self.operateActions.append(action(Ordertype.BUY, stock.count, buy.price, curdate))

    def updateStauts(self):
        self.preStatus = self.stockStatus[:]

        today = self.stockStatus[self.curdataindex]
        for salerule in self.saleRules:
            if salerule(today):
                pass
