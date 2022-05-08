import pandas as pd
import numpy as np
import getstockid
import os
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib as mpl
from matplotlib.widgets import Button
import mplfinance as mpf
from mpl_finance import candlestick2_ochl
from matplotlib.font_manager import FontManager
from matplotlib.widgets import Cursor
from stockutils import calculateMACD
import indicatorCal
import datetime
import time
ff = FontManager()
# ff.ttflist   list all support font name

rootpath = getstockid.getrootpath()


def draw_fun_ld(file_path):
    df = pd.read_csv(file_path)
    fig, ax = plt.subplots(facecolor=(0, 0.3, 0.5), figsize=(12, 8))
    fig.subplots_adjust(bottom=0.1)
    ax.grid(True)
    ax.xaxis_date()
    plt.xticks(rotation=30)
    candlestick2_ochl(ax, opens=df['open'].values, closes=df['close'].values, highs=df['high'].values,
                      lows=df['low'].values, width=0.6, colorup='red', colordown='green')
    df['close'].rolling(5).mean().plot(color='white', label='5day')
    df['close'].rolling(10).mean().plot(color='yellow', label='10day')
    df['close'].rolling(20).mean().plot(color='red', label='20day')
    df['close'].rolling(60).mean().plot(color='gray', label='60day')
    df['close'].rolling(120).mean().plot(color='blue', label='120day')
    df['close'].rolling(250).mean().plot(color='cyan', label='250day')
    # plt.legend(loc='best')
    # plt.xticks(range(len(df.index.values)), df.index.values, rotation=30)

    plt.title('601518_')
    plt.show()


def draw_fun_new(file_path, id, name):
    # 设置基本参数
    # type： 绘制图形的类型，candle，renko，line等
    # mav(moving average): 均线类型，此处设置7，30，60等
    # volume: 布尔类型，设置是否显示成交量
    # title: 设置标题
    # y_label: 设置纵轴主标题
    # y_label_low: 设置成交量一栏的标题
    # figratio:设置图形纵横比
    # figscale: 设置图形尺寸（数值越大，图形质量越高）
    df = pd.read_csv(file_path)
    df = df[['date', 'open', 'close', 'high', 'low', 'volume']]
    df['Date'] = pd.to_datetime((df['date']))
    df.set_index(['Date'], inplace=True)
    param = dict(left=100, drawrange=100)
    # plt.ion()
    # plt.figure(1)
    # plt.figure(2)
    dfdraw = df[param['left']:param['left'] + param['drawrange']]
    kwargs = dict(
        type='candle',
        mav=(5, 10, 20),
        volume=True,
        title='\Stock %s, %s candle_line' % (id, name),
        ylabel='OHLC candles',
        ylabel_lower='Trade Volume',
        figratio=(15, 10),
        figscale=5)
    mc = mpf.make_marketcolors(up='red', down='green', edge='i', wick='i', volume='in', inherit=True)
    # gridaxis: 设置网格线位置
    # gridstyle: 设置网格线线型
    # y_on_right: 设置Y轴是否在右
    s = mpf.make_mpf_style(gridaxis='both', gridstyle='-.', y_on_right=False, marketcolors=mc)
    # 设置均线颜色
    mpl.rcParams['axes.prop_cycle'] = cycler(color=['dodgerblue', 'deeppink', 'navy', 'teal', 'maroon', 'darkorange', 'indigo'])
    mpl.rcParams['lines.linewidth'] = .5
    # show_nontrading: 是否显示非交易日
    # savegif: 导出图片
    mpf.plot(dfdraw, **kwargs, style=s, show_nontrading=False)
    leftDraw = plt.axes([0.5, 0.015, 0.2, 0.045])
    def drawLeft(event):
        param['left'] = param['left'] + 100
        dfdraw = df[param['left']:param['left'] + param['drawrange']]
        mpf.plot(dfdraw, **kwargs, style=s, show_nontrading=False)
        # mpf.draw()
    leftBtn = Button(leftDraw, 'Move left')
    leftBtn.on_clicked(drawLeft)

    rightDraw = plt.axes([0.8, 0.015, 0.5, 0.045])
    def drawright(event):
        param.left = param.left - 100
        dfdraw = df[param.left: param.left + param.drawrange]
        mpf.plot(dfdraw, **kwargs, style=s, show_nontrading=False)
        # mpf.draw()

    rightBtn = Button(rightDraw, 'Move right')
    rightBtn.on_clicked(drawright)
    # plt.draw()
    # plt.ioff()
    plt.show()


# template style define
template_color='(0.82, 0.83, 0.85)'
fig_color = '(0.24, 0.24, 0.24)'
my_color = mpf.make_marketcolors(up='r', down='aqua', edge='i', wick='i', volume='i')
# https://stackoverflow.com/questions/60599812/how-can-i-customize-mplfinance-plot
# https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&color_mode=auto&commit=1f710caa65d872a761ccb5831fb1befd7324881f&device=unknown&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d6174706c6f746c69622f6d706c66696e616e63652f316637313063616136356438373261373631636362353833316662316265666437333234383831662f6578616d706c65732f7374796c65732e6970796e62&logged_in=false&nwo=matplotlib%2Fmplfinance&path=examples%2Fstyles.ipynb&platform=android&repository_id=226144726&repository_type=Repository&version=98
# 设置黑色背景，base_mpf_style， 设置价格字体颜色base_mpl_style
# base_mpf_style='nightclouds', base_mpl_style='seaborn',
my_style = mpf.make_mpf_style(marketcolors=my_color, figcolor=fig_color, gridcolor=fig_color, facecolor='black', gridaxis='horizontal', gridstyle='-')

title_font = {'fontname': 'pingfang HK',
              'size': '16',
              'color': 'black',
              'weight': 'bold',
              'va':     'bottom',
              'ha':     'center'}

large_red_font = {'fontname': 'Arial',
                  'size':      '24',
                  'color':      'red',
                  'weight':     'bold',
                  'va':         'bottom'}

large_green_font = {'fontname': 'Arial',
                    'size':     '24',
                    'color':    'green',
                    'weight':   'bold',
                    'va':       'bottom'}

small_red_font = {'fontname': 'Arial',
                    'size':     '12',
                    'color':    'red',
                    'weight':   'bold',
                    'va':       'bottom'}

small_green_font = {'fontname': 'Arial',
                    'size':     '12',
                    'color':    'green',
                    'weight':   'bold',
                    'va':       'bottom'}

normal_label_font = {'fontname': 'pingfang HK',
                    'size':     '12',
                    'color':    'black',
                    'va':       'bottom',
                     'ha':      'right'}

normal_font = {'fontname': 'Arial',
                    'size':     '12',
                    'color':    'black',
                    'va':       'bottom',
               'ha':    'left'}

class InterCandle:
    def fillfulldata(self, id, name):
        idfile = os.path.join(rootpath, id + '.csv')
        df = pd.read_csv(idfile)
        df['Date'] = pd.to_datetime((df['date']))
        df['change'] = df['close'] - df['preclose']
        macd, diff, dea = calculateMACD(df['close'])
        df['macd-m'] = diff
        df['macd-s'] = dea
        df['macd-h'] = macd
        # df['ma5'] = df['close'].rolling(5).mean()
        # df['ma10'] = df['close'].rolling(10).mean()
        # df['ma20'] = df['close'].rolling(20).mean()
        # # df['ma30'] = df['close'].rolling(30).mean()
        # df['ma60'] = df['close'].rolling(60).mean()
        # df['ma120'] = df['close'].rolling(120).mean()
        # df['ma250'] = df['close'].rolling(250).mean()
        if id.startswith('300') or id.startswith('688'):
            df['upper_lim'] = df['preclose'] * 1.2
            df['lower_lim'] = df['preclose'] * 0.8
        else:
            df['upper_lim'] = df['preclose'] * 1.1
            df['lower_lim'] = df['preclose'] * 0.9
        df.set_index(['Date'], inplace=True)
        self.df = df
        # load week data
        weekfile = os.path.join(rootpath, 'week', id + '.csv')
        weekdf = pd.read_csv(weekfile)
        weekdf['preclose'] = weekdf['close'].shift(1, fill_value=0)
        weekdf['Date'] = pd.to_datetime((weekdf['date']))
        weekdf['change'] = weekdf['close'] - weekdf['preclose']
        macd, diff, dea = calculateMACD(weekdf['close'])
        weekdf['macd-m'] = diff
        weekdf['macd-s'] = dea
        weekdf['macd-h'] = macd
        self.weekdf = weekdf
        weekdf.set_index(['Date'], inplace=True)


    def addframecontext(self, fig, id, name):
        """
        添加界面上的各种组建和参数文字
        """
        # 添加图标，数字代表了左下角的坐标，以及款0.88，高0.60
        self.ax1 = fig.add_axes([0.06, 0.25, 0.88, 0.60])
        self.ax2 = fig.add_axes([0.06, 0.15, 0.88, 0.10], sharex=self.ax1)
        self.ax3 = fig.add_axes([0.06, 0.05, 0.88, 0.10], sharex=self.ax1)
        # 设置三张图标的Y轴标签
        self.ax1.set_ylabel('price')
        self.ax2.set_ylabel('volume')
        self.ax3.set_ylabel('mace')
        # 在figure对象上添加文本对象，显示各种价格和标题
        self.t1 = fig.text(0.50, 0.94, '%s - %s' % (id, name), **title_font)
        self.t2 = fig.text(0.12, 0.90, '开/收: ', **normal_label_font)
        # f'{np.round(last_data["open"], 3)}/{np.round(last_data["close"], 3)}'
        self.t3 = fig.text(0.14, 0.89, '', **large_red_font)
        change_font = small_red_font
        # if last_data['change'] < 0:
        #     change_font = small_green_font
        self.t4 = fig.text(0.17, 0.86, '涨跌: ', **normal_label_font)
        # f'{np.round(last_data["change"], 3)}'
        self.t5 = fig.text(0.17, 0.86, '', **small_red_font)
        self.t6 = fig.text(0.27, 0.86, '涨跌幅: ', **normal_label_font)
        # f'{np.round(last_data["change"] / last_data["preclose"] * 100, 2)}%'
        self.t7 = fig.text(0.28, 0.86, '', **small_red_font)
        # f'{last_data.name.date()}'
        self.t8 = fig.text(0.12, 0.86, '', **normal_label_font)
        self.t9 = fig.text(0.40, 0.90, '高: ', **normal_label_font)
        # f'{last_data["high"]}'
        self.t10 = fig.text(0.40, 0.90, '', **small_red_font)
        self.t11 = fig.text(0.40, 0.86, '低: ', **normal_label_font)
        # f'{last_data["low"]}'
        self.t12 = fig.text(0.40, 0.86, '', **small_green_font)
        self.t13 = fig.text(0.55, 0.90, '量(万手): ', **normal_label_font)
        # f'{np.round(last_data["volume"] / 10000, 3)}'
        self.t14 = fig.text(0.55, 0.90, '', **normal_font)
        self.t15 = fig.text(0.55, 0.86, '额(亿元): ', **normal_label_font)
        # f'{last_data["amount"]}'
        self.t16 = fig.text(0.55, 0.86, '', **normal_font)
        self.t17 = fig.text(0.70, 0.90, '涨停: ', **normal_label_font)
        # f'{np.round(last_data["upper_lim"], 3)}'
        self.t18 = fig.text(0.70, 0.90, '', **small_red_font)
        self.t19 = fig.text(0.70, 0.86, '跌停: ', **normal_label_font)
        # f'{np.round(last_data["lower_lim"], 3)}'
        self.t20 = fig.text(0.70, 0.86, '', **small_green_font)
        self.t21 = fig.text(0.85, 0.86, '昨收: ', **normal_label_font)
        # f'{last_data["preclose"]}'
        self.t22 = fig.text(0.85, 0.86, '', **normal_font)
        self.t23 = fig.text(0.85, 0.90, '换手率: ', **normal_label_font)
        self.t24 = fig.text(0.85, 0.90, '', **normal_font)

    def loadnewiddata(self, id, loadcontext=False):
        # load ids data, include stock id and stock name
        self.idsinfo = pd.read_csv(os.path.join(rootpath, 'ids.csv'))
        self.name = self.idsinfo.loc[self.idsinfo['symbol'] == int(id.split('.')[0])]['name'].values[0]
        self.id = id
        self.fillfulldata(id, self.name)
        self.calculateindicators()
        # df = df[['date', 'open', 'close', 'high', 'low', 'volume', 'amount']]
        self.style = my_style
        # show the latest data
        self.idx_range = 150
        self.resetdrawdata()
        self.idx_start = len(self.drawdata) - self.idx_range - 1

        if loadcontext:
            self.addframecontext(self.fig, id, self.name)
        self.pressed = False
        self.xpress = None
        self.avg_type = 'ma'
        self.indicator = 'macd'

    def calculateindicators(self):
        # indicator for day data
        indicatorCal.boll_bands(self.df, 20)
        indicatorCal.DKJ(self.df, 9, 3, 3)
        indicatorCal.DMA(self.df, 30, 5, 8)
        indicatorCal.rsi_cal(self.df, 20)
        indicatorCal.mavaluecalculate(self.df)
        # indicator for week data
        indicatorCal.boll_bands(self.weekdf, 20)
        indicatorCal.DKJ(self.weekdf, 9, 3, 3)
        indicatorCal.rsi_cal(self.weekdf, 20)
        indicatorCal.DMA(self.weekdf, 20, 3, 5)
        indicatorCal.mavaluecalculate(self.weekdf)

    def __init__(self, ids):
        self.stockids = ids
        self.curstockindex = 0
        self.fig = mpf.figure(style=my_style, figsize=(15.9, 10.6), facecolor=(0.82, 0.83, 0.85))
        # choose the first id to show
        self.loadnewiddata(self.stockids[self.curstockindex], True)

        # 下面代码在__init__ 中，告诉matplotlib哪些回调函数用于响应哪些事件
        self.cursor = Cursor(self.ax1, useblit=False, color='red', linewidth=1)
        # 鼠标按下事件与self.on_press回调函数绑定
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        # 鼠标按键释放事件与self.on_release回调函数绑定
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        # 鼠标移动事件与self.on_motion回调函数绑定
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        # 新增回调函数和鼠标滚轮事件绑定
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        # 新增键盘按下消息响应
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)



    def on_key(self, event):
        if event.key == 'w':
            self.drawdatatype = 'w'
            self.drawdata = self.weekdf
            self.idx_start = len(self.drawdata) - self.idx_range - 1
            self.refreshwholeplot()
        elif event.key == 'd':
            self.drawdatatype = 'd'
            self.drawdata = self.df
            self.idx_start = len(self.drawdata) - self.idx_range - 1
            self.refreshwholeplot()
        elif event.key == 'n':
            self.resetdrawdata()
            if self.curstockindex < len(self.stockids) - 1:
                self.curstockindex += 1
                self.loadnewiddata(self.stockids[self.curstockindex])
                self.refreshwholeplot()
        elif event.key == 'p':
            self.resetdrawdata()
            if self.curstockindex > 0:
                self.curstockindex -= 1
                self.loadnewiddata(self.stockids[self.curstockindex])
                self.refreshwholeplot()

    def on_scroll(self, event):
        if event.inaxes != self.ax1:
            return
        if event.button == 'down':
            scale_factor = 0.8
        if event.button == 'up':
            scale_factor = 1.2
        self.idx_range = int(self.idx_range * scale_factor)
        data_length = len(self.drawdata)
        if self.idx_range >= data_length - self.idx_start:
            self.idx_range = data_length - self.idx_start
        if self.idx_range <= 30:
            self.idx_range = 30
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.refresh_texts(self.drawdata.iloc[self.idx_start + self.idx_range])
        self.refresh_plot(self.idx_start, self.idx_range)

    def resetdrawdata(self):
        self.drawdata = self.df
        self.drawdatatype = 'd'

    def refreshwholeplot(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        # 更新图表里的文字内容
        self.refresh_texts(self.drawdata.iloc[self.idx_start + self.idx_range])
        self.refresh_plot(self.idx_start, self.idx_range)

    def on_press(self, event):
        # event.inaxes 可用于判断事件发生时，鼠标是否在某个Axes内
        # 程序指定，只有鼠标在ax1内，才能平移k线图，否则就退出事件处理函数
        if event.inaxes == self.ax3 and event.dblclick == 1:
            if self.indicator == 'macd':
                self.indicator = 'dma'
            elif self.indicator == 'dma':
                self.indicator = 'rsi'
            elif self.indicator == 'rsi':
                self.indicator = 'kdj'
            else:
                self.indicator = 'macd'
            self.refreshwholeplot()
            # 切换当前ma的类型，在ma，bb,none之间循环
        elif event.inaxes == self.ax1 and event.dblclick == 1:
            if self.avg_type == 'ma':
                self.avg_type = 'bb'
            elif self.avg_type == 'bb':
                self.avg_type = 'none'
            else:
                self.avg_type = 'ma'
            self.refreshwholeplot()
        else:
            if not event.inaxes == self.ax1:
                return
            # 检测是否按下了鼠标左键，如果不是，则退出
            if event.button != 1:
                return
            # 设置状态位press状态
            self.pressed = True
            # 记录按下位置的x坐标
            self.xpress = event.xdata
            # self.refreshwholeplot()



    def on_motion(self, event):
        # 如果鼠标没有按下，则什么事情都不需要做
        if not self.pressed:
            # normal move, just refresh context, show the data detail the cursor focus
            if not event.inaxes == self.ax1:
                return
            self.refresh_texts(self.drawdata.iloc[self.idx_start + int(event.xdata)])
            return
        if not event.inaxes == self.ax1:
            return
        dx = int(event.xdata - self.xpress)
        # 新的起点N(new) = N - dx
        new_start = self.idx_start - dx
        # 设定平移的结果，控制平移的范围不超过界限
        if new_start < 0:
            new_start = 0
        if new_start > len(self.drawdata) - self.idx_range - 1:
            new_start = len(self.drawdata) - self.idx_range - 1
        # 清除各个表格内的内容，为重绘做准备
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        # 更新图表里的文字内容
        self.idx_start = new_start
        self.refresh_texts(self.drawdata.iloc[self.idx_start + self.idx_range])
        self.refresh_plot(new_start, self.idx_range)


    def on_release(self, event):
        if not self.pressed:
            return
        self.pressed = False
        # 更新K线图的起点，否则下次拖拽的时候就不会从这次起点开始移动
        dx = int(event.xdata - self.xpress)
        self.idx_start -= dx
        if self.idx_start < 0:
            self.idx_start = 0
        if self.idx_start > len(self.drawdata) - self.idx_range - 1:
            self.idx_start = len(self.drawdata) - self.idx_range - 1

    def refresh_plot(self, idx_start, idx_range = 100):
        """ 根据最新的参数，重绘整个图表
        """
        all_data = self.drawdata
        plot_data = all_data.iloc[idx_start: idx_start + idx_range]
        # 添加均线到ax1
        ap3 = []
        if self.avg_type == 'ma':
            # 设置每条均线的颜色，因为不支持colors参数
            # 颜色示例 https://matplotlib.org/3.1.0/gallery/color/named_colors.html
            for [name, color] in [['ma5', 'white'], ['ma10', 'yellow'], ['ma20', 'fuchsia'], ['ma30', 'lime'], ['ma60', 'darkgrey'], ['ma120', 'royalblue'], ['ma250', 'lightskyblue']]:
                ap3.append(mpf.make_addplot(plot_data[[name]], ax=self.ax1, color=color))
                # ap3.append(mpf.make_addplot(plot_data[['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250']], ax=self.ax1))
        else:
            ap3.append(
                mpf.make_addplot(plot_data[['bb-u', 'bb-m', 'bb-l']], ax=self.ax1))
        # macd, 需要添加多个plot
        # 绘制快线和慢线
        if self.indicator == 'macd':
            ap3.append(mpf.make_addplot(plot_data[['macd-m', 'macd-s']], ax=self.ax3))
            # 根据柱状图绘制快线和慢线的差值，根据差值的大小，分别用红色和绿色填充
            # 红色的和绿色部分需要分别填充，因此先生成两组数据，分别包括大雨零和小雨等于零的数据
            bar_r = np.where(plot_data['macd-h'] > 0, plot_data['macd-h'], 0)
            bar_g = np.where(plot_data['macd-h'] <= 0, plot_data['macd-h'], 0)
            # 使用柱状图填充，设置颜色分别位红色和绿色
            ap3.append(mpf.make_addplot(bar_r, type='bar', color='red', ax=self.ax3))
            ap3.append(mpf.make_addplot(bar_g, type='bar', color='green', ax=self.ax3))
        elif self.indicator == 'rsi':
            ap3.append(mpf.make_addplot([75] * len(plot_data), color=(0.75, 0.6, 0.6), ax=self.ax3))
            ap3.append(mpf.make_addplot([30] * len(plot_data), color=(0.6, 0.75, 0.6), ax=self.ax3))
            ap3.append(mpf.make_addplot(plot_data['rsi'], ylabel='rsi', ax=self.ax3))
        elif self.indicator == 'dma':
            ap3.append(mpf.make_addplot(plot_data[['DIF', 'AMA']], ylabel='dema', ax=self.ax3))
        elif self.indicator == 'kdj':
            ap3.append(mpf.make_addplot(plot_data[['kdj_k', 'kdj_d', 'kdj_j']], ylabel='kdj', ax=self.ax3))
        mpl.rcParams['axes.prop_cycle'] = cycler(
            color=['white', 'gray', 'navy', 'teal', 'maroon', 'darkorange', 'indigo'])
        # tight_layout 紧密布局
        mpf.plot(plot_data, ax=self.ax1, volume=self.ax2, addplot=ap3, type='candle', style=my_style, xrotation=15, datetime_format='%Y-%m-%d', tight_layout=True)
        # 重绘十字
        self.cursor = Cursor(self.ax1, useblit=False, color='red', linewidth=1)
        # 快速刷新，否则如果不操作任何东西，永远也不会刷新
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.1)
        print('refresh plot complete')


    def refresh_texts(self, last_data):
        self.t1.set_text('%s - %s' % (self.id, self.name))
        self.t3.set_text(f'{np.round(last_data["open"], 3)}/{np.round(last_data["close"], 3)}')
        self.t5.set_text(f'{np.round(last_data["change"], 3)}')
        self.t7.set_text(f'{np.round(last_data["change"] / last_data["preclose"] * 100, 2)}%')
        self.t8.set_text(f'{last_data.name.date()}')
        self.t10.set_text(f'{last_data["high"]}')
        self.t12.set_text(f'{last_data["low"]}')
        self.t14.set_text(f'{np.round(last_data["volume"] / 1000000, 3)}')
        self.t16.set_text(f'{np.round(last_data["amount"] / 100000000, 3)}')
        if self.drawdatatype == 'd':
            self.t18.set_text(f'{np.round(last_data["upper_lim"], 3)}')
            self.t20.set_text(f'{np.round(last_data["lower_lim"], 3)}')
        else:
            self.t18.set_text('')
            self.t20.set_text('')
        self.t22.set_text(f'{last_data["preclose"]}')
        self.t24.set_text(f'{np.round(last_data["turn"], 2)}%')
        if last_data['change'] > 0:
            close_number_color = 'red'
        elif last_data['change'] < 0:
            close_number_color = 'green'
        else:
            close_number_color = 'gray'
        self.t3.set_color(close_number_color)
        self.t5.set_color(close_number_color)
        self.t7.set_color(close_number_color)


if __name__ == '__main__':
    ids = ['603518.SH','300193.SZ']
    # idfile = os.path.join(rootpath, id + '.csv')
    candle = InterCandle(ids)
    candle.refresh_texts(candle.df.iloc[candle.idx_start + candle.idx_range])
    candle.refresh_plot(candle.idx_start, candle.idx_range)
    plt.show()
