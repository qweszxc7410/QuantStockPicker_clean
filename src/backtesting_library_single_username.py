import pandas as pd
import os
import feather
from backtesting import Backtest
from backtesting import Strategy
from time import perf_counter
import backtesting as bt

################ Class是標準格式 ################


class BuyOne_SellOne(Strategy):

    Describe = "有訊號就買"
    MathDescribe = "df == 1 "
    DataSetting = {'Version': 5, 'Column': ['Open', 'High', 'Low', 'Close', 'MACD', 'MACDsignal', 'MACDhist']}
    
    def init(self):...

    def next(self):
        if self.position.size == 0:  # 進場條件:沒有持倉
            if self.data.In_Signal[-1] == 1:  
                self.buy()

        elif self.position.size != 0:  # 出場
            if self.data.Out_Signal[-1] == 1:  
                self.position.close()
class BuyOne_SellUnderZero(Strategy):

    Describe = "有訊號就買"
    MathDescribe = "df == 1 "
    DataSetting = {'Version': 5, 'Column': ['Open', 'High', 'Low', 'Close', 'MACD', 'MACDsignal', 'MACDhist']}
    
    def init(self):...

    def next(self):
        if self.position.size == 0:  # 進場條件:沒有持倉
            if self.data.In_Signal[-1] == 1:  
                self.buy()

        elif self.position.size != 0:  # 出場
            if self.data.Out_Signal[-1]<= 0:  
                self.position.close()
class BuyOne_SellZero(Strategy):

    Describe = "有訊號就買"
    MathDescribe = "df == 1 "
    DataSetting = {'Version': 5, 'Column': ['Open', 'High', 'Low', 'Close', 'MACD', 'MACDsignal', 'MACDhist']}
    
    def init(self):...

    def next(self):
        if self.position.size == 0:  # 進場條件:沒有持倉
            if self.data.In_Signal[-1] == 1:  
                self.buy()

        elif self.position.size != 0:  # 出場
            if self.data.Out_Signal[-1] == 0:  
                self.position.close()

class BuyZero_SellOne(Strategy):

    Describe = "有訊號就買"
    MathDescribe = "df == 1 "
    DataSetting = {'Version': 5, 'Column': ['Open', 'High', 'Low', 'Close', 'MACD', 'MACDsignal', 'MACDhist']}
    
    def init(self):...

    def next(self):
        if self.position.size == 0:  # 進場條件:沒有持倉
            if self.data.In_Signal[-1] == 0:  
                self.buy()

        elif self.position.size != 0:  # 出場
            if self.data.Out_Signal[-1] == 1:  
                self.position.close()
         
class BuyOne_Hold20(Strategy):
    
    Describe = "買點持有20日就出"
    MathDescribe = "df == 1 & Max_hold_period = 5"
    DataSetting = {'Version': 5, 'Column': ['Open', 'High', 'Low', 'Close', 'MACD', 'MACDsignal', 'MACDhist']}
    Max_hold_period = 20
    
    def init(self):...

    def next(self):
        if self.position.size == 0:  # 進場條件:沒有持倉
            self.hold_peiod = 0
            if self.data.In_Signal[-1] == 1:  
                self.buy()

        elif self.position.size != 0:  # 出場
            self.hold_peiod = self.hold_peiod + 1
            if self.hold_peiod == self.Max_hold_period:
                self.position.close()
                self.hold_peiod = 0

class BuyOne_Hold60(Strategy):
    
    Describe = "買點持有60日就出"
    MathDescribe = "df == 1 & Max_hold_period = 60"
    DataSetting = {'Version': 5, 'Column': ['Open', 'High', 'Low', 'Close', 'MACD', 'MACDsignal', 'MACDhist']}
    Max_hold_period = 60
    
    def init(self):...

    def next(self):
        if self.position.size == 0:  # 進場條件:沒有持倉
            self.hold_peiod = 0
            if self.data.In_Signal[-1] == 1:  
                self.buy()

        elif self.position.size != 0:  # 出場
            self.hold_peiod = self.hold_peiod + 1
            if self.hold_peiod == self.Max_hold_period:
                self.position.close()
                self.hold_peiod = 0

class BuyOne_Hold120(Strategy):
    
    Describe = "買點持有120日就出"
    MathDescribe = "df == 1 & Max_hold_period = 120"
    DataSetting = {'Version': 5, 'Column': ['Open', 'High', 'Low', 'Close', 'MACD', 'MACDsignal', 'MACDhist']}
    Max_hold_period = 120
    
    def init(self):...

    def next(self):
        if self.position.size == 0:  # 進場條件:沒有持倉
            self.hold_peiod = 0
            if self.data.In_Signal[-1] == 1:  
                self.buy()

        elif self.position.size != 0:  # 出場
            self.hold_peiod = self.hold_peiod + 1
            if self.hold_peiod == self.Max_hold_period:
                self.position.close()
                self.hold_peiod = 0   
                
class BuyOne_Hold252(Strategy):
    
    Describe = "買點持有252日就出"
    MathDescribe = "df == 1 & Max_hold_period = 252"
    DataSetting = {'Version': 5, 'Column': ['Open', 'High', 'Low', 'Close', 'MACD', 'MACDsignal', 'MACDhist']}
    Max_hold_period = 252
    
    def init(self):...

    def next(self):
        if self.position.size == 0:  # 進場條件:沒有持倉
            self.hold_peiod = 0
            if self.data.In_Signal[-1] == 1:  
                self.buy()

        elif self.position.size != 0:  # 出場
            self.hold_peiod = self.hold_peiod + 1
            if self.hold_peiod == self.Max_hold_period:
                self.position.close()
                self.hold_peiod = 0   

class SellOne_Hold20(Strategy):
    
    Describe = "賣點持有20日就出"
    MathDescribe = "df == 1 & Max_hold_period = 5"
    DataSetting = {'Version': 5, 'Column': ['Open', 'High', 'Low', 'Close', 'MACD', 'MACDsignal', 'MACDhist']}
    Max_hold_period = 20
    
    def init(self):...

    def next(self):
        if self.position.size == 0:  # 進場條件:沒有持倉
            self.hold_peiod = 0
            if self.data.In_Signal[-1] == 1:  
                self.sell()

        elif self.position.size != 0:  # 出場
            self.hold_peiod = self.hold_peiod + 1
            if self.hold_peiod == self.Max_hold_period:
                self.position.close()
                self.hold_peiod = 0

class SellOne_Hold60(Strategy):
    
    Describe = "賣點持有60日就出"
    MathDescribe = "df == 1 & Max_hold_period = 60"
    DataSetting = {'Version': 5, 'Column': ['Open', 'High', 'Low', 'Close', 'MACD', 'MACDsignal', 'MACDhist']}
    Max_hold_period = 60
    
    def init(self):...

    def next(self):
        if self.position.size == 0:  # 進場條件:沒有持倉
            self.hold_peiod = 0
            if self.data.In_Signal[-1] == 1:  
                self.sell()

        elif self.position.size != 0:  # 出場
            self.hold_peiod = self.hold_peiod + 1
            if self.hold_peiod == self.Max_hold_period:
                self.position.close()
                self.hold_peiod = 0

class SellOne_Hold120(Strategy):
    
    Describe = "賣點持有120日就出"
    MathDescribe = "df == 1 & Max_hold_period = 120"
    DataSetting = {'Version': 5, 'Column': ['Open', 'High', 'Low', 'Close', 'MACD', 'MACDsignal', 'MACDhist']}
    Max_hold_period = 120
    
    def init(self):...

    def next(self):

        if self.position.size == 0:  # 進場條件:沒有持倉
            self.hold_peiod = 0
            if self.data.In_Signal[-1] == 1:  
                self.sell()

        elif self.position.size != 0:  # 出場
            self.hold_peiod = self.hold_peiod + 1
            if self.hold_peiod == self.Max_hold_period:
                self.position.close()
                self.hold_peiod = 0   
                
class SellOne_Hold252(Strategy):
    
    Describe = "賣點持有252日就出"
    MathDescribe = "df == 1 & Max_hold_period = 252"
    DataSetting = {'Version': 5, 'Column': ['Open', 'High', 'Low', 'Close', 'MACD', 'MACDsignal', 'MACDhist']}
    Max_hold_period = 252
    
    def init(self):...

    def next(self):
        if self.position.size == 0:  # 進場條件:沒有持倉
            self.hold_peiod = 0
            if self.data.In_Signal[-1] == 1:  
                self.buy()

        elif self.position.size != 0:  # 出場
            self.hold_peiod = self.hold_peiod + 1
            if self.hold_peiod == self.Max_hold_period:
                self.position.close()
                self.hold_peiod = 0      
                
if __name__ == '__main__':
    ################ 準備資料 ################
    TopPath = os.path.normpath(os.getcwd())
    Open = pd.read_feather(os.path.join(os.path.normpath(os.getcwd()),'data','measure_data','Open.feather'))
    High = pd.read_feather(os.path.join(os.path.normpath(os.getcwd()),'data','measure_data','High.feather'))
    Low = pd.read_feather(os.path.join(os.path.normpath(os.getcwd()),'data','measure_data','Low.feather'))
    Close = pd.read_feather(os.path.join(os.path.normpath(os.getcwd()),'data','measure_data','Close.feather'))
    signal_1 = pd.read_feather(os.path.join(TopPath,'data','rule_data','MACD向上交叉零線.feather'))
    signal_2 = pd.read_feather(os.path.join(TopPath,'data','rule_data','MACD向下交叉零線.feather'))


    df = pd.DataFrame()
    Ticker = '1101'
    df['Date'] = signal_1['Date']
    df['Open'] = Open[Ticker]
    df['High'] = High[Ticker]
    df['Low'] = Low[Ticker]
    df['Close'] = Close[Ticker]
    df['In_Signal'] = signal_1[Ticker]
    df['Out_Signal'] = signal_2[Ticker]

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    
    if 1:# 回測

        start_time = perf_counter()
        bt = Backtest(df, SellOne_Hold252, cash = 10000_000, commission = 0.003, exclusive_orders = True, hedging=False,) #margin=0.05
        bt_stats = bt.run()
        print(bt_stats)
        print(f"執行時間 {perf_counter() - start_time}")

