from time import perf_counter
from backtesting import Backtest
import numpy as np 
import backtesting_library_single_username as BacktestingLibrary
import os
import pandas as pd
import rule_library_username as RuleLibrary
import re
from varname import nameof

class SignalData:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        
def run_by_ticker(ticker_list:list,BacktestingStrategy:str, signal_name:str, in_signal_df = None, out_signal_df = None,Open = None,High = None,Low = None,Close = None):    
    TopPath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'output','backtesting')
    directory_path = os.path.join(TopPath, BacktestingStrategy)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    subdirectory_path = os.path.join(directory_path, signal_name)
    if not os.path.exists(subdirectory_path):
        os.makedirs(subdirectory_path)
        
    for ticker in ticker_list:
        df = pd.DataFrame()
        df['Date'] = Open['Date']
        df['Open'] = Open[ticker]
        df['High'] = High[ticker]
        df['Low'] = Low[ticker]
        df['Close'] = Close[ticker]
        df_filled = df.ffill().replace(np.nan, 0)

        if in_signal_df is not None:
            in_signal_df.index.name = 'Date'
            in_signal_df.index = pd.to_datetime(in_signal_df['Date'])
        if out_signal_df is not None:
            out_signal_df.index.name = 'Date'
            out_signal_df.index = pd.to_datetime(out_signal_df['Date'])
        df_filled['Date'] = pd.to_datetime(df_filled['Date'])
        df_filled.set_index('Date', inplace=True)

        # 使用 pd.concat 合併 DataFrame，保持索引對齊
        if in_signal_df is not None:
            df_filled['In_Signal'] = in_signal_df[ticker].reindex(df_filled.index)
        if out_signal_df is not None:
            df_filled['Out_Signal'] = out_signal_df[ticker].reindex(df_filled.index)


    ##############
        start_time = perf_counter()
        bt = Backtest(df_filled, eval(f"BacktestingLibrary.{BacktestingStrategy}"), cash = 10000_000, commission = 0.003, exclusive_orders = True, hedging=False,) #margin=0.05
        try:
            bt_stats = bt.run()
            print(f"{ticker} 執行時間 {(perf_counter() - start_time):.2f} | Save {os.path.join(subdirectory_path,f'{ticker}.csv')}")
            bt_stats.to_csv(os.path.join(subdirectory_path,f"{ticker}.csv"))
        except Exception as e:
            # Handle the exception and print the error message
            print(f"An error occurred: {e}")
            
if __name__ == '__main__':

    if 1:
        measure_data_path = os.path.join(os.path.normpath(os.getcwd()),'data','measure_data')
        rule_data_path = os.path.join(os.path.normpath(os.getcwd()),'data','rule_data')
        Open = pd.read_feather(os.path.join(measure_data_path,'Open.feather'))
        High = pd.read_feather(os.path.join(measure_data_path,'High.feather'))
        Low = pd.read_feather(os.path.join(measure_data_path,'Low.feather'))
        Close = pd.read_feather(os.path.join(measure_data_path,'Close.feather'))
        Ticker_list = [col for col in list(Close.columns) if col != "Date"]
        
    if 1:
        
        base_path = os.path.dirname(os.path.abspath(__file__))  # 取得目前檔案的路徑
        rule_path = os.path.join(os.path.dirname(base_path),'data','rule_data')
        measure_path = os.path.join(os.path.dirname(base_path),'data','measure_data')

        Open = pd.read_feather(os.path.join(measure_path,'Open.feather'))
        build_rule_list = ["Rule_6","Rule_7"]
        for build_id in build_rule_list:
            print(f"建立 {build_id} 資料")
            file_name_list = RuleLibrary.RuleLibrary_V1(build_id).getCrossSectionDataFileName()
            
            for idx,name in enumerate(file_name_list):
                
                globals()[name] = pd.read_feather(os.path.join(measure_path,name + '.feather'))
                run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyOne_Hold120",signal_name = "Rule_6",in_signal_df = globals()[name],Open = Open,High = High,Low = Low,Close = Close)  
                run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyOne_Hold252",signal_name = "Rule_6",in_signal_df = globals()[name],Open = Open,High = High,Low = Low,Close = Close)  
                


        print()
        
        # Rule_001_1 = pd.read_feather(os.path.join(rule_data_path,'本益比小於25.feather')) # Rule_7
        # Rule_001_2 = pd.read_feather(os.path.join(rule_data_path,'股價跌出布林通道下軌.feather')) # Rule_6

      
        # run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "SellOne_Hold252",signal_name = "Rule_001_2",in_signal_df = Rule_001_2,Open = Open,High = High,Low = Low,Close = Close)
        # run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyOne_Hold20",signal_name = "Rule_001_1",in_signal_df = Rule_001_1,Open = Open,High = High,Low = Low,Close = Close)
        # run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyOne_Hold60",signal_name = "Rule_001_1",in_signal_df = Rule_001_1,Open = Open,High = High,Low = Low,Close = Close)        
        # run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyOne_Hold252",signal_name = "Rule_001_1",in_signal_df = Rule_001_1,Open = Open,High = High,Low = Low,Close = Close)
        # run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "SellOne_Hold20",signal_name = "Rule_001_2",in_signal_df = Rule_001_2,Open = Open,High = High,Low = Low,Close = Close)
        # run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "SellOne_Hold60",signal_name = "Rule_001_2",in_signal_df = Rule_001_2,Open = Open,High = High,Low = Low,Close = Close)        
        # run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "SellOne_Hold120",signal_name = "Rule_001_2",in_signal_df = Rule_001_2,Open = Open,High = High,Low = Low,Close = Close)        
        # run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "SellOne_Hold252",signal_name = "Rule_5",in_signal_df = Rule_001_2,Open = Open,High = High,Low = Low,Close = Close)

    if 0:
        Rule_001_1 = pd.read_feather(os.path.join(rule_data_path,'本益比小於25.feather'))
        Rule_001_2 = pd.read_feather(os.path.join(rule_data_path,'股價跌出布林通道下軌.feather'))
        Rule_001_3 = pd.read_feather(os.path.join(rule_data_path,'近5年稅前淨利成長率平均大於0_03.feather'))
        Rule_001_1.iloc[:,1:] = Rule_001_1.iloc[:,1:] * 1
        Rule_001_4 = Rule_001_2.copy()
        for col in Rule_001_2.columns:
            if col != 'Date':
                Rule_001_4[col] = Rule_001_2[col] | Rule_001_3[col]  # 使用按位或操作以合併布爾值
        
        Rule_001_5 = Rule_001_2.copy()
        for col in Rule_001_2.columns:
            if col != 'Date':
                Rule_001_5[col] = Rule_001_2[col] - Rule_001_1[col] - Rule_001_3[col]  # 使用按位或操作以合併布爾值
                if min(Rule_001_5[col])<0:
                    print("")
                Rule_001_5[col] = (Rule_001_5[col] > 0 )*1
        Rule_001_6 = Rule_001_2.copy()
        for col in Rule_001_2.columns:
            if col != 'Date':
                Rule_001_6[col] = Rule_001_2[col] + Rule_001_3[col] - Rule_001_1[col]  # 使用按位或操作以合併布爾值
                Rule_001_6[col] = (Rule_001_6[col] > 0 )*1
        Rule_001_7 = Rule_001_2.copy()
        for col in Rule_001_2.columns:
            if col != 'Date':
                Rule_001_7[col] = Rule_001_2[col] + Rule_001_3[col] - Rule_001_1[col]  # 使用按位或操作以合併布爾值
                Rule_001_7[col] = (Rule_001_7[col] > 1 )*1     
        run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyOne_SellZero",signal_name = "Rule_001_1",in_signal_df = Rule_001_1,out_signal_df = Rule_001_1,Open = Open,High = High,Low = Low,Close = Close)
        run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyOne_SellZero",signal_name = "Rule_001_2",in_signal_df = Rule_001_2,out_signal_df = Rule_001_2,Open = Open,High = High,Low = Low,Close = Close)
        run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyOne_SellZero",signal_name = "Rule_001_3",in_signal_df = Rule_001_3,out_signal_df = Rule_001_3,Open = Open,High = High,Low = Low,Close = Close)
        run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyOne_SellZero",signal_name = "Rule_001_5",in_signal_df = Rule_001_5,out_signal_df = Rule_001_5,Open = Open,High = High,Low = Low,Close = Close)
        run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyOne_SellZero",signal_name = "Rule_001_6",in_signal_df = Rule_001_6,out_signal_df = Rule_001_5,Open = Open,High = High,Low = Low,Close = Close)
        run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyOne_SellZero",signal_name = "Rule_001_7",in_signal_df = Rule_001_7,out_signal_df = Rule_001_7,Open = Open,High = High,Low = Low,Close = Close)

        run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyZero_SellOne",signal_name = "Rule_001_1",in_signal_df = Rule_001_1,out_signal_df = Rule_001_1,Open = Open,High = High,Low = Low,Close = Close)
        run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyZero_SellOne",signal_name = "Rule_001_2",in_signal_df = Rule_001_2,out_signal_df = Rule_001_2,Open = Open,High = High,Low = Low,Close = Close)
        run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyZero_SellOne",signal_name = "Rule_001_3",in_signal_df = Rule_001_3,out_signal_df = Rule_001_3,Open = Open,High = High,Low = Low,Close = Close)
        run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyZero_SellOne",signal_name = "Rule_001_5",in_signal_df = Rule_001_5,out_signal_df = Rule_001_5,Open = Open,High = High,Low = Low,Close = Close)
        run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyZero_SellOne",signal_name = "Rule_001_6",in_signal_df = Rule_001_6,out_signal_df = Rule_001_6,Open = Open,High = High,Low = Low,Close = Close)
        run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyZero_SellOne",signal_name = "Rule_001_7",in_signal_df = Rule_001_7,out_signal_df = Rule_001_7,Open = Open,High = High,Low = Low,Close = Close)
    
    if 0:
        Rule_001_1 = pd.read_feather(os.path.join(rule_data_path,'本益比小於25.feather'))
        Rule_001_2 = pd.read_feather(os.path.join(rule_data_path,'股價跌出布林通道下軌.feather'))
        Rule_001_3 = pd.read_feather(os.path.join(rule_data_path,'近5年稅前淨利成長率平均大於0_03.feather'))
        Rule_001_1.iloc[:,1:] = Rule_001_1.iloc[:,1:] * 1
        Rule_001_8 = Rule_001_2.copy()
        for col in Rule_001_2.columns:
            if col != 'Date':
                Rule_001_8[col] = 0.3581*Rule_001_1[col] + 1.4659*Rule_001_2[col]   + (-0.8240)*Rule_001_3[col] # 使用按位或操作以合併布爾值
        threshold = 1.324
        Rule_001_8 = (((Rule_001_8.iloc[:,1:]) < threshold * (Rule_001_8.iloc[:,1:]))  + (Rule_001_8.iloc[:,1:]) >= threshold)*1
        run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyOne_SellZero",signal_name = "Rule_001_8",in_signal_df = Rule_001_8,out_signal_df = Rule_001_8,Open = Open,High = High,Low = Low,Close = Close)
    
    if 0:
        Rule_001_1 = pd.read_feather(os.path.join(rule_data_path,'本益比小於25.feather'))
        Rule_001_2 = pd.read_feather(os.path.join(rule_data_path,'股價跌出布林通道下軌.feather'))
        Rule_001_3 = pd.read_feather(os.path.join(rule_data_path,'近5年稅前淨利成長率平均大於0_03.feather'))
        Rule_001_1.iloc[:,1:] = Rule_001_1.iloc[:,1:] * 1
        Rule_001_9 = Rule_001_2.copy()
        for col in Rule_001_2.columns:
            if col != 'Date':
                Rule_001_9[col] = 0.3581*Rule_001_1[col] + 1.4659*Rule_001_2[col]   + (-0.8240)*Rule_001_3[col] # 使用按位或操作以合併布爾值
        threshold = 0.5
        Rule_001_9 = (((Rule_001_9.iloc[:,1:]) < threshold * (Rule_001_9.iloc[:,1:]))  + (Rule_001_9.iloc[:,1:]) >= threshold)*1
        run_by_ticker(ticker_list = Ticker_list,BacktestingStrategy = "BuyOne_SellZero",signal_name = "Rule_001_9 ",in_signal_df = Rule_001_9,out_signal_df = Rule_001_9,Open = Open,High = High,Low = Low,Close = Close)



