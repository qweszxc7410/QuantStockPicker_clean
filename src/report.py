import os
import pandas as pd

# 要合併的檔案目錄路徑
def combine_backtesting_report_by_rule(data_path:str ,save_path:str ,strategy:str, rule:str):
    ''' 存檔的csv 不包含'_equity_curve', '_trades' 這兩個欄位 '''
    directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'output','backtesting', strategy, rule)
    stocks_perform = pd.DataFrame()
    # 讀取目錄中的所有 CSV 檔案
    for filename in os.listdir(directory): 
        filepath = os.path.join(directory, filename)
        # 讀取 CSV 檔案並將其添加到合併的 DataFrame 中
        if os.path.isfile(filepath) and filename.endswith('.csv'):
            df = pd.read_csv(filepath, encoding = "utf-8-sig")
            df.columns = ['Title', os.path.splitext(filename)[0]]            # 重設column名稱
            stocks_perform = pd.concat([stocks_perform, df], axis=1, ignore_index=False)

    stocks_perform = stocks_perform.loc[:, ~stocks_perform.columns.duplicated()].set_index('Title')
    stocks_perform = stocks_perform.T.reset_index()
    stocks_perform.rename(columns={'index':'Ticker'}, inplace=True)
    filtered_stocks_perform = stocks_perform.drop(columns=['_equity_curve', '_trades'])
    filtered_stocks_perform['Rule'] = rule
    filtered_stocks_perform.to_csv(os.path.join(save_path, f"{strategy}_{rule}.csv"),encoding = "utf-8-sig")

def combine_big_backtesting_report_by_ticker_by_rule(data_path:str ,save_path:str):
    ''' 存檔的csv 不包含'_equity_curve', '_trades' 這兩個欄位 '''
    directory = os.path.join(os.path.normpath(data_path))

    stocks_perform = pd.DataFrame()
    # 讀取目錄中的所有 CSV 檔案
    for filename in os.listdir(directory): 
        filepath = os.path.join(directory, filename)
        # 讀取 CSV 檔案並將其添加到合併的 DataFrame 中
        if os.path.isfile(filepath) and filename.endswith('.csv'):
            
            df = pd.read_csv(filepath, encoding = "utf-8-sig", index_col=False)
            print()
            print("-"*50)
            print(f"Processing file: {filename} | {directory}")
            print(f"filename {filename} | Total Return ann. median {df['Return (Ann.) [%]'].median():.4f} | Total Return median {df['Return [%]'].median():.4f} | Sharpe Ration {df['Sharpe Ratio'].median():.4f} | # Trades {df['# Trades'].sum()}")
            print(f"filename {filename} | Total Return ann. sum {df['Return (Ann.) [%]'].sum():.4f} | Total Return sum {df['Return [%]'].sum():.4f} | Sharpe Ration {df['Sharpe Ratio'].mean():.4f} | # Trades {df['# Trades'].sum()}")            
            df = df.drop(columns=['Unnamed: 0'])
            stocks_perform = pd.concat([stocks_perform, df], axis=0, ignore_index=True)
    stocks_perform.to_csv(os.path.join(os.path.dirname(data_path), "bigtable.csv"),encoding = "utf-8-sig")
    print(f"合併後的資料 (大表) 已儲存到 {os.path.join(save_path, 'bigtable.csv')}")
    
if __name__ == '__main__':
    
    if 1:
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'output')
        report_path = os.path.join(base_path, 'report',"StrategyAndRule")
        if not os.path.exists(report_path):
            os.mkdir(report_path)
        
        data_path = os.path.join(base_path, 'backtesting')
        
        # 不同策略手動新增
        if 1:
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_Hold252', rule ='Rule_6')
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_Hold120', rule ='Rule_6')

        if 0:
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_Hold20', rule ='Rule_001_1')
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_Hold60', rule ='Rule_001_1')
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_Hold120', rule ='Rule_001_1')
        if 0:
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellZero', rule ='Rule_001_1')
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellZero', rule ='Rule_001_2')
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellZero', rule ='Rule_001_3')
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellZero', rule ='Rule_001_5')
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellZero', rule ='Rule_001_6')
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellZero', rule ='Rule_001_7')

            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyZero_SellOne', rule ='Rule_001_1')
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyZero_SellOne', rule ='Rule_001_2')
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyZero_SellOne', rule ='Rule_001_3')
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyZero_SellOne', rule ='Rule_001_5')
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyZero_SellOne', rule ='Rule_001_6')
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyZero_SellOne', rule ='Rule_001_7')
        if 0:
            # combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellZero', rule ='Rule_001_1')
            # combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellZero', rule ='Rule_001_2')
            # combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellZero', rule ='Rule_001_3')
            # combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellZero', rule ='Rule_001_8')
            # combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellZero', rule ='Rule_001_9')
            # combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellZero', rule ='Rule_001_10')
            # combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellZero', rule ='TPB_1')
            # combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellZero', rule ='TPB_2')
            # combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellZero', rule ='TPB_3')
            # combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellZero', rule ='Rule_001_11')
            # combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellZero', rule ='Rule_001_12')
            # combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellZero', rule ='Rule_001_13')
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellOne', rule ='Rule_102_1')
            # combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellOne', rule ='Rule_101_2')
            # combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyPositiveThree_SellNegativeThree', rule ='Rule_101_3')
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'BuyOne_SellOne', rule ='Rule_102_3')
            combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = 'SellOne_Hold252', rule ='Rule_5')

    if 1:
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'output')
        data_path = os.path.join(base_path, 'report','StrategyAndRule')
        report_path = os.path.join(base_path, 'report')
        
        combine_big_backtesting_report_by_ticker_by_rule(data_path = data_path , save_path = report_path)


