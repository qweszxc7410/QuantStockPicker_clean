
from sqlalchemy import create_engine
from datetime import datetime
import pandas as pd
import os
from backtesting import Strategy
from backtesting.lib import crossover
import talib as ta
import numpy as np
from datetime import time
import matplotlib.pyplot as plt
import glob
from varname import nameof
from backtesting import Backtest
from numpy import percentile
import warnings
from sqlalchemy import create_engine
import pandas as pd
import pymysql
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

load_dotenv()
USER = os.getenv("USER")
PW = os.getenv("PW")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
DB = os.getenv("DB")


def establish_db_connection():
    max_retries = 3
    retries = 0
    db_conn = None
    
    # Database connection parameters
    db_params = {'user': USER ,'pw': PW ,'host': HOST ,'port': PORT ,'db': DB }
    
    while retries < max_retries:
        try:
            db_conn = create_engine("mysql+pymysql://{user}:{pw}@{host}:{port}/{db}".format(**db_params)).connect()
            print("Database connection established successfully!")
            break  # Break out of the loop if the connection is successful
        except OperationalError as e:
            print(f"Error connecting to the database: {str(e)}!")
            retries += 1
            if retries < max_retries:
                print(f"Retrying... (Attempt {retries}/{max_retries})")
                time.sleep(3)
    
    if db_conn is None:
        print("Unable to establish a connection to the database after multiple attempts!")
    
    return db_conn


def get_data(db_conn,stock:str,start_date:str,end_date:str,extra_column:list(),rename_extra_column:list()): #start_date = '20130101',end_date = '20230931'
    """
    get the data from sql and save it as a csv file for a stock
    """
        
    if len(extra_column) == 0:
        _SQL = f'Select date,ticker,`開盤價_後復權`,`最高價_後復權`,`最低價_後復權`,`收盤價_後復權`,`成交量` from indistockdb_5.{stock} where `date` >= "{start_date}" and `date` <= "{end_date}" order by `date` asc'
    else:
        extra_column_str = ', '.join([f'`{col}`' for col in extra_column])
        _SQL = f'Select date,ticker,`開盤價_後復權`,`最高價_後復權`,`最低價_後復權`,`收盤價_後復權`,`成交量`,{extra_column_str} from indistockdb_5.{stock} where `date` >= "{start_date}" and `date` <= "{end_date}" order by `date` asc'
    
    if db_conn: 
        df = pd.read_sql_query(_SQL,db_conn)#, index_col='date'
        new_column_names = ["Date","ticker","Open","High","Low","Close","Volume"] + rename_extra_column
        column_mapping = {old_col: new_col for old_col, new_col in zip(df.columns, new_column_names)}
        df = df.rename(columns=column_mapping)
        #df.insert(0,"Stock",stock)
        df.Date = pd.to_datetime(df.Date)
        df = df.set_index('Date')
        
    return df

def sort_data(df:pd.DataFrame()): #start_date = '20130101',end_date = '20230931'

    columns_list = list(df.columns)
    columns_list = [column for column in columns_list if column not in ['Date', 'ticker']]
    dict_df = {column: pd.DataFrame() for column in columns_list}
    
    df1 = df.reset_index(drop = False)
    for col in columns_list:
        col_data = df1.pivot(index='Date', columns='ticker', values=col)
        dict_df[col] = pd.concat([dict_df[col],col_data],axis = 1)
    


if __name__ == "__main__":
    start_date0 = '20130101'
    end_date0 = '20230931' 
    save_path = os.path.join(os.path.normpath(os.getcwd()),"data","measure_data")
    db_conn = establish_db_connection()    
    stock0050 = [2330,2454,2317,2308,2303,2382,2881,2891,2412,2886,2882,3711,1303,2884,1216,2885,1301,2002,2892,3231,2345,5880,3034,2357,5871,2301,1326,2887,2890,2880,2327,3008,3037,1101,2207,2379,2883,3045,1590,5876,4938,2395,2912,6669,4904,2801,6505,2603,9910,2408]
    stock0050 = [str(stock) for stock in stock0050]
    # stock0050 = stock0050[0:10] 
    extra_column0 = ['漲幅','K9','D9','公告基本每股盈餘', '單月營收', '流通在外股數', '股價淨值比', '權益總計', '綜合損益', '營業毛利', '營業收入淨額','外資買賣超','自營商買賣超','投信買賣超','本益比4', '單月營收年成長率', '稅前純益', '負債總計', '資產總計','MACD','正DI14','負DI14','ADX14']
    rename_extra_column0 = ['漲幅','K9','D9','公告基本每股盈餘', '單月營收', '流通在外股數', '股價淨值比', '權益總計', '綜合損益', '營業毛利', '營業收入淨額','外資買賣超','自營商買賣超','投信買賣超','本益比4', '單月營收年成長率', '稅前純益', '負債總計', '資產總計','MACD','正DI14','負DI14','ADX14']
    #threshold_names_list = ['PE_Ratio_threshold','Mon_Rev_YoY2_threshold','EBT_YoY5_threshold','Debt_Ratio_threshold']
    
    warnings.filterwarnings("ignore")
    
    for stock in stock0050:
        stock_data = get_data(db_conn,stock,start_date0,end_date0,extra_column0,rename_extra_column0)

        columns_list = list(stock_data.columns)
        columns_list = [column for column in columns_list if column not in ['date', 'ticker']]
        if 'dict_df' not in globals(): # Create 'dict1' as an empty dictionary
            dict_df = {column: pd.DataFrame() for column in columns_list}
        
        stock_data1 = stock_data.reset_index(drop = False)
        for col in columns_list:
            col_data = stock_data1.pivot(index='Date', columns='ticker', values=col)
            dict_df[col] = pd.concat([dict_df[col],col_data],axis = 1)        

    for col in dict_df:
        dict_df[col].reset_index(drop = False).to_feather(os.path.join(save_path,f"{col}.feather"))
            
    if db_conn:
        db_conn.close()
        print("Database connection closed!")
        
