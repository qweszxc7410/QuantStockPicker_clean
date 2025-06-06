import pandas as pd
import os
import talib as ta
import numpy as np
from varname import nameof
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning) # 關閉 FutureWarning 警示


def saveProcessVariable(df:pd.DataFrame,df_name:str):
    df.insert(0,"Date",date)
    df.to_feather(os.path.join(os.path.normpath(os.getcwd()),'data',"measure_data",f"{df_name}.feather"))
    
    
class MeasureLibrary():
    def __init__(self,ProcessName = ""):

        self.UseData = ""
        self.df = ""
        self.ProcessName = ProcessName
        self.ProcessVariable = ""
        self.UseMethod = "CrossSection"
        if ProcessName != "": #送進空的物件仍可使用下列func.
            self.StandardProcess()
            

    def StandardProcess(self):
        try:
            eval('self.'+ self.ProcessName + "()")
        except:
            print("[Error] [Rule_Library] 沒有建置 : " + self.ProcessName )

    def getCrossSectionData(self):
        # 如果變數存在
        if hasattr(self,'CrossSectionData'):
            return self.CrossSectionData
        else:
            return []

    def getRuleDataframe(self):
        return self.df

    def getRulelist(self):
        return [method for method in dir(eval(type(self).__qualname__)) if method.startswith('__') is False]

    def getUseData (self):
        return self.UseData
    
    def getRuleCommand(self):
        return  self.RuleCommand
    
    def getProcessVariable(self):
        return  self.ProcessVariable

    def getUseData_ByCrossSectionData (self):
        return self.UseData

    def getCrossSectionDataFileName(self):
        return self.CrossSectionDataFileName

    def getRuleDescription(self):
        return self.Description

    def Measure_1(self):

        self.Description = " EMA(200) "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['Close']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['Close']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData =  ['Close']

            self.ProcessVariable = "Close.apply(lambda close: ta.EMA(close.fillna(method='ffill'),200))"   

    def Measure_2(self):

        self.Description = " MACD: Signal "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['MACD']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['MACD']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData =  ['MACD']

            self.ProcessVariable = "MACD.apply(lambda close: ta.EMA(close.fillna(method='ffill'),9))" 
    
    def Measure_3(self):

        self.Description = " MACD(12,26,9): MACD, Signal, Histogram "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['MACD','MACD_Signal']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['MACD','MACD_Signal']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData =  ['MACD','MACD_Signal']

            self.ProcessVariable = "Close.apply(lambda close: ta.MACD(close.fillna(method='ffill'),12,26,9))"  
           
    def Measure_4(self):

        self.Description = " 過去60根K棒50百分位數的量 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['Volume']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['Volume']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData =  ['Volume']

            self.ProcessVariable = "Volume.rolling(60).quantile(.5, interpolation='lower')"  

    def Measure_5(self):

        self.Description = " Bollinger Bands(20,2): upper_band, middle_band, lower_band "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['Close']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['Close']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData =  ['Close']

            self.ProcessVariable = "Close.apply(lambda close: ta.BBANDS(close.fillna(method='ffill'),20,2))"  

    def Measure_6(self):

        self.Description = " RSI(14) "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['Close']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['Close']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData =  ['Close']

            self.ProcessVariable = "Close.apply(lambda close: ta.RSI(close.fillna(method='ffill'),14))"  
           
    def Measure_7(self):

        self.Description = " 三大法人買賣超總和 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['外資買賣超','自營商買賣超','投信買賣超']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['外資買賣超','自營商買賣超','投信買賣超']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData =  ['外資買賣超','自營商買賣超','投信買賣超']

            self.ProcessVariable = "外資買賣超 + 自營商買賣超 + 投信買賣超"    
            
    def Measure_8(self):

        self.Description = " SMA5 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['Close']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['Close']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData =  ['Close']

            self.ProcessVariable = "Close.apply(lambda close: ta.SMA(close.fillna(method='ffill'),5))"  
            
    def Measure_9(self):

        self.Description = " SMA10 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['Close']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['Close']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData =  ['Close']

            self.ProcessVariable = "Close.apply(lambda close: ta.SMA(close.fillna(method='ffill'),10))"  
            
    def Measure_10(self):

        self.Description = " SMA20 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['Close']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['Close']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData =  ['Close']

            self.ProcessVariable = "Close.apply(lambda close: ta.SMA(close.fillna(method='ffill'),20))"  
            
    def Measure_11(self):

        self.Description = " SMA60 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['Close']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['Close']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData =  ['Close']

            self.ProcessVariable = "Close.apply(lambda close: ta.SMA(close.fillna(method='ffill'),60))"  
 
    def Measure_12(self):

        self.Description = " Average Volume 10 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['Volume']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['Volume']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData =  ['Volume']

            self.ProcessVariable = "Volume.apply(lambda volume: ta.SMA(volume.fillna(method='ffill'),10))" 
    
    def Measure_13(self):

        self.Description = " Average Volume 30 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['Volume']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['Volume']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData =  ['Volume']

            self.ProcessVariable = "Volume.apply(lambda volume: ta.SMA(volume.fillna(method='ffill'),30))" 
    
    def Measure_14(self):

        self.Description = " J 值 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['K9','D9']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['K9','D9']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['K9','D9']

            self.ProcessVariable = "3 * K9 - 2 * D9"    
    
    def Measure_15(self):

        self.Description = " 負債比 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['負債總計', '資產總計']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['負債總計', '資產總計']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['負債總計', '資產總計']
        
            self.ProcessVariable = "負債總計/資產總計"    
    
    def Measure_16(self):

        self.Description = " 稅前淨利成長率 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['稅前純益']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['稅前純益']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['稅前純益']
        
            self.ProcessVariable = "稅前純益.pct_change()"    

               
    def Measure_17(self):

        self.Description = " 每股營收 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['單月營收','流通在外股數']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['單月營收','流通在外股數']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['單月營收','流通在外股數']
        
            self.ProcessVariable = "單月營收/流通在外股數"

    
    def Measure_18(self):

        self.Description = " 股東權益報酬率 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['綜合損益','權益總計']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['綜合損益','權益總計']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['綜合損益','權益總計']
        
            self.ProcessVariable = "綜合損益/權益總計" 


    def Measure_19(self):

        self.Description = " 毛利率 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['營業毛利','營業收入淨額']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['營業毛利','營業收入淨額']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['營業毛利','營業收入淨額']
        
            self.ProcessVariable = "營業毛利/營業收入淨額" 
    
    def Measure_20(self):

        self.Description = " 近5年，每年的公告基本每股盈餘都大於 1 元 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['公告基本每股盈餘']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['公告基本每股盈餘']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['公告基本每股盈餘']

            self.ProcessVariable = "( (公告基本每股盈餘 > 1).rolling(window= 5*252).apply(np.prod) ) * 1"    

    def Measure_21(self):

        self.Description = " 近12個月，每股營收大於 1.5 元 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['每股營收']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['每股營收']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['每股營收']

            self.ProcessVariable = "( (每股營收 > 1.5).rolling(window= 1*252).apply(np.prod) ) * 1"    

    def Measure_22(self):

        self.Description = " 股價淨值比小於 1.5 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['股價淨值比']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['股價淨值比']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['股價淨值比']

            self.ProcessVariable = "(股價淨值比 < 1.5) * 1"    

    def Measure_23(self):

        self.Description = " 近4季，股東權益報酬率大於5% "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['股東權益報酬率']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['股東權益報酬率']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['股東權益報酬率']

            self.ProcessVariable = "( (股東權益報酬率 > 0.05).rolling(window= 1*252).apply(np.prod) ) * 1"    

    def Measure_24(self):

        self.Description = " 近5年，毛利率每年都大於10% "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['毛利率']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['毛利率']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['毛利率']

            self.ProcessVariable = "( (毛利率 > 0.1).rolling(window= 5*252).apply(np.prod) ) * 1"  

                                                  
if __name__ == '__main__':
    if 1:
        measure_path = os.path.join(os.path.normpath(os.getcwd()),'data','measure_data')
        Open = pd.read_feather(os.path.join(measure_path,'Open.feather'))
        date = Open['Date']
        Open = Open.iloc[:,Open.columns!='Date'] # get all col. except the "Date"
        High = pd.read_feather(os.path.join(measure_path,'High.feather'))
        High = High.iloc[:,High.columns!='Date'] # get all col. except the "Date"
        Low = pd.read_feather(os.path.join(measure_path,'Low.feather'))
        Low = Low.iloc[:,Low.columns!='Date'] # get all col. except the "Date"
        Close = pd.read_feather(os.path.join(measure_path,'Close.feather'))
        Close = Close.iloc[:,Close.columns!='Date'] # get all col. except the "Date"
        Volume = pd.read_feather(os.path.join(measure_path,'Volume.feather'))
        Volume = Volume.iloc[:,Volume.columns!='Date'] # get all col. except the "Date"
        

        外資買賣超 = pd.read_feather(os.path.join(measure_path,'外資買賣超.feather'))
        外資買賣超 = 外資買賣超.iloc[:,外資買賣超.columns!='Date']
        自營商買賣超 = pd.read_feather(os.path.join(measure_path,'自營商買賣超.feather'))
        自營商買賣超 = 自營商買賣超.iloc[:,自營商買賣超.columns!='Date']
        投信買賣超 = pd.read_feather(os.path.join(measure_path,'投信買賣超.feather'))
        投信買賣超 = 投信買賣超.iloc[:,投信買賣超.columns!='Date']
        
        K9 = pd.read_feather(os.path.join(measure_path,'K9.feather'))
        K9 = K9.iloc[:,K9.columns!='Date']
        D9 = pd.read_feather(os.path.join(measure_path,'D9.feather'))
        D9 = D9.iloc[:,D9.columns!='Date']

        本益比4 = pd.read_feather(os.path.join(measure_path,'本益比4.feather'))
        本益比4 = 本益比4.iloc[:,本益比4.columns!='Date']
        單月營收年成長率 = pd.read_feather(os.path.join(measure_path,'單月營收年成長率.feather'))
        單月營收年成長率 = 單月營收年成長率.iloc[:,單月營收年成長率.columns!='Date']
        資產總計 = pd.read_feather(os.path.join(measure_path,'資產總計.feather'))
        資產總計 = 資產總計.iloc[:,資產總計.columns!='Date']
        負債總計 = pd.read_feather(os.path.join(measure_path,'負債總計.feather'))
        負債總計 = 負債總計.iloc[:,負債總計.columns!='Date']
        稅前純益 = pd.read_feather(os.path.join(measure_path,'稅前純益.feather'))
        稅前純益 = 稅前純益.iloc[:,稅前純益.columns!='Date']

        公告基本每股盈餘 = pd.read_feather(os.path.join(measure_path,'公告基本每股盈餘.feather'))
        公告基本每股盈餘 = 公告基本每股盈餘.iloc[:,公告基本每股盈餘.columns!='Date']
        單月營收 = pd.read_feather(os.path.join(measure_path,'單月營收.feather'))
        單月營收 = 單月營收.iloc[:,單月營收.columns!='Date']
        流通在外股數 = pd.read_feather(os.path.join(measure_path,'流通在外股數.feather'))
        流通在外股數 = 流通在外股數.iloc[:,流通在外股數.columns!='Date']
        股價淨值比 = pd.read_feather(os.path.join(measure_path,'股價淨值比.feather'))
        股價淨值比 = 股價淨值比.iloc[:,股價淨值比.columns!='Date']
        權益總計 = pd.read_feather(os.path.join(measure_path,'權益總計.feather'))
        權益總計 = 權益總計.iloc[:,權益總計.columns!='Date']
        綜合損益 = pd.read_feather(os.path.join(measure_path,'綜合損益.feather'))
        綜合損益 = 綜合損益.iloc[:,綜合損益.columns!='Date']
        營業毛利 = pd.read_feather(os.path.join(measure_path,'營業毛利.feather'))
        營業毛利 = 營業毛利.iloc[:,營業毛利.columns!='Date']
        營業收入淨額 = pd.read_feather(os.path.join(measure_path,'營業收入淨額.feather'))
        營業收入淨額 = 營業收入淨額.iloc[:,營業收入淨額.columns!='Date']

        # Process variables
        EMA200 = eval(MeasureLibrary('Measure_1').getProcessVariable())
        MACD_All = eval(MeasureLibrary('Measure_3').getProcessVariable())

        MACD = MACD_All.apply(lambda x: pd.Series(x[0]), axis=0)
        MACD_Signal = MACD_All.apply(lambda x: pd.Series(x[1]), axis=0)
        MACD_Histogram = MACD_All.apply(lambda x: pd.Series(x[2]), axis=0)

        過去60根K棒50百分位數的量 = eval(MeasureLibrary('Measure_4').getProcessVariable())

        saveProcessVariable(EMA200,'EMA200')
        saveProcessVariable(MACD,'MACD')
        saveProcessVariable(MACD_Signal,'MACD_Signal')
        saveProcessVariable(MACD_Histogram,'MACD_Histogram')
        saveProcessVariable(過去60根K棒50百分位數的量,'過去60根K棒50百分位數的量')        
        
        Bollinger_All = eval(MeasureLibrary('Measure_5').getProcessVariable())
        Bollinger_Upper = Bollinger_All.apply(lambda x: pd.Series(x[0]), axis=0)
        Bollinger_Middle = Bollinger_All.apply(lambda x: pd.Series(x[1]), axis=0)
        Bollinger_Lower = Bollinger_All.apply(lambda x: pd.Series(x[2]), axis=0)       
        RSI14 = eval(MeasureLibrary('Measure_6').getProcessVariable())  

        saveProcessVariable(Bollinger_Upper,'Bollinger_Upper')
        saveProcessVariable(Bollinger_Middle,'Bollinger_Middle')
        saveProcessVariable(Bollinger_Lower,'Bollinger_Lower')
        saveProcessVariable(RSI14,'RSI14')   
                
        三大法人買賣超 = eval(MeasureLibrary('Measure_7').getProcessVariable())
        saveProcessVariable(三大法人買賣超,'三大法人買賣超')
        
        SMA5 = eval(MeasureLibrary('Measure_8').getProcessVariable())  
        saveProcessVariable(SMA5,'SMA5')  
        SMA10 = eval(MeasureLibrary('Measure_9').getProcessVariable())  
        saveProcessVariable(SMA10,'SMA10')  
        SMA20 = eval(MeasureLibrary('Measure_10').getProcessVariable())  
        saveProcessVariable(SMA20,'SMA20')  
        SMA60 = eval(MeasureLibrary('Measure_11').getProcessVariable())  
        saveProcessVariable(SMA60,'SMA60')  
        
        Average_Volume10 = eval(MeasureLibrary('Measure_12').getProcessVariable())  
        saveProcessVariable(Average_Volume10,'Average_Volume10') 
        Average_Volume30 = eval(MeasureLibrary('Measure_13').getProcessVariable())  
        saveProcessVariable(Average_Volume30,'Average_Volume30') 
        
        J9 = eval(MeasureLibrary('Measure_14').getProcessVariable())
        saveProcessVariable(J9,'J9')

        負債比 = eval(MeasureLibrary('Measure_15').getProcessVariable())
        稅前淨利成長率 = eval(MeasureLibrary('Measure_16').getProcessVariable())

        saveProcessVariable(負債比,'負債比')
        saveProcessVariable(稅前淨利成長率,'稅前淨利成長率')

        
        每股營收 = eval(MeasureLibrary('Measure_17').getProcessVariable())
        股東權益報酬率 = eval(MeasureLibrary('Measure_18').getProcessVariable())
        毛利率 = eval(MeasureLibrary('Measure_19').getProcessVariable())
        近5年公告基本每股盈餘大於1 = eval(MeasureLibrary('Measure_20').getProcessVariable())
        近1年每股營收大於1_5 = eval(MeasureLibrary('Measure_21').getProcessVariable())
        股價淨值比小於1_5 = eval(MeasureLibrary('Measure_22').getProcessVariable())
        近1年股東權益報酬率大於0_05 = eval(MeasureLibrary('Measure_23').getProcessVariable())
        近5年毛利率大於0_1 = eval(MeasureLibrary('Measure_24').getProcessVariable())
        saveProcessVariable(每股營收,'每股營收')
        saveProcessVariable(股東權益報酬率,'股東權益報酬率')
        saveProcessVariable(毛利率,'毛利率')
        saveProcessVariable(近5年公告基本每股盈餘大於1,'近5年公告基本每股盈餘大於1')
        saveProcessVariable(近1年每股營收大於1_5,'近1年每股營收大於1_5')
        saveProcessVariable(股價淨值比小於1_5,'股價淨值比小於1_5')
        saveProcessVariable(近1年股東權益報酬率大於0_05,'近1年股東權益報酬率大於0_05')        
        saveProcessVariable(近5年毛利率大於0_1,'近5年毛利率大於0_1')        


        