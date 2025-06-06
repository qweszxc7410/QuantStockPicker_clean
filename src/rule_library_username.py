import pandas as pd
import os
import talib as ta
import numpy as np
from varname import nameof

def saveProcessVariable(df:pd.DataFrame,df_name:str):
    df.insert(0,"Date",date)
    df.to_feather(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data",'rule_data',f"{df_name}.feather"))

    
class RuleLibrary_V1():
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

    def Rule_1(self):

        self.Description = " MACD: Histogram "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['MACD','MACD_Signal']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['MACD','MACD_Signal']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData =  ['MACD','MACD_Signal']

            self.ProcessVariable = "MACD - MACD_Signal"  

    def Rule_2(self):

        self.Description = " MACD 在零線以下 + 上穿信號線 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['Close','MACD','MACD_Signal']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['Close','MACD','MACD_Signal']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['Close','MACD','MACD_Signal']
            
            self.ProcessVariable = "( (MACD < 0) & (MACD > MACD_Signal) & (MACD.shift(-1,axis=0) < MACD_Signal.shift(-1,axis=0)) ) * 1"    

    def Rule_3(self):

        self.Description = " MACD 在零線以上 + 下穿信號線 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['Close','MACD','MACD_Signal']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['Close','MACD','MACD_Signal']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['Close','MACD','MACD_Signal']
            
            self.ProcessVariable = "( (MACD > 0) & (MACD < MACD_Signal) & (MACD.shift(-1,axis=0) > MACD_Signal.shift(-1,axis=0)) ) * 1"    
    
            
    def Rule_4(self):

        self.Description = " MACD向上交叉零線 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['Close','MACD','MACD_Signal']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['Close','MACD','MACD_Signal']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['Close','MACD','MACD_Signal']
            
            self.ProcessVariable = "( (MACD > 0) & (MACD.shift(-1,axis=0) < 0) ) * 1"    

    def Rule_5(self):

        self.Description = " MACD向下交叉零線 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['Close','MACD','MACD_Signal']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['Close','MACD','MACD_Signal']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['Close','MACD','MACD_Signal']
            
            self.ProcessVariable = "( (MACD < 0) & (MACD.shift(-1,axis=0) > 0) ) * 1"  

    def Rule_6(self):

        self.Description = " 股價跌出布林通道下軌 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['Close','Bollinger_Lower']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['Close','Bollinger_Lower']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData =  ['Close','Bollinger_Lower']

            self.ProcessVariable = "( (Close.fillna(0) > 0) & (pd.notna(Close)) & (Close < Bollinger_Lower) & (Close.shift(-1,axis=0) > Bollinger_Lower.shift(-1,axis=0))  > 0)"    

    def Rule_7(self):

        self.Description = " 本益比小於25 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['本益比4']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['本益比4']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['本益比4']
        
            self.ProcessVariable = "本益比4 < 25" 
            
    def Rule_8(self):

        self.Description = " 近2年營收成長率平均大於0.1 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['單月營收年成長率']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['單月營收年成長率']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['單月營收年成長率']
            
            self.ProcessVariable = "( 單月營收年成長率.rolling(window= 2*252).mean()> 0.1 ) * 1"    
    
    def Rule_9(self):

        self.Description = " 近5年稅前淨利成長率，平均大於0.03 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['稅前淨利成長率']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['稅前淨利成長率']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['稅前淨利成長率']
            
            self.ProcessVariable = "( 稅前淨利成長率.rolling(window= 5*252).mean()> 0.03 ) * 1"    

    def Rule_10(self):

        self.Description = " 最近一季負債比小於0.25 "

        if self.UseMethod == "Ticker":
            pass

        elif self.UseMethod == "CrossSection":
            # 資料庫的欄位名稱
            self.UseData = ['負債比']
            # Excel檔名稱(分因子資料夾)( $ 代替 : )
            self.CrossSectionDataFileName = ['負債比']
            # 合成資料用的名稱(跟檔案名稱不同)(dataretriever用)(沒有 $ 只有 : )
            self.CrossSectionData = ['負債比']
            
            self.ProcessVariable = "( 負債比.rolling(window= 1*90).mean()< 0.25 ) * 1"    


if __name__ == '__main__':
    if 1:
        base_path = os.path.dirname(os.path.abspath(__file__))  # 取得目前檔案的路徑
        rule_path = os.path.join(os.path.dirname(base_path),'data','rule_data')
        measure_path = os.path.join(os.path.dirname(base_path),'data','measure_data')

        Open = pd.read_feather(os.path.join(measure_path,'Open.feather'))
        build_rule_list = ["Rule_4","Rule_5","Rule_6","Rule_7"]
        for build_id in build_rule_list:
            print(f"建立 {build_id} 資料")
            file_name_list = RuleLibrary_V1(build_id).getCrossSectionDataFileName()
            
            for idx,name in enumerate(file_name_list):
                
                globals()[name] = pd.read_feather(os.path.join(measure_path,name + '.feather'))
                if idx == 0:
                    date = globals()[name]['Date']
                globals()[name] = globals()[name].iloc[:,globals()[name] .columns!='Date'] 
                print(f"建立 {name} 資料")
            saveProcessVariable(eval(RuleLibrary_V1(build_id).getProcessVariable()),name)

        print()

        
    if 0:
        rule_path = os.path.join(os.path.normpath(os.getcwd()),'data','rule_data')
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
        
        MACD = pd.read_feather(os.path.join(measure_path,'MACD.feather'))
        MACD = MACD.iloc[:,MACD.columns!='Date'] 
        MACD_Signal = pd.read_feather(os.path.join(measure_path,'MACD_Signal.feather'))
        MACD_Signal = MACD_Signal.iloc[:,MACD_Signal.columns!='Date'] 
        MACD_Histogram = pd.read_feather(os.path.join(measure_path,'MACD_Histogram.feather'))
        MACD_Histogram = MACD_Histogram.iloc[:,MACD_Histogram.columns!='Date'] 
        
        MACD零線以下且上穿信號線 = eval(RuleLibrary_V1('Rule_2').getProcessVariable())
        MACD零線以上且下穿信號線 = eval(RuleLibrary_V1('Rule_3').getProcessVariable())
        MACD向上交叉零線 = eval(RuleLibrary_V1('Rule_4').getProcessVariable())
        MACD向下交叉零線 = eval(RuleLibrary_V1('Rule_5').getProcessVariable())

        saveProcessVariable(MACD零線以下且上穿信號線,'MACD零線以下且上穿信號線')
        saveProcessVariable(MACD零線以上且下穿信號線,'MACD零線以上且下穿信號線')        
        saveProcessVariable(MACD向上交叉零線,'MACD向上交叉零線') 
        saveProcessVariable(MACD向下交叉零線,'MACD向下交叉零線') 
        
        

        Bollinger_Upper = pd.read_feather(os.path.join(measure_path,'Bollinger_Upper.feather'))
        Bollinger_Upper = Bollinger_Upper.iloc[:,Bollinger_Upper.columns!='Date'] 
        Bollinger_Middle = pd.read_feather(os.path.join(measure_path,'Bollinger_Middle.feather'))
        Bollinger_Middle = Bollinger_Middle.iloc[:,Bollinger_Middle.columns!='Date'] 
        Bollinger_Lower = pd.read_feather(os.path.join(measure_path,'Bollinger_Lower.feather'))
        Bollinger_Lower = Bollinger_Lower.iloc[:,Bollinger_Lower.columns!='Date'] 
        
        股價跌出布林通道下軌 = eval(RuleLibrary_V1('Rule_6').getProcessVariable())
        saveProcessVariable(股價跌出布林通道下軌,'股價跌出布林通道下軌')               
             
        稅前淨利成長率 = pd.read_feather(os.path.join(measure_path,'稅前淨利成長率.feather'))
        稅前淨利成長率 = 稅前淨利成長率.iloc[:,稅前淨利成長率.columns!='Date'] 



        負債比 = pd.read_feather(os.path.join(measure_path,'負債比.feather'))
        負債比 = 負債比.iloc[:,負債比.columns!='Date'] 
        
        
        本益比小於25 = eval(RuleLibrary_V1('Rule_7').getProcessVariable())
        近2年營收成長率平均大於0_1 = eval(RuleLibrary_V1('Rule_8').getProcessVariable())
        近5年稅前淨利成長率平均大於0_03 = eval(RuleLibrary_V1('Rule_9').getProcessVariable())
        最近一季負債比小於0_25 = eval(RuleLibrary_V1('Rule_10').getProcessVariable())

        saveProcessVariable(本益比小於25,'本益比小於25')
        saveProcessVariable(近2年營收成長率平均大於0_1,'近2年營收成長率平均大於0_1')
        saveProcessVariable(近5年稅前淨利成長率平均大於0_03,'近5年稅前淨利成長率平均大於0_03')        
        saveProcessVariable(最近一季負債比小於0_25,'最近一季負債比小於0_25')  
        



        