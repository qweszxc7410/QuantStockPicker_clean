import os
import pandas as pd
from backtesting_total_username import run_by_ticker
from report import combine_backtesting_report_by_rule,combine_big_backtesting_report_by_ticker_by_rule
class ProfolioBacktestingTickerbaserCheck():
    '''' 簡單把策略組合起來進行回測 '''
    def __init__(self,open=None,high=None,low=None,close=None,
                 rule_file_name_list=None,weight_list=None,strategy_start_number=1,
                 threshold_list = [0.2,0.5,0.8],rule_data_path = None,strategy_method_name = "BuyOne_SellZero"):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.rule_file_name_list = rule_file_name_list
        self.weight_list = weight_list
        self.strategy_start_number = strategy_start_number
        self.threshold_list = threshold_list
        self.rule_data_path = rule_data_path
        self.strategy_method_name = strategy_method_name
        self.execute_backtesting_name = []
        
    def read_rule_data(self):
        self.score_df = None
        for weight, filename in zip(self.weight_list, self.rule_file_name_list):
            print(f"{filename} weight: {weight:.4f}")
            self.score_date = pd.read_feather(os.path.join(self.rule_data_path,filename + '.feather')).index
            if self.score_df is None:
                self.score_df = pd.read_feather(os.path.join(self.rule_data_path,filename + '.feather')).iloc[:, 1:]*1*weight
                self.ticker_list = pd.read_feather(os.path.join(self.rule_data_path,filename + '.feather')).iloc[:, 1:].columns
            else:
                self.score_df = self.score_df + pd.read_feather(os.path.join(self.rule_data_path,filename + '.feather')).iloc[:, 1:]*1*weight
        self.score_date = pd.read_feather(os.path.join(self.rule_data_path,filename + '.feather'))['Date']      
    def backtesting_tickerbase(self):
        for i,strategy_number in enumerate(self.threshold_list): # 根據不同門檻值測試
            strategy_name = "TPB_M_" + str(strategy_start_number + i) # 策略名稱
            self.execute_backtesting_name.append(strategy_name)
            print(strategy_name,strategy_number)
            # threshold = round(sum(abs(x) for x in weight_list) * strategy_number + min(weight_list),4) # 前 X %
            threshold = round(sum(abs(x) for x in weight_list) * strategy_number + max(0, min(weight_list)), 4) # 前 X % 避免負值影響：
            print(f"Strategy {strategy_name} threshold: {strategy_number:.4f} threshold {threshold:.4f}")
            # score_df_backtesting = (((self.score_df) < threshold * (self.score_df))  + (self.score_df) >= threshold)*1 # 原始 應該是錯的
            score_df_backtesting = ((self.score_df >= threshold) * 1)
            # print(f"Score Matrix Max{score_df_backtesting.max().max()} Min{score_df_backtesting.min().min()}")
            score_df_backtesting['Date'] = self.score_date
            # score_df_backtesting.to_csv("score_df_backtesting.csv")
            # self.score_df.to_csv("score_df.csv")
            if 1:
                run_by_ticker(ticker_list = self.ticker_list,BacktestingStrategy = self.strategy_method_name,signal_name = strategy_name,
                            in_signal_df = score_df_backtesting,out_signal_df = score_df_backtesting,
                            Open = self.open,High = self.high,Low = self.low,Close = self.close)
            del score_df_backtesting


    def get_backtesting_strategy_name(self):
        return self.execute_backtesting_name
    
    
    def run(self):
        self.read_rule_data()
        self.backtesting_tickerbase()

if __name__ == '__main__':
    if 0: # 簡單把策略組合起來進行回測
        # setting
        measure_data_path = os.path.join(os.path.normpath(os.getcwd()),'data','measure_data')
        rule_data_path = os.path.join(os.path.normpath(os.getcwd()),'data','rule_data')
        Open = pd.read_feather(os.path.join(measure_data_path,'Open.feather'))
        High = pd.read_feather(os.path.join(measure_data_path,'High.feather'))
        Low = pd.read_feather(os.path.join(measure_data_path,'Low.feather'))
        Close = pd.read_feather(os.path.join(measure_data_path,'Close.feather'))
        thresholTicker_listd_list = [0.2,0.5,0.8]
        rule_file_name_list = ['本益比小於25', '股價跌出布林通道下軌', '近5年稅前淨利成長率平均大於0_03']
        weight_list = [ 0.3581,  1.4659, -0.8240]
        measure_data_path = os.path.join(os.path.normpath(os.getcwd()),'data','measure_data')
        rule_data_path = os.path.join(os.path.normpath(os.getcwd()),'data','rule_data')
        strategy_start_number = 4
        strategy_method_name = "BuyOne_SellZero"
        # run
        obj_ProfolioBacktestingTickerbaserCheck = ProfolioBacktestingTickerbaserCheck(
            open=Open,
            high=High,
            low=Low,
            close=Close,
            rule_file_name_list=rule_file_name_list,
            weight_list=weight_list,
            strategy_start_number=strategy_start_number,
            threshold_list = thresholTicker_listd_list,
            rule_data_path = rule_data_path,
            strategy_method_name = strategy_method_name
        )
        obj_ProfolioBacktestingTickerbaserCheck.run()
        execute_backtesting_name_list = obj_ProfolioBacktestingTickerbaserCheck.get_backtesting_strategy_name()

        if 1: #產生報表
            base_path = os.path.dirname(os.path.abspath(__file__))
            report_path = os.path.join(base_path, 'report',"StrategyAndRule")
            data_path = os.path.join(base_path, 'backtesting')
                
            for execute_backtesting_name in execute_backtesting_name_list:
                print(f"{execute_backtesting_name = }")
                combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = strategy_method_name, rule = execute_backtesting_name)


            base_path = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(base_path, 'report','StrategyAndRule')
            report_path = os.path.join(os.getcwd(), 'report')
            combine_big_backtesting_report_by_ticker_by_rule(data_path = data_path , save_path = report_path)

    if 1: # 複雜把策略組合起來進行回測
        # setting
        measure_data_path = os.path.join(os.path.normpath(os.getcwd()),'data','measure_data')
        rule_data_path = os.path.join(os.path.normpath(os.getcwd()),'data','rule_data')
        Open = pd.read_feather(os.path.join(measure_data_path,'Open.feather'))
        High = pd.read_feather(os.path.join(measure_data_path,'High.feather'))
        Low = pd.read_feather(os.path.join(measure_data_path,'Low.feather'))
        Close = pd.read_feather(os.path.join(measure_data_path,'Close.feather'))
        thresholTicker_listd_list = [0.2,0.5,0.8]
        thresholTicker_listd_list = [0.5]
        
        rule_file_name_list = ['本益比小於25', '股價跌出布林通道下軌', '近5年稅前淨利成長率平均大於0_03']
        # weight_list_list 權重相加為1
        weight_list_list = [
                [0.5614891052246094, 0.6109549403190613, -0.17244406044483185],
                [0.3995233476161957, 1.0982190370559692, -0.49774235486984253],
                # [0.35809335112571716, 1.4659380912780762, -0.8240314722061157],
                # [0.3065713346004486, 0.984082043170929, -0.2906533181667328],
                # [0.33692193031311035, 0.922614574432373, -0.259536474943161],
                # [0.35945671796798706, 0.9672375917434692, -0.3266943693161011],
                # [0.42609283328056335, 1.048132061958313, -0.47422486543655396],
                # [0.4284682273864746, 0.7151430249214172, -0.14361128211021423],
                # [0.43563857674598694, 0.6503143310546875, -0.08595292270183563],
                # [0.4667704403400421, 0.6489984393119812, -0.11576884984970093],
                # [0.526644229888916, 0.6290081739425659, -0.15565244853496552],
                # [0.560153067111969, 0.48885834217071533, -0.049011390656232834],
                # [0.5531840920448303, 0.41499388217926025, 0.03182206302881241],
                # [0.5778454542160034, 0.40093734860420227, 0.02121722139418125],
                # [0.6216303110122681, 0.35131317377090454, 0.027056561782956123],
                # [0.6912123560905457, 0.2757418155670166, 0.03304590284824371],
                # [0.6583433151245117, 0.22907258570194244, 0.11258413642644882],
                # [0.6688755750656128, 0.22035779058933258, 0.11076666414737701],
                # [0.7039415836334229, 0.15739861130714417, 0.13865980505943298],
                # [0.8366585373878479, 0.08440813422203064, 0.07893333584070206],
                # [0.7411849498748779, 0.09197138994932175, 0.16684366762638092]

                    ]
        strategy_start_number = 800
        for weight_list in weight_list_list:
            measure_data_path = os.path.join(os.path.normpath(os.getcwd()),'data','measure_data')
            rule_data_path = os.path.join(os.path.normpath(os.getcwd()),'data','rule_data')

            strategy_method_name = "BuyOne_Hold120"
            # run
            obj_ProfolioBacktestingTickerbaserCheck = ProfolioBacktestingTickerbaserCheck(
                open=Open,
                high=High,
                low=Low,
                close=Close,
                rule_file_name_list=rule_file_name_list,
                weight_list=weight_list,
                strategy_start_number=strategy_start_number,
                threshold_list = thresholTicker_listd_list,
                rule_data_path = rule_data_path,
                strategy_method_name = strategy_method_name
            )
            obj_ProfolioBacktestingTickerbaserCheck.run()
            execute_backtesting_name_list = obj_ProfolioBacktestingTickerbaserCheck.get_backtesting_strategy_name()

            if 1: #產生報表
                base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                report_path = os.path.join(base_path,'output', 'report',"StrategyAndRule")
                data_path = os.path.join(base_path,'output', 'backtesting')
                    
                for execute_backtesting_name in execute_backtesting_name_list:
                    print(execute_backtesting_name)
                    combine_backtesting_report_by_rule(data_path = data_path , save_path = report_path,strategy = strategy_method_name, rule = execute_backtesting_name)


                base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                data_path = os.path.join(base_path,'output', 'report','StrategyAndRule')
                report_path = os.path.join(base_path, 'report')
                combine_big_backtesting_report_by_ticker_by_rule(data_path = data_path , save_path = report_path)
            strategy_start_number = strategy_start_number + 1

