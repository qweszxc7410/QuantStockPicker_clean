import pandas as pd
import matplotlib.pyplot as plt
import os
import rule_library_username as rule_library_username

file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'output','report')
data = pd.read_csv(os.path.join(file_path,'bigtable.csv'), encoding='utf-8-sig')

# 讓使用者逐步輸入最多 4 個 Rule 的數字部分和底線
rule_names = []
rule_names = list(set(data['Rule']))  # 取得所有的 Rule 名稱
data['Buy & Hold Return [%]'] = data['Buy & Hold Return [%]'].replace([float('inf'), float('-inf')], 0)

# 用於保存選擇後的資料
selected_data = pd.DataFrame()

for idx, rule_name in enumerate(rule_names):
    rule_data = data[data['Rule'] == rule_name]

    if rule_data.empty:
        print(f"找不到與 {rule_name} 對應的資料。")
        continue

    # 建立樞紐分析表
    pivot_data = rule_data.pivot_table(
        index=['_strategy', 'Rule'],
        values=['Return [%]', 'Buy & Hold Return [%]', 'Exposure Time [%]', 
                'Equity Final [$]', 'Equity Peak [$]', 'Return (Ann.) [%]', 
                'Volatility (Ann.) [%]', 'Sharpe Ratio', 'Sortino Ratio', 
                'Calmar Ratio', 'Max. Drawdown [%]', 'Avg. Drawdown [%]', 
                'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]', 
                'Avg. Trade [%]', 'Profit Factor', 'Expectancy [%]', 'SQN'],
            aggfunc={
                'Return [%]': 'sum',
                'Buy & Hold Return [%]': 'sum',
                'Exposure Time [%]': 'mean',
                'Equity Final [$]': 'sum',
                'Equity Peak [$]': 'sum',
                'Return (Ann.) [%]': 'mean',
                'Volatility (Ann.) [%]': 'mean',
                'Sharpe Ratio': 'mean',
                'Sortino Ratio': 'mean',
                'Calmar Ratio': 'mean',
                'Max. Drawdown [%]': 'sum',
                'Avg. Drawdown [%]': 'mean',
                'Win Rate [%]': 'sum',
                'Best Trade [%]': 'mean',
                'Worst Trade [%]': 'mean',
                'Avg. Trade [%]': 'mean',
                'Profit Factor': 'mean',
                'Expectancy [%]': 'mean',
                'SQN': 'mean'
            }
    ).reset_index()

    # 將實際的索引設置為使用者輸入的 Rule+_strategy 名稱
    pivot_data['_strategy'] = rule_data['_strategy'].unique()  # 使用原始資料中的 _strategy
    pivot_data.set_index('_strategy', inplace=True)
    
    # 保存選擇的資料
    selected_data = pd.concat([selected_data, pivot_data])

pivot_data_sorted = selected_data.sort_values(by='Return (Ann.) [%]', ascending=False)
pivot_data_sorted.reset_index(inplace=True)
# _strategy  重新命名成 Strategy
pivot_data_sorted.rename(columns={'_strategy':'Strategy'}, inplace=True)
def get_rule_description(rule_name):
    try:
        return rule_library_username.RuleLibrary_V1(rule_name).getRuleDescription()
    except AttributeError:
        return "無資料"

# 使用 apply 調用該輔助函數
pivot_data_sorted['Rule描述'] = pivot_data_sorted['Rule'].apply(get_rule_description)
columns_order = ['Strategy', 'Rule描述', 'Rule'] + [col for col in pivot_data_sorted.columns if col not in ['Strategy', 'Rule描述', 'Rule']]
# 重新排列 pivot_data_sorted 的欄位順序
pivot_data_sorted = pivot_data_sorted[columns_order]
pivot_data_sorted.to_csv(os.path.join(file_path,'pivot_data.csv'), index=True,encoding = "utf-8-sig")
print(f"已將樞紐分析表儲存到 {os.path.join(file_path,'pivot_data.csv')}")