
import pandas as pd
import os
import glob
import numpy as np

strategies_description = pd.read_excel("Comparison.xlsx", index_col=0, sheet_name = 'description')

strategy_list = []
strategies_results = []
strategies_summary = []

for item in os.listdir():
    item_path = os.path.join(os.getcwd(), item)

    # Check if the item is a directory and its name contains "Strategy"
    if os.path.isdir(item_path) and "Strategy" in item:
        # List files in the directory
        for filename in os.listdir(item_path):
            # Check if the filename contains "results"
            if "results" in filename:
                # Read the file or perform other actions
                file_path = os.path.join(item_path, filename)
                strategy_name = filename.replace("_results.csv","") 
                strategy_list.append(strategy_name)    
                result_df = pd.read_csv(file_path, index_col=0, encoding="utf-8-sig")
                strategies_results.append(result_df)
                
            if "summary" in filename:
                # Read the file or perform other actions
                file_path = os.path.join(item_path, filename)
                strategy_name = filename.replace("_summary.csv","") 
                summary_df = pd.read_csv(file_path, index_col=0, encoding="utf-8-sig")
                strategies_summary.append(summary_df)

strategies_results = pd.concat(strategies_results,keys = strategy_list,axis = 0, ignore_index=True)
strategies_results.insert(0, 'StrategyID', strategies_results.pop('_strategy'))
strategies_results = strategies_results.iloc[:,strategies_results.columns!='_strategy']

strategies_summary = pd.concat(strategies_summary,keys = strategy_list,axis = 0, ignore_index=True)
strategies_summary


# Exclude columns 'StrategyID', 'ticker', and 'Start'
columns_to_exclude = ['StrategyID', 'ticker', 'Start','End']

# Create a mask to identify columns to rank
columns_to_rank = strategies_results.columns.difference(columns_to_exclude)
'''
# Apply rank to selected columns
strategies_results_rank = strategies_results.copy()
strategies_results_rank[columns_to_rank] = strategies_results[columns_to_rank].rank(pct=True)*100
'''

rank_strategies = strategies_results.copy() # 哪個標的最適合該策略
rank_ticker = strategies_results.copy() # 哪個策略最適合該標的


for col in columns_to_rank:
    rank_strategies[col] = strategies_results.groupby('StrategyID')[col].transform('rank')/len(strategies_results['ticker'].unique())
    rank_ticker[col] = strategies_results.groupby('ticker')[col].transform('rank')/len(strategies_results['StrategyID'].unique())
target_col = ['StrategyID', 'ticker','Return (Ann.) [%]', 'Volatility (Ann.) [%]','Sharpe Ratio','Max. Drawdown [%]','# Trades', 'Win Rate [%]','Max. Trade Duration']
rank_strategies = rank_strategies[target_col]
rank_ticker = rank_ticker[target_col]

criteria_col = ['Return (Ann.) [%]', 'Volatility (Ann.) [%]', 'Sharpe Ratio','Max. Drawdown [%]', '# Trades', 'Win Rate [%]', 'Max. Trade Duration']
rank_strategies["Overall"] = rank_strategies[criteria_col].sum(axis =1)
rank_ticker["Overall"] = rank_ticker[criteria_col].sum(axis =1)
rank_strategies_overall = rank_strategies.pivot(values=['Overall'], index='ticker',columns=['StrategyID'])
rank_ticker_overall = rank_ticker.pivot(values=['Overall'], index='ticker',columns=['StrategyID'])


# Write DataFrame to Excel file with automatic column width adjustment
with pd.ExcelWriter('Output.xlsx') as writer:
    strategies_description.to_excel(writer, sheet_name='description', float_format="%.2f", encoding="utf-8-sig")
    strategies_summary.to_excel(writer, sheet_name='summary', float_format="%.2f", encoding="utf-8-sig")
    strategies_results.to_excel(writer, sheet_name='results', float_format="%.2f", encoding="utf-8-sig")         
    #strategies_results_rank.to_excel(writer, sheet_name='results_rank', float_format="%.2f", encoding="utf-8-sig")         
    rank_strategies.to_excel(writer, sheet_name='rank_by_strategies', float_format="%.2f", encoding="utf-8-sig")         
    rank_ticker.to_excel(writer, sheet_name='rank_by_ticker', float_format="%.2f", encoding="utf-8-sig")         
    rank_strategies_overall.to_excel(writer, sheet_name='rank_by_strategies_overall', float_format="%.2f", encoding="utf-8-sig")         
    rank_ticker_overall.to_excel(writer, sheet_name='rank_by_ticker_overall', float_format="%.2f", encoding="utf-8-sig")         

    worksheet_description = writer.sheets['description']
    worksheet_summary = writer.sheets['summary']
    worksheet_results = writer.sheets['results']
    #worksheet_results = writer.sheets['results_rank']
    worksheet_rank_by_strategies = writer.sheets['rank_by_strategies']
    worksheet_rank_by_ticker = writer.sheets['rank_by_ticker']
    worksheet_rank_by_strategies_overall = writer.sheets['rank_by_strategies_overall']
    worksheet_rank_by_ticker_overall = writer.sheets['rank_by_ticker_overall']    
    # Auto-adjust columns' width
    for i, column in enumerate(strategies_description.columns):
        column_width = max(strategies_description[column].astype(str).apply(len).max(), len(column)) + 2  # Add padding
        worksheet_description.set_column(i+1, i+1, column_width)

    # Auto-adjust columns' width
    for i, column in enumerate(strategies_summary.columns):
        column_width = max(strategies_summary[column].astype(str).apply(len).max(), len(column)) + 2  # Add padding
        worksheet_summary.set_column(i+1, i+1, column_width)
        
    # Auto-adjust columns' width
    for i, column in enumerate(strategies_results.columns):
        column_width = max(len((column)),(strategies_results[column].astype(str).apply(len).max()), len(column)) + 2  # Add padding
        worksheet_results.set_column(i+1, i+1, column_width)
    '''
    # Auto-adjust columns' width
    for i, column in enumerate(strategies_results_rank.columns):
        column_width = max(len((column)),(strategies_results_rank[column].astype(str).apply(len).max()), len(column)) + 2  # Add padding
        worksheet_results.set_column(i+1, i+1, column_width)
    '''
    # Auto-adjust columns' width
    for i, column in enumerate(rank_strategies.columns):
        column_width = max(len((column)),(rank_strategies[column].astype(str).apply(len).max()), len(column)) + 2  # Add padding
        worksheet_rank_by_strategies.set_column(i+1, i+1, column_width)

    # Auto-adjust columns' width
    for i, column in enumerate(rank_ticker.columns):
        column_width = max(len((column)),(rank_ticker[column].astype(str).apply(len).max()), len(column)) + 2  # Add padding
        worksheet_rank_by_ticker.set_column(i+1, i+1, column_width)

    # Auto-adjust columns' width
    for i, column in enumerate(rank_strategies_overall.columns):
        column_width = max(len((column)),(rank_strategies_overall[column].astype(str).apply(len).max()), len(column)) + 2  # Add padding
        worksheet_rank_by_strategies_overall.set_column(i+1, i+1, column_width)

    # Auto-adjust columns' width
    for i, column in enumerate(rank_ticker_overall.columns):
        column_width = max(len((column)),(rank_ticker_overall[column].astype(str).apply(len).max()), len(column)) + 2  # Add padding
        worksheet_rank_by_ticker_overall.set_column(i+1, i+1, column_width)
        
        
  