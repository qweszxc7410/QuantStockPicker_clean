#!/bin/bash
# 設定根目錄
SRC_DIR="./src"
export LANG=zh_TW.UTF-8
export LC_ALL=zh_TW.UTF-8

# 步驟 1: 刪除資料
# echo "🗑️ [1/7] 執行 delete_data.py 刪除資料..."
# python "$SRC_DIR/delete_data.py"

# 步驟 2: 從 FRED 抓資料
# echo "📥 [2/7] 執行 get_data.py 抓取資料..."
# python "$SRC_DIR/get_data.py"

# 步驟 3: 建立 DB 以外的 measure
# echo "📊 [3/7] 執行 measure_library_username.py 建立自定義 measure..."
# python "$SRC_DIR/measure_library_username.py"

# # 步驟 4: 建立 DB 以外的 rule data
echo "📏 [4/7] 執行 rule_library_username.py 建立自定義 rule data..."
python "$SRC_DIR/rule_library_username.py"

# # 步驟 5: 執行回測
echo "📈 [5/7] 執行 backtesting_total_username.py 進行回測分析..."
python "$SRC_DIR/backtesting_total_username.py"

# # 步驟 6: 產生大表和 RulePerformance
echo "📋 [6/7] 執行 report.py 產生報表資料..."
python "$SRC_DIR/report.py"

# # 步驟 7: 產生策略報表
echo "📊 [7/7] 執行 report_by_strategy.py 產生策略報表..."
python "$SRC_DIR/report_by_strategy.py"

echo "✅ 所有流程已完成！"