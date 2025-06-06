#!/bin/bash

# 要清空的資料夾列表
TARGET_FOLDERS=(
    "data/measure_data"
    "data/rule_data"
    "output/backtesting"
    "output/model"
    "output/report"
    "output/score",
    "output/report/*.csv"
)

echo "🔄 開始清空目標資料夾內容..."

for folder in "${TARGET_FOLDERS[@]}"; do
    full_path="$folder"
    if [ -d "$full_path" ]; then
        echo "🧹 清空 $full_path ..."
        rm -rf "$full_path"/* "$full_path"/.* 2>/dev/null
    else
        echo "❌ 資料夾 $full_path 不存在，略過。"
    fi
done

echo "✅ 所有資料夾已清空完畢！"
