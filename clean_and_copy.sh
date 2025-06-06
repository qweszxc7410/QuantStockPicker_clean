#!/bin/bash
# 不含刪除原始檔，只做遮蔽與資料夾複製

# === 設定 ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
DEST_DIR="${PARENT_DIR}/${PROJECT_NAME}_clean"
ENV_ORIG="${SCRIPT_DIR}/.env"
ENV_CLEAN="${DEST_DIR}/.env"

echo "📁 原始專案目錄：$SCRIPT_DIR"
echo "📦 交付輸出目錄：$DEST_DIR"

# === 複製整個專案目錄（含所有檔案與資料夾） ===
rm -rf "$DEST_DIR"
cp -r "$SCRIPT_DIR" "$DEST_DIR"

# === 清除不該交付的內容 ===
echo "🧹 移除敏感與暫存資料..."
rm -rf "$DEST_DIR/.git"
rm -rf "$DEST_DIR/.conda"
find "$DEST_DIR" -type d -name "__pycache__" -exec rm -rf {} +

# === 處理 .env（遮蔽值，保留格式） ===
if [ -f "$ENV_ORIG" ]; then
    echo "🔐 遮蔽 .env 中的值..."
    mkdir -p "$(dirname "$ENV_CLEAN")"
    > "$ENV_CLEAN"  # 清空目標檔案，確保不是 append

    while IFS= read -r line || [[ -n "$line" ]]; do
        # 空行或註解直接原樣寫入
        if [[ "$line" =~ ^[[:space:]]*$ || "$line" =~ ^[[:space:]]*# ]]; then
            echo "$line" >> "$ENV_CLEAN"
            continue
        fi

        key=$(echo "$line" | cut -d= -f1)
        value=$(echo "$line" | cut -d= -f2-)

        # 去除左右空白與引號
        value=$(echo "$value" | sed -E 's/^[[:space:]]*["'\'']?(.*?)["'\'']?[[:space:]]*$/\1/')

        len=${#value}
        if (( len <= 8 )); then
            masked="********"
        else
            start=${value:0:len/4}
            end=${value: -len/4}
            masked="${start}********${end}"
        fi

        echo "${key}=${masked}" >> "$ENV_CLEAN"
    done < "$ENV_ORIG"

else
    echo "⚠️ 找不到 .env，跳過遮蔽處理。"
fi


echo "🎉 完成！交付版本建立於：$DEST_DIR"
