```
按下ctrl + shit + v 可以看到預覽
```
目標 : 一個月完成到階段二

# 1.安裝
- VS CODE [https://code.visualstudio.com/](https://code.visualstudio.com/)
- anaconda [https://www.anaconda.com/download](https://www.anaconda.com/download)
- tabnine (用免費版 是VSCODE套件)[https://www.tabnine.com/install/vs-code/](https://www.tabnine.com/install/vs-code/)
- git [https://git-scm.com/](https://git-scm.com/)

# 設定
## DB設定
將.env中的參數設定正確
## git 設定 (不影醒python執行，只影響每週提交)
[https://git-scm.com/book/zh-tw/v2/%E9%96%8B%E5%A7%8B-%E5%88%9D%E6%AC%A1%E8%A8%AD%E5%AE%9A-Git](https://git-scm.com/book/zh-tw/v2/%E9%96%8B%E5%A7%8B-%E5%88%9D%E6%AC%A1%E8%A8%AD%E5%AE%9A-Git)
雙引號內填入資料只在你的機器上有 目的是方便識別 事必填
```
git config --global user.name "您的名字"
git config --global user.email "您的電子郵件"
```
---

# 執行步驟
## 階段一 先可以執行

1. 持行[delete_data.py](src/delete_data.py) 作用: 刪除資料
2. 執行[get_data.py](src/get_data.py)  作用: 抓資料 -> 資料存到 /data/measure_data
3. 執行[measure_library_username.py](src/measure_library_username.py) 作用: 建立DB以外的measure -> 資料存到 /data/measure_data
4. 執行[rule_library_username.py](src/rule_library_username.py) 作用: 建立DB以外的rule data -> 資料存到 /data/rule_data
5. 執行[backtesting_total_username.py](src/backtesting_total_username) 作用: 執行回測 -> 資料存到 /Backtesting
6. 執行[report.py](src/report.py) 作用: 產生大表big.csv和RulePerformance
7. 執行[report_by_strategy.py](src/report_by_strategy) 作用: 產生大表big.csv和RulePerformance
---

## 階段二 改成客製化的內容
1. 把username改成新的英文名子
2. 重新測試階段一的流程
3. 在rule_library_username.py新增新的規則並同時新增需要處理的程式碼

## 階段三 訓練因子組合模型 安裝可在gpu上執行的 torch (進階)
0. 安裝pytorch [https://pytorch.org/](https://pytorch.org/)
1. 執行shared_linear_model.py
2. 執行 portfolio_backtesting_tickerbase. py 作用: 建立、儲存模型 並顯示參數
3. 利用參數結果在Backtesting_username.py建立新的回測並執行
4. 新增新的報告分析程式碼在Report.py 並執行觀察效果
