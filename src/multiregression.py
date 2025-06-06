import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import random
from datetime import datetime
import os
import math

result = math.log(1/2)
print(f"{result:.6f}")

# 固定亂數 確保重複執行會一樣
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

device = "cpu" # 裝置
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

base_path = os.path.dirname(os.path.abspath(__file__))  # 取得目前檔案的路徑
measure_data_path = os.path.join(base_path, 'data', 'measure_data')  # 設定因子路徑
rule_data_path = os.path.join(base_path, 'data', 'rule_data')  # 設定規則路徑
model_path = os.path.join(base_path, 'model')  # 設定模型路徑

# 讀資料 第一欄為 Date 內容是True 和 False
factor1 = pd.read_feather(os.path.join(rule_data_path,'本益比小於25.feather')).iloc[:, 1:]*1
factor2 = pd.read_feather(os.path.join(rule_data_path,'股價跌出布林通道下軌.feather')).iloc[:, 1:]*1
factor3 = pd.read_feather(os.path.join(rule_data_path,'近5年稅前淨利成長率平均大於0_03.feather')).iloc[:, 1:]*1
returns = pd.read_feather(os.path.join(measure_data_path,'漲幅.feather')).iloc[:, 1:]*1

return_shift_day = 2 # 因子須往前shift幾天 2天表示今天(T)收盤出訊號 T+1收盤買入 T+2就是完整的漲幅
returns_shift = returns.iloc[return_shift_day:,:]
factor1_shift = factor1.iloc[:factor1.shape[0]-return_shift_day,:]
factor2_shift = factor2.iloc[:factor2.shape[0]-return_shift_day,:]
factor3_shift = factor3.iloc[:factor3.shape[0]-return_shift_day,:]

m, n = factor1.shape
o = 3 # 只用3個Rule

factor1_tensor = torch.tensor(factor1_shift.values, dtype=torch.float32).to(device) # 轉換成tensor 並且放到裝置上
factor2_tensor = torch.tensor(factor2_shift.values, dtype=torch.float32).to(device)
factor3_tensor = torch.tensor(factor3_shift.values, dtype=torch.float32).to(device)
returns_tensor = torch.tensor(returns_shift.values, dtype=torch.float32).to(device)
factor1_tensor[torch.isnan(factor1_tensor)] = 0 # NaN 轉成 0
factor2_tensor[torch.isnan(factor2_tensor)] = 0
factor3_tensor[torch.isnan(factor3_tensor)] = 0
returns_tensor[torch.isnan(returns_tensor)] = 0

# 將因子組合成 (m, n, o) 的維度
data = torch.stack([factor1_tensor, factor2_tensor, factor3_tensor], dim=-1)

# 產生Y Label ，>0 表示只想預測是否上漲
labels = (returns_tensor > 0).float().to(device)

# 確認資料維度
print(f"Data shape: {data.shape}")  # 維度是 (m, n, o)
print(f"Labels shape: {labels.shape}")  # 維度是 (m, n)
print(f"Returns shape: {returns_tensor.shape}")  # 維度是 (m, n)


class SharedLinearModel(nn.Module):
    def __init__(self, num_tickers, input_dim):
        super(MultiLinearModel, self).__init__()
        self.models = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_tickers)])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        outputs = [self.models[i](x[:, i, :]).squeeze() for i in range(len(self.models))]
        outputs = torch.stack(outputs, dim=1)
        outputs = self.sigmoid(outputs)  
        return outputs

# 客製化的損失函數
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, outputs, labels, returns):
        bce_loss = self.bce_loss(outputs, labels)
        reward_loss = torch.mean(outputs * returns)
        total_loss = bce_loss - reward_loss  # 極大化reward loss

        if torch.sum(outputs) < 20: # 交易次數過低給於乘法 # 主觀給定
            penalty = 0.2  # 主觀給定
            total_loss += total_loss*penalty
            
        return total_loss
    
num_tickers = n
input_dim = o
model = MultiLinearModel(num_tickers, input_dim).to(device)


if 0:# 列出模型參數
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
# 使用客製化的損失函數
criterion = CustomLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
start_time = time.perf_counter()
print(f"Training model with custom loss function...  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# 訓練模型
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels, returns_tensor)
    loss.backward()

    if 0: # 列出模型梯度
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f'Gradient of {name}: {param.grad}')
            
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        bce_loss = criterion.bce_loss(outputs, labels)
        reward_loss = torch.mean(outputs * returns_tensor)
        expect_return = sum((outputs > 0.5).int() * returns_tensor*100).sum().item()
        print(f'Epoch [{epoch + 1}/{num_epochs}] | random loss : {abs(math.log(1/2)):.6f}| Loss: {loss.item():.6f} | BCE Loss: {bce_loss.item():.6f} | Reward Loss: {reward_loss.item():.6f} | Expect Return: {expect_return:.6f} %')

        
print("Training complete.")
print(f"Time taken: {time.perf_counter() - start_time:.2f} seconds")
if 1: # 列出模型權重
    for i, submodel in enumerate(model.models):
        print(f"Ticker {returns.columns[i]}:")
        print(submodel.weight)
        sigmoid = nn.Sigmoid()
        print("-"*50)

torch.save(model.state_dict(), os.path.join(model_path,'model_weights.pth')) # 儲存模型權重

if 1: # 取得模型權重
    new_model = MultiLinearModel(num_tickers, input_dim).to(device)

    new_model.load_state_dict(torch.load( os.path.join(model_path,'model_weights.pth')))

    def compare_weights(model1, model2):
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            if not torch.equal(param1, param2):
                return False
        return True

    if compare_weights(model, new_model):
        print("權重相同！")
    else:
        print("權重不同！")