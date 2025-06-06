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
import csv
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

def calculate_total_return(outputs, returns_tensor):
    return sum((outputs > 0.5).int() * returns_tensor*100).sum().item()

def create_tensor(data, device):
    # 轉換成tensor 並且放到裝置上
    tensor = torch.tensor(data.values, dtype=torch.float32).to(device)
    tensor[torch.isnan(tensor)] = 0 # NaN 轉成 0
    return tensor

def get_device(device = None): # 執行裝置
    ''' 取得裝置'''
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    return device
                
class MultiLinearModel(nn.Module):
    def __init__(self, input_dim):
        super(MultiLinearModel, self).__init__()
        self.models = nn.ModuleDict(dict(
        layer1 = nn.Linear(input_dim, 1),
        ))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.models['layer1'](x)
        outputs = self.sigmoid(x)  
        return outputs
    def normalize_weights(self):
        with torch.no_grad():
            weights =  self.models['layer1'].weight.data
            normalized_weights = weights / weights.sum()
            self.models['layer1'].weight.data = normalized_weights
            
# 客製化的損失函數
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, outputs, labels, returns):
        bce_loss = self.bce_loss(outputs, labels)
        reward_loss = calculate_total_return(outputs, returns)
        total_loss = bce_loss - reward_loss  # 極大化reward loss

        if torch.sum(outputs) < 20: # 交易次數過低給於乘法 # 主觀給定
            penalty = 0.2  # 主觀給定
            total_loss += total_loss*penalty
            
        return total_loss

class TrainingModel():
    def __init__(self,model,data,label,retun,num_epochs = 10000,batch = 32, device = 'cpu',model_path = None):
        self.num_epochs = num_epochs
        self.batch = batch
        self.max_expect_return = -9999
        self.model_path = model_path
        self.device = device
        self.model = model.to(self.device)
        self.data = data
        self.labels = label
        self.returns_tensor = retun
    
    def print_model_parameters(self): # 列出模型權重
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
    def print_model_grad(self): # 列出模型梯度
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.grad)
    def train(self):
        filename = "output.csv"
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)

        if 1:
            # 使用客製化的損失函數
            criterion = CustomLoss()
            self.optimizer = optim.AdamW(self.model.parameters(), lr=0.00001, weight_decay=1e-4)
            start_time = time.perf_counter()
            print(f"Training model with custom loss function...  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if 1:
            step = 0
            step_total = 0
            self.num_epochs = 20000
            for epoch in range(self.num_epochs):
                self.model.train()
                self.optimizer.zero_grad()
                if (step+1)*self.batch > self.data.shape[0]:
                    step = 0
                sub_data = self.data[step*self.batch:(step+1)*self.batch]
                sub_labels = self.labels[step*self.batch:(step+1)*self.batch]
                sub_returns_tensor = self.returns_tensor[step*self.batch:(step+1)*self.batch]
                step = step + 1
                step_total = step_total + 1
                outputs = self.model(sub_data)
                loss = criterion(outputs,  sub_labels, sub_returns_tensor)
                loss.backward()

                if 0: # 列出模型梯度
                    self.print_model_grad()
                        
                self.optimizer.step()
                self.model.normalize_weights()
                if (epoch +1 ) % 1000 ==0 or epoch ==0:
                    self.model.eval()
                    with open(filename, mode='a', newline='') as file:  # 使用'a'模式来追加数据
                        with torch.no_grad():
                            outputs = self.model(self.data)
                            bce_loss = criterion(outputs,  self.labels, self.returns_tensor)
                            reward_loss = torch.mean(outputs * self.returns_tensor)
                            expect_return = calculate_total_return(outputs, self.returns_tensor)
                        writer = csv.writer(file)
                        writer.writerow([epoch,expect_return,self.get_model_parameter()[0].tolist()[0]])
                        print(f'Epoch [{epoch + 1}/{self.num_epochs}] | random loss : {abs(math.log(1/2)):.6f}| Loss: {loss.item():.6f} | BCE Loss: {bce_loss.item():.6f} | Reward Loss: {reward_loss.item():.6f} | Expect Total Return: {expect_return:.4f} % | outputs max: {outputs.max():.2f} min: {outputs.min():.2f} | 買進持有 {self.returns_tensor.sum()*100:.2f}% | 學習到最大報酬{self.max_expect_return:.2f}%')
                    
                    if expect_return > self.max_expect_return: # 報酬率比之前的好就更新紀錄(self.max_expect_return) 並存模型
                        self.max_expect_return = expect_return
                        torch.save(self.model.state_dict(), os.path.join(self.model_path,'model_weights.pth'))
                if  0 and ((epoch + 1) % 1000 == 0 or epoch == (self.num_epochs-1)):
                    self.model.eval()
                    with torch.no_grad():
                        outputs = self.model(self.data)
                        bce_loss = criterion(outputs,  self.labels, self.returns_tensor)
                        reward_loss = torch.mean(outputs * self.returns_tensor)
                        expect_return = calculate_total_return(outputs, self.returns_tensor)
                        if expect_return > self.max_expect_return: # 報酬率比之前的好就更新紀錄(self.max_expect_return) 並存模型
                            self.max_expect_return = expect_return
                            torch.save(self.model.state_dict(), os.path.join(self.model_path,'model_weights.pth'))
                        print(f'Epoch [{epoch + 1}/{self.num_epochs}] | random loss : {abs(math.log(1/2)):.6f}| Loss: {loss.item():.6f} | BCE Loss: {bce_loss.item():.6f} | Reward Loss: {reward_loss.item():.6f} | Expect Total Return: {expect_return:.4f} % | outputs max: {outputs.max():.2f} min: {outputs.min():.2f} | 買進持有 {self.returns_tensor.sum()*100:.2f}% | 學習到最大報酬{self.max_expect_return:.2f}%')
                        if expect_return < 0.1 and self.max_expect_return >0.1:
                            print(f"training stop at {epoch + 1} epoch and step at {step}")
                            break
            print(f"Training complete. | epoch:{epoch = }")
            print(f"Time taken: {time.perf_counter() - start_time:.2f} seconds")
    def get_model_parameter(self):
        result = []
        for name, submodel in self.model.models.items():
            result.append(submodel.weight)
        return result
    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.model_path,'model_weights.pth'))
class BuildTrainingData():
    
    def __init__(self,rule_data_path,rule_file_name_list,return_shift_day = 2,device = 'cpu'):
        self.rule_data_path = rule_data_path
        self.rule_file_name_list = rule_file_name_list
        self.return_shift_day = return_shift_day # 因子須往前shift幾天 2天表示今天(T)收盤出訊號 T+1收盤買入 T+2就是完整的漲幅
        self.device = device
        
    def build_data(self):
        # 自動建立變數 df_本益比小於25 要是.feather
        for filename in self.rule_file_name_list:
            globals()["df_" + filename] = pd.read_feather(os.path.join(self.rule_data_path,filename + '.feather')).iloc[:, 1:]*1
            # 原始程式似 factor1_shift = factor1.iloc[:factor1.shape[0]-return_shift_day,:]
            # df_近5年稅前淨利成長率平均大於0_03_shift
            globals()["df_" + filename + "_shift"] = globals()["df_" + filename].iloc[:globals()["df_" + filename].shape[0]-self.return_shift_day,:]
            globals()["df_" + filename + "_shift_tensor"] = create_tensor(globals()["df_" + filename + "_shift"],device= self.device)
            
        date_list = pd.read_feather(os.path.join(self.rule_data_path,self.rule_file_name_list[0] + '.feather'))['Date']
        ticker_list = pd.read_feather(os.path.join(self.rule_data_path,self.rule_file_name_list[0] + '.feather')).columns[1:]
        
        tensor_list = [globals()["df_" + filename + "_shift_tensor"] for filename in rule_file_name_list]
        self.data_tensor = torch.stack(tensor_list, dim=-1) # 將因子組合成 (m, n, o) 的維度
        returns = pd.read_feather(os.path.join(measure_data_path,'漲幅.feather'))
        # 確認欄位名稱是否一樣，並篩選出交集欄位的資料
        common_columns = list(returns.columns.intersection(ticker_list))
        returns = returns[common_columns]
        returns_shift = returns.iloc[self.return_shift_day:,:]
        self.returns_tensor = create_tensor(returns_shift, self.device)

        self.number_date, self.number_ticker = eval("df_" + filename + "_shift").shape # 用因子來決定 m, n
        self.number_facotr = len(rule_file_name_list) # 只用3個Rule
        assert self.returns_tensor.shape == (self.number_date, self.number_ticker), "Returns shape does not match factor shape"
        
        self.labels_tensor = (self.returns_tensor > 0).float().to(self.device)# 產生Y Label ，>0 表示只想預測是否上漲

        # 確認資料維度
        print(f"Data shape: {self.data_tensor.shape} | type: {type(self.data_tensor)}")  # 維度是 (number_date, number_ticker, number_facotr)
        print(f"Labels shape: {self.labels_tensor.shape} | type: {type(self.labels_tensor)}")  # 維度是 (number_date, number_ticker, number_facotr)
        print(f"Returns shape: {self.returns_tensor.shape} | type: {type(self.returns_tensor)}")  # 維度是 (number_date, number_ticker, number_facotr)
    
    def get_dimession_need(self):
        ''' date , ticker , factor'''
        return self.number_date, self.number_ticker, self.number_facotr
    
    def get_data(self):
        ''' 取得資料 X Y 報酬率''' 
        return self.data_tensor, self.labels_tensor, self.returns_tensor
    
    def get_data_after_resahpe(self):
        ''' 取得資料 X Y 報酬率''' 
        return self.data_tensor.reshape(-1,self.data_tensor.shape[2]), self.labels_tensor.reshape(-1,1), self.returns_tensor.reshape(-1,1)

if __name__ == '__main__':
    device = get_device()
    base_path = os.path.dirname(os.path.abspath(__file__))  # 取得目前檔案的路徑
    measure_data_path = os.path.join(base_path, 'data', 'measure_data')  # 設定因子路徑
    rule_data_path = os.path.join(base_path, 'data', 'rule_data')  # 設定規則路徑
    model_path = os.path.join(base_path, 'model')  # 設定模型路徑

    if 1:
        # rule_file_name_list =  [osrule_file_name_list = .path.splrule_file_name_list = itext(file)[0] for file in os.listdir(rule_data_path) if file.endswith('.feather')]
        rule_file_name_list = ['MACD向上交叉零線', 'MACD向下交叉零線', 'MACD零線以上且下穿信號線', 'MACD零線以下且上穿信號線', '最近一季負債比小於0_25', '本益比小於25', '股價跌出布林通道下軌', '近2年營收成長率平均大於0_1', '近5年稅前淨利成長率平均大於0_03']
        rule_file_name_list = ['本益比小於25', '股價跌出布林通道下軌', '近5年稅前淨利成長率平均大於0_03']

        obj_BuildTrainingData = BuildTrainingData(rule_data_path = rule_data_path,rule_file_name_list = rule_file_name_list,return_shift_day = 2,device= device)
        obj_BuildTrainingData.build_data()
        number_factor = obj_BuildTrainingData.get_dimession_need()[2]
        data,labels,returns = obj_BuildTrainingData.get_data_after_resahpe()
    if 1:

        model = MultiLinearModel(number_factor)
        obj_train = TrainingModel(model = model,data = data,label = labels ,retun = returns,num_epochs = 10000,batch = 32, device = device,model_path = model_path)
        obj_train.train()
        model_weight = obj_train.get_model_parameter()
        print(model_weight)
        print(f"factor : {model_weight}")

    if 0: # 匯入模型
        number_factor = 3
        new_model = MultiLinearModel(number_factor).to(device)
        new_model.load_state_dict(torch.load( os.path.join(model_path,'model_weights.pth')))
        print()
        # 列出模型的參數名稱和數值
        print("Model Parameters:")
        for name, param in new_model.named_parameters():
            print(f"Name: {name}")
            print(f"Value: {param.data}")  # 列出參數的值
            print(f"Shape: {param.shape}")  # 列出參數的形狀
            print("-" * 40)