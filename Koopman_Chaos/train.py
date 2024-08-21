import torch

import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import nolds
from scipy.stats import linregress
from datetime import datetime
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import mean_squared_error

from src.args import Parser
from src.Lorenz import L63, L63_with_init
from src.data_engr import load_data
from src.model.Koopman_network import KoopmanNetwork
from src.model.learn import model_train, model_test

args = Parser().parse()

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Torch device: {}".format(args.device))
# 使用 L63 函数生成数据
train_data = []
test_data = []

# 生成训练数据
for _ in range(64):
    sol = L63(sigma=10, rho=28, beta=8/3)
    train_data.append(sol.y[:, :1000])  # 每条轨迹长度为1000

# 生成测试数据
for _ in range(10):
    sol = L63(sigma=10, rho=28, beta=8/3)
    test_data.append(sol.y[:, :1000])  # 每条轨迹长度为1000

train_data = np.array(train_data)
test_data = np.array(test_data)

# 使用load_data 函数将数据加载为 DataLoader
train_loader, test_loader = load_data(train_data, args)

# 初始化模型
model = KoopmanNetwork(input_dim=3, output_dim=12, hidden_dim=24, device=args.device).to(args.device)

# 定义优化器和学习率调度
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-7)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    train_loss = model_train(model, train_loader, optimizer, args.device)
    test_loss = model_test(model, test_loader, args.device)
    scheduler.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# 保存模型
torch.save(model.state_dict(), 'my_results/trained_koopman_model2.pth')
