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



if __name__ == '__main__':
    now = datetime.now()
    print(f"-------------{now.strftime('%Y-%m-%d %H:%M:%S-------------')}")

    args = Parser().parse()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Torch device: {}".format(args.device))

    model = KoopmanNetwork(args.dim_dyn_sys, args.koopman_obs_dim, args.koopman_hidden_dim, device='cpu')
    model.load_state_dict(torch.load('my_results/Lorenz63_model_diag_mat.pth', map_location=torch.device('cpu')))
    model.eval().cpu()


    # 初始条件
    x0 = np.array([-8.0, 8.0, 27.0])
    delta_x = 0.001  # 微小扰动


    # 计算原始轨迹
    sol = L63_with_init(x0, args.sigma, args.rho, args.beta)
    trajectory = torch.from_numpy(np.array(sol.y)).type(torch.FloatTensor)

    # 计算扰动后的轨迹
    x0_perturbed = x0.copy()
    x0_perturbed[0] += delta_x  # 在第0维添加微小扰动
    sol_perturbed = L63_with_init(x0_perturbed, args.sigma, args.rho, args.beta)
    trajectory_perturbed = torch.from_numpy(np.array(sol_perturbed.y)).type(torch.FloatTensor)

    # 预测轨迹
    trajectory_pred = torch.zeros((3, 1000))
    trajectory_pred[:, 0] = torch.from_numpy(x0)
    trajectory_pred_perturbed = torch.zeros((3, 1000))
    trajectory_pred_perturbed[:, 0] = torch.from_numpy(x0_perturbed)

    y_pred = torch.zeros((2, 12, 1000))
    y_pred_perturbed = torch.zeros((2, 12, 1000))

    x = trajectory_pred[:, 0]
    x_perturbed = trajectory_pred_perturbed[:, 0]

    with torch.no_grad():
        for i in range(999):
            y, y_next, x_next = model(x.unsqueeze(dim=0))
            y = y.squeeze()
            y_next = y_next.squeeze()
            x_next = x_next.squeeze()
            y_pred[0, :, i] = y
            y_pred[1, :, i] = y_next
            trajectory_pred[:, i + 1] = x_next
            x = x_next
            #x = trajectory[:, i + 1]

            # 扰动轨迹的预测
            y_perturbed, y_next_perturbed, x_next_perturbed = model(x_perturbed.unsqueeze(dim=0))
            y_perturbed = y_perturbed.squeeze()
            y_next_perturbed = y_next_perturbed.squeeze()
            x_next_perturbed = x_next_perturbed.squeeze()
            y_pred_perturbed[0, :, i] = y_perturbed
            y_pred_perturbed[1, :, i] = y_next_perturbed
            trajectory_pred_perturbed[:, i + 1] = x_next_perturbed
            x_perturbed = x_next_perturbed
            #x_perturbed = trajectory_perturbed[:, i + 1]

    # 绘制原始轨迹与预测轨迹的对比图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制预测的轨迹
    ax.plot(trajectory_pred[0, 350:1000].numpy(), trajectory_pred[1, 350:1000].numpy(), trajectory_pred[2, 350:1000].numpy(),
            color='r', alpha=0.6, label='Predicted Trajectory by Koopman Model')

    # 绘制扰动后的预测轨迹
    #ax.plot(trajectory_pred_perturbed[0, :].numpy(), trajectory_pred_perturbed[1, :].numpy(),
    #        trajectory_pred_perturbed[2, :].numpy(),
    #        color='orange', alpha=0.6, label='Predicted Perturbed Trajectory by Koopman Model')

    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Comparison of True and Predicted Lorenz63 Trajectories (With Perturbation)')

    # 添加图例
    ax.legend()

    # 显示图像
    plt.show()

    # 绘制x_perturbed - x 的变化过程
    deltax = trajectory_pred_perturbed - trajectory_pred
    # 计算每个时间步的模值（Euclidean norm）
    delta_norm = torch.norm(deltax, dim=0)
    # 对模值取对数，准备进行线性拟合
    log_delta_norm = torch.log(delta_norm)


    # 绘制误差的三维轨迹图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制误差轨迹
    ax.plot(deltax[0, :].numpy(), deltax[1, :].numpy(), deltax[2, :].numpy(), color='r', label='Δx')

    # 设置标签和标题
    ax.set_xlabel('Δx')
    ax.set_ylabel('Δy')
    ax.set_zlabel('Δz')
    ax.set_title('3D Trajectory of Prediction Error Δx')

    # 添加图例
    ax.legend()

    # 显示图像
    plt.show()
    # 选择指数增长的时间区间
    start_idx = 0
    end_idx = 750

    # 拟合选定区间的数据
    slope, intercept, r_value, p_value, std_err = linregress(np.arange(start_idx, end_idx),
                                                             log_delta_norm[start_idx:end_idx].numpy())

    # 绘制原始模值的对数变化
    plt.figure(figsize=(12, 8))
    plt.plot(log_delta_norm[:1000].numpy(), label='log(||Δx||)', color='r')
    plt.plot(np.arange(start_idx, end_idx), slope * np.arange(start_idx, end_idx) + intercept,
             label=f'Linear Fit: λ ≈ {100 * slope:.4f}', color='b')
    plt.xlabel('Time step')
    plt.ylabel('log(||Δx||)')
    plt.title('Log of Norm of Difference with Lyapunov Exponent Fit')
    plt.legend()
    plt.show()

    # 打印拟合得到的Lyapunov指数
    print(f"拟合得到的Lyapunov指数 λ ≈ {100 * slope:.4f}")




    delta_y = y_pred_perturbed[0, :, :] - y_pred[0, :, :]

    log_delta_y = torch.log(torch.abs(delta_y) + 1e-10)

    # 计算每个时间步的模值（Euclidean norm）
    delta_y_norm = torch.norm(delta_y, dim=0)
    log_delta_y_norm = torch.log(delta_y_norm)


    delta_y_next = y_pred_perturbed[1, :, :] - y_pred[1, :, :]

    # 计算每个时间步的模值（Euclidean norm）
    delta_y_next_norm = torch.norm(delta_y_next, dim=0)
    log_delta_y_next_norm = torch.log(delta_y_next_norm)


    # 绘制两者的模值变化曲线
    plt.figure(figsize=(12, 8))
    plt.plot(log_delta_y_norm.numpy(), label='||Δy|| at step i', color='r')
    plt.plot(log_delta_y_next_norm.numpy(), label='||Δy_next|| at step i+1', color='b')
    plt.xlabel('Time step')
    plt.ylabel('Norm of Difference')
    plt.title('Comparison of ||Δy|| and ||Δy_next||')
    plt.legend()
    plt.show()

    print(f"-----------------------end-----------------------")