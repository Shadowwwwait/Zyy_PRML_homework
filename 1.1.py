"""
第一问：线性拟合
分别使用最小二乘法、梯度下降法(GD)和牛顿法对数据进行线性拟合，
观察训练误差与测试误差。

数据路径:
    训练数据: E:\各科作业\模式识别\作业1\Training Data.csv
    测试数据: E:\各科作业\模式识别\作业1\Test Data.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================== 1. 数据加载 ====================
train_path = r'E:\各科作业\模式识别\作业1\Training Data.csv'
test_path = r'E:\各科作业\模式识别\作业1\Test Data.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 提取第一列作为 x，第二列作为 y
x_train = train_df.iloc[:, 0].values.reshape(-1, 1)  # (100, 1)
y_train = train_df.iloc[:, 1].values.reshape(-1, 1)  # (100, 1)
x_test = test_df.iloc[:, 0].values.reshape(-1, 1)    # (100, 1)
y_test = test_df.iloc[:, 1].values.reshape(-1, 1)    # (100, 1)

n_train = len(x_train)
n_test = len(x_test)

# 构造增广矩阵 X = [x, 1]，用于拟合 y = w*x + b
X_train = np.hstack([x_train, np.ones((n_train, 1))])  # (100, 2)
X_test = np.hstack([x_test, np.ones((n_test, 1))])      # (100, 2)

print(f"训练集样本数: {n_train}, 测试集样本数: {n_test}")
print(f"x 范围: [{x_train.min():.3f}, {x_train.max():.3f}]")
print(f"y 范围: [{y_train.min():.3f}, {y_train.max():.3f}]")


# ==================== 2. 误差计算函数 ====================
def compute_mse(y_true, y_pred):
    """计算均方误差 (Mean Squared Error)"""
    return np.mean((y_true - y_pred) ** 2)


# ==================== 3. 最小二乘法 (Least Squares) ====================
"""
原理: 对于线性模型 y = X * theta，最小二乘的解析解为
      theta = (X^T * X)^(-1) * X^T * y
      该解直接最小化残差平方和，无需迭代。
"""
theta_ls = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
w_ls, b_ls = theta_ls[0, 0], theta_ls[1, 0]

y_pred_train_ls = X_train @ theta_ls
y_pred_test_ls = X_test @ theta_ls

mse_train_ls = compute_mse(y_train, y_pred_train_ls)
mse_test_ls = compute_mse(y_test, y_pred_test_ls)

print("\n" + "=" * 60)
print("【最小二乘法 (Least Squares) — 解析解】")
print(f"拟合直线: y = {w_ls:.6f} * x + {b_ls:.6f}")
print(f"训练 MSE: {mse_train_ls:.6f}")
print(f"测试 MSE: {mse_test_ls:.6f}")


# ==================== 4. 梯度下降法 (Gradient Descent) ====================
"""
原理: 通过迭代沿着损失函数梯度的反方向更新参数。
      损失函数: J = (1/2n) * sum((y_pred - y)^2)
      梯度:     grad = (1/n) * X^T * (X*theta - y)
      更新:     theta := theta - lr * grad
"""
lr = 0.01          # 学习率 (learning rate)
epochs = 10000     # 迭代轮数
theta_gd = np.zeros((2, 1))  # 初始化参数 [w; b] = [0; 0]
loss_history = []  # 记录每轮损失，用于绘制收敛曲线

for epoch in range(epochs):
    y_pred = X_train @ theta_gd
    error = y_pred - y_train
    gradient = (X_train.T @ error) / n_train
    theta_gd -= lr * gradient

    loss = compute_mse(y_train, y_pred)
    loss_history.append(loss)

    if (epoch + 1) % 2000 == 0:
        print(f"  Epoch {epoch+1:5d}/{epochs}, MSE: {loss:.6f}")

w_gd, b_gd = theta_gd[0, 0], theta_gd[1, 0]
y_pred_train_gd = X_train @ theta_gd
y_pred_test_gd = X_test @ theta_gd

mse_train_gd = compute_mse(y_train, y_pred_train_gd)
mse_test_gd = compute_mse(y_test, y_pred_test_gd)

print("\n【梯度下降法 (Gradient Descent) — 迭代优化】")
print(f"拟合直线: y = {w_gd:.6f} * x + {b_gd:.6f}")
print(f"训练 MSE: {mse_train_gd:.6f}")
print(f"测试 MSE: {mse_test_gd:.6f}")


# ==================== 5. 牛顿法 (Newton's Method) ====================
"""
原理: 利用二阶导数 (Hessian矩阵) 信息加速收敛。
      对于线性回归，Hessian 矩阵 H = (1/n) * X^T * X 是常数。
      更新公式: theta := theta - H^(-1) * grad
      由于目标函数是二次的，牛顿法理论上一步即可收敛到最优解。
"""
H = (X_train.T @ X_train) / n_train
H_inv = np.linalg.inv(H)

theta_newton = np.zeros((2, 1))

print("\n【牛顿法 (Newton\'s Method) — 二阶优化】")
for step in range(10):
    y_pred = X_train @ theta_newton
    error = y_pred - y_train
    gradient = (X_train.T @ error) / n_train
    theta_newton -= H_inv @ gradient

    loss = compute_mse(y_train, y_pred)
    print(f"  Step {step+1}, MSE: {loss:.6f}")
    if step > 0 and abs(loss - compute_mse(y_train, X_train @ theta_newton)) < 1e-10:
        break  # 已收敛

w_nt, b_nt = theta_newton[0, 0], theta_newton[1, 0]
y_pred_train_nt = X_train @ theta_newton
y_pred_test_nt = X_test @ theta_newton

mse_train_nt = compute_mse(y_train, y_pred_train_nt)
mse_test_nt = compute_mse(y_test, y_pred_test_nt)

print(f"拟合直线: y = {w_nt:.6f} * x + {b_nt:.6f}")
print(f"训练 MSE: {mse_train_nt:.6f}")
print(f"测试 MSE: {mse_test_nt:.6f}")


# ==================== 6. 结果汇总 ====================
print("\n" + "=" * 60)
print("【三种方法结果汇总】")
print(f"{'方法':<20} {'训练 MSE':<15} {'测试 MSE':<15}")
print("-" * 60)
print(f"{'最小二乘法':<20} {mse_train_ls:<15.6f} {mse_test_ls:<15.6f}")
print(f"{'梯度下降法':<20} {mse_train_gd:<15.6f} {mse_test_gd:<15.6f}")
print(f"{'牛顿法':<20} {mse_train_nt:<15.6f} {mse_test_nt:<15.6f}")
print("=" * 60)

print("\n【分析】")
print("三种方法得到的参数几乎完全相同，这是因为线性回归的损失函数是凸函数，")
print("存在唯一的全局最优解。最小二乘法直接给出解析解；梯度下降通过足够多轮")
print("迭代收敛到该解；牛顿法利用二阶信息，仅需2-3步即可收敛。")
print(f"\n然而，线性模型的测试 MSE 高达 {mse_test_ls:.3f}，拟合效果较差，")
print("说明数据本身并非线性关系，需要引入非线性模型。")


# ==================== 7. 可视化 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) 原始数据分布
ax = axes[0, 0]
ax.scatter(x_train, y_train, c='blue', alpha=0.5, s=40, label='Training Data')
ax.scatter(x_test, y_test, c='red', alpha=0.5, s=40, label='Test Data')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Original Data Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# (b) 三种方法拟合直线对比
ax = axes[0, 1]
x_line = np.linspace(0, 10, 200).reshape(-1, 1)
X_line = np.hstack([x_line, np.ones((200, 1))])

ax.scatter(x_train, y_train, c='blue', alpha=0.3, s=30, label='Training Data')
ax.plot(x_line, X_line @ theta_ls, 'r-', linewidth=2.5, label='Least Squares')
ax.plot(x_line, X_line @ theta_gd, 'g--', linewidth=2, label='Gradient Descent')
ax.plot(x_line, X_line @ theta_newton, 'm:', linewidth=2.5, label="Newton's Method")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Linear Fitting Comparison (Three Methods)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) 梯度下降收敛曲线
ax = axes[1, 0]
ax.plot(loss_history, 'b-', linewidth=1)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE')
ax.set_title('Gradient Descent Convergence Curve')
ax.grid(True, alpha=0.3)

# (d) 残差分布 (以最小二乘法为例)
ax = axes[1, 1]
residual_train = (y_train - y_pred_train_ls).flatten()
residual_test = (y_test - y_pred_test_ls).flatten()
ax.hist(residual_train, bins=15, alpha=0.5, color='blue', label='Train Residual')
ax.hist(residual_test, bins=15, alpha=0.5, color='red', label='Test Residual')
ax.set_xlabel('Residual')
ax.set_ylabel('Frequency')
ax.set_title('Residual Distribution (Least Squares)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('第一问_线性拟合结果.png', dpi=200)
plt.show()
