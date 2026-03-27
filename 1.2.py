import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 读取数据
# ============================================================
train_data = pd.read_csv(r"D:\shadowww\课程\大三下\模式识别\作业一\Training Data.csv")
test_data  = pd.read_csv(r"D:\shadowww\课程\大三下\模式识别\作业一\Test Data.csv")

x_train = train_data.iloc[:, 0].values
y_train = train_data.iloc[:, 1].values
x_test  = test_data.iloc[:, 0].values
y_test  = test_data.iloc[:, 1].values

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# ============================================================
# 方法1：正弦函数拟合
# 模型: y = A * sin(w * x + phi) + c
# 使用 curve_fit（非线性最小二乘）拟合四个参数
# ============================================================
def sine_model(x, A, w, phi, c):
    return A * np.sin(w * x + phi) + c

params, _ = curve_fit(sine_model, x_train, y_train,
                      p0=[1.0, 1.0, 0.0, 0.0], maxfev=10000)
A_fit, w_fit, phi_fit, c_fit = params

y_train_pred_sine = sine_model(x_train, *params)
y_test_pred_sine  = sine_model(x_test,  *params)
sine_train_mse    = mse(y_train, y_train_pred_sine)
sine_test_mse     = mse(y_test,  y_test_pred_sine)

print("=" * 50)
print("【方法1：正弦函数拟合】")
print(f"  振幅  A   = {A_fit:.4f}")
print(f"  角频率 w   = {w_fit:.4f}  （周期 T = {2*np.pi/w_fit:.4f}）")
print(f"  初相位 phi = {phi_fit:.4f}")
print(f"  偏移   c   = {c_fit:.4f}")
print(f"  训练 MSE   = {sine_train_mse:.4f}")
print(f"  测试 MSE   = {sine_test_mse:.4f}")

# ============================================================
# 方法2：傅里叶级数拟合
# 模型: y = a0 + sum_{k=1}^{K} [a_k*cos(k*w0*x) + b_k*sin(k*w0*x)]
# 这是线性模型，直接用最小二乘法求解
# ============================================================
def fourier_features(x, K, w0):
    """构造傅里叶特征矩阵，shape: (N, 2K+1)"""
    cols = [np.ones(len(x))]
    for k in range(1, K + 1):
        cols.append(np.cos(k * w0 * x))
        cols.append(np.sin(k * w0 * x))
    return np.column_stack(cols)

w0 = 2 * np.pi / 10   # 基频，对应数据总长度 x ∈ [0, 10]

print("\n【方法2：傅里叶级数拟合】")
print(f"  {'K':<6} {'训练MSE':<14} {'测试MSE':<14} {'参数个数'}")
print("  " + "-" * 46)

fourier_results = {}
for K in range(1, 11):
    X_tr = fourier_features(x_train, K, w0)
    X_te = fourier_features(x_test,  K, w0)
    w    = np.linalg.lstsq(X_tr, y_train, rcond=None)[0]
    tr   = mse(y_train, X_tr @ w)
    te   = mse(y_test,  X_te @ w)
    fourier_results[K] = {'w': w, 'train_mse': tr, 'test_mse': te}
    print(f"  K={K:<4} {tr:<14.4f} {te:<14.4f} {2*K+1}")

best_K            = min(fourier_results, key=lambda k: fourier_results[k]['test_mse'])
best_w            = fourier_results[best_K]['w']
fourier_train_mse = fourier_results[best_K]['train_mse']
fourier_test_mse  = fourier_results[best_K]['test_mse']

print("  " + "-" * 46)
print(f"  最优 K   = {best_K}")
print(f"  训练 MSE = {fourier_train_mse:.4f}")
print(f"  测试 MSE = {fourier_test_mse:.4f}")

# ============================================================
# 汇总对比
# ============================================================
print("\n" + "=" * 50)
print("【两种方法汇总对比】")
print(f"  {'方法':<20} {'训练MSE':<14} {'测试MSE':<14} {'参数个数'}")
print("  " + "-" * 54)
print(f"  {'正弦函数拟合':<20} {sine_train_mse:<14.4f} {sine_test_mse:<14.4f} 4")
print(f"  {'傅里叶级数(K='+str(best_K)+')':<20} {fourier_train_mse:<14.4f} {fourier_test_mse:<14.4f} {2*best_K+1}")

# ============================================================
# 可视化：2行2列，共4张子图
# ============================================================
x_line      = np.linspace(0, 10, 500)
y_line_true = np.sin(x_line)
y_line_sine = sine_model(x_line, *params)
y_line_four = fourier_features(x_line, best_K, w0) @ best_w

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- 左上：正弦拟合结果 ---
ax = axes[0, 0]
ax.scatter(x_train, y_train, color='gray',  alpha=0.5, s=20, label='训练数据')
ax.scatter(x_test,  y_test,  color='black', alpha=0.5, s=20, marker='x', label='测试数据')
ax.plot(x_line, y_line_sine, color='red',  linewidth=2,   label='正弦拟合')
ax.plot(x_line, y_line_true, color='blue', linewidth=1.5, linestyle='--', label='sin(x) 真实曲线')
ax.set_title(f'方法1：正弦函数拟合\n训练MSE={sine_train_mse:.4f}  测试MSE={sine_test_mse:.4f}')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

# --- 右上：正弦拟合残差图 ---
ax = axes[0, 1]
ax.scatter(x_train, y_train - y_train_pred_sine, color='gray',  alpha=0.5, s=20, label='训练残差')
ax.scatter(x_test,  y_test  - y_test_pred_sine,  color='black', alpha=0.5, s=20, marker='x', label='测试残差')
ax.axhline(0, color='red', linewidth=1.5, linestyle='--')
ax.set_title('方法1 残差图')
ax.set_xlabel('x')
ax.set_ylabel('残差 = 真实值 - 预测值')
ax.legend()
ax.grid(True, alpha=0.3)

# --- 左下：傅里叶级数拟合结果 ---
ax = axes[1, 0]
ax.scatter(x_train, y_train, color='gray',  alpha=0.5, s=20, label='训练数据')
ax.scatter(x_test,  y_test,  color='black', alpha=0.5, s=20, marker='x', label='测试数据')
ax.plot(x_line, y_line_four, color='orange', linewidth=2,   label=f'傅里叶拟合 K={best_K}')
ax.plot(x_line, y_line_true, color='blue',   linewidth=1.5, linestyle='--', label='sin(x) 真实曲线')
ax.set_title(f'方法2：傅里叶级数拟合（K={best_K}）\n训练MSE={fourier_train_mse:.4f}  测试MSE={fourier_test_mse:.4f}')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

# --- 右下：傅里叶不同K下的MSE变化 ---
ax = axes[1, 1]
K_list         = list(fourier_results.keys())
train_mse_list = [fourier_results[k]['train_mse'] for k in K_list]
test_mse_list  = [fourier_results[k]['test_mse']  for k in K_list]
ax.plot(K_list, train_mse_list, 'o-', color='blue',   label='训练 MSE')
ax.plot(K_list, test_mse_list,  's-', color='red',    label='测试 MSE')
ax.axvline(best_K, color='green', linestyle='--', label=f'最优 K={best_K}')
ax.set_title('方法2：不同阶数 K 下的 MSE')
ax.set_xlabel('傅里叶阶数 K')
ax.set_ylabel('MSE')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('两种非线性拟合方法对比', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(r"D:\shadowww\课程\大三下\模式识别\作业一\method1_vs_method2.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n图像已保存。")
