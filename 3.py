import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')

# ================== 参数设置 ==================
LOOK_BACK = 24  # 使用过去24小时的数据
BATCH_SIZE = 128
EPOCHS = 50  # 可适当增加，配合早停
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ================== 1. 数据加载 ==================
train_path = r"E:\各科作业\模式识别\作业3\archive\LSTM-Multivariate_pollution.csv"
test_path = r"E:\各科作业\模式识别\作业3\archive\pollution_test_data1.csv"

train_raw = pd.read_csv(train_path, parse_dates=['date'])
test_raw = pd.read_csv(test_path)  # 测试集没有date列

# 给测试集添加 pollution 占位列（预处理函数需要，但值不重要，会被覆盖）
if 'pollution' not in test_raw.columns:
    test_raw['pollution'] = 0.0

print("训练集形状:", train_raw.shape)
print("测试集形状:", test_raw.shape)


# ================== 2. 改进后的预处理（包含 pollution 作为特征） ==================
def preprocess_data_v2(df, train=True, scaler=None, ohe=None, y_scaler=None):
    """
    特征列表包含: pollution, dew, temp, press, wnd_spd, snow, rain
    风向独热编码 → 合并 → 标准化全部特征（包括 pollution）
    """
    input_features = ['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']
    wnd_dir = df['wnd_dir'].values.reshape(-1, 1)

    # 风向独热编码
    if ohe is None:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit(wnd_dir)
    wnd_encoded = ohe.transform(wnd_dir)

    # 提取所有输入特征（含 pollution）
    data = df[input_features].values  # shape: (N, 7)
    X_data = np.hstack([data, wnd_encoded])

    # 标准化所有特征
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data)
    else:
        X_scaled = scaler.transform(X_data)

    if train:
        # 目标值：原始污染值，单独标准化（用于损失计算）
        y = df['pollution'].values.reshape(-1, 1)
        if y_scaler is None:
            y_scaler = StandardScaler()
            y_scaled = y_scaler.fit_transform(y)
        else:
            y_scaled = y_scaler.transform(y)
        return X_scaled, y_scaled, scaler, ohe, y_scaler
    else:
        return X_scaled, None, scaler, ohe, y_scaler


# 预处理训练集
X_train_full, y_train_full, scaler, ohe, y_scaler = preprocess_data_v2(train_raw, train=True)
print("训练集特征维度:", X_train_full.shape)
print("训练集目标维度:", y_train_full.shape)

# 预处理测试集
X_test, _, _, _, _ = preprocess_data_v2(test_raw, train=False, scaler=scaler, ohe=ohe, y_scaler=y_scaler)
print("测试集特征维度:", X_test.shape)


# ================== 3. 构造时间序列（含污染自回归） ==================
def create_sequences(features, target, look_back):
    X, y = [], []
    for i in range(look_back, len(features)):
        X.append(features[i - look_back:i])  # 窗口内已包含历史 pollution
        y.append(target[i])
    return np.array(X), np.array(y)


X_seq, y_seq = create_sequences(X_train_full, y_train_full, LOOK_BACK)
print("序列数据形状:", X_seq.shape, y_seq.shape)

# 训练/验证集划分（保持时间顺序）
X_train, X_val, y_train, y_val = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)

# ================== 4. 构建 LSTM 模型 ==================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOK_BACK, X_train.shape[2])),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

# 使用 Adam 优化器，初始学习率稍大，后期通过 ReduceLROnPlateau 降低
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
model.summary()

# 回调函数：早停 + 学习率衰减
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# 绘制训练曲线
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()

# ================== 5. 测试集预测（递归自回归策略） ==================
# 初始窗口：训练集最后 LOOK_BACK 个样本的特征（含 pollution 列）
current_window = X_train_full[-LOOK_BACK:].copy()
predictions = []

for i in range(len(X_test)):
    window = np.expand_dims(current_window[-LOOK_BACK:], axis=0)
    pred_scaled = model.predict(window, verbose=0)[0, 0]
    predictions.append(pred_scaled)

    # 构造下一时刻的特征向量：取测试集第 i 样本的特征，并将标准化后的预测污染值填到第一列
    next_feat = X_test[i].copy()
    next_feat[0] = pred_scaled  # 特征的第 0 列是 pollution
    current_window = np.vstack([current_window, next_feat])

# 反标准化得到真实污染值
predictions_real = y_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# 保存结果
test_results = test_raw.copy()
test_results['predicted_pollution'] = predictions_real
# 测试集中的原始 pollution 列是占位无意义的，删除它，保留气象特征和预测值
test_results.drop(columns=['pollution'], inplace=True)
test_results.to_csv('test_predictions_improved.csv', index=False)
print("预测结果已保存至 test_predictions_improved.csv")

# 显示前10行预测值与真实值（如果你有测试集的真实污染，这里可以对比，目前只能显示预测）
print("前10行预测结果：")
print(test_results.head(10))
