import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from pylab import mpl
import os

# 设置 Matplotlib 中文支持
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

# 1. 读取数据
file_path = r"E:\merge_process.csv"  # 替换为您的文件路径
data = pd.read_csv(file_path)

# 2. 数据预处理
# 替换 -1 和空值为 NaN 并删除无效行
data.replace(-1, np.nan, inplace=True)
data.dropna(inplace=True)

# 去掉非数值列
non_numeric_cols = ['Rk', 'Player', 'Pos', 'Tm', 'Year']
numeric_data = data.drop(columns=non_numeric_cols)

# 使用文件中 pre_pts 作为下一年的真实数据
numeric_data['Next_Year_PTS'] = numeric_data['pre_pts']
numeric_data.dropna(inplace=True)  # 删除无效行

# 随机抽样（有放回，扩充数据集）
data_sampled = resample(numeric_data, replace=True, n_samples=len(numeric_data) * 2, random_state=42)

# 3. 数据归一化
scaler = MinMaxScaler()
features = data_sampled.drop(columns=['PTS', 'pre_pts', 'Next_Year_PTS'])
target = data_sampled['Next_Year_PTS']

features_scaled = scaler.fit_transform(features)

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# 5. 构建神经网络模型
model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))  # 增加神经元数量
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))  # 输出层

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 6. 模型训练
history = model.fit(X_train, y_train, epochs=300, batch_size=64, validation_split=0.1, shuffle=True, verbose=1)

# 7. 模型评估
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"测试集 MAE: {mae:.3f}")

# 8. 预测下一年
predictions = model.predict(X_test).flatten()

# 9. 计算评价指标
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# 10. 计算准确率（允许误差范围 ±1 分）
threshold = 1.0  # 允许的误差范围
accuracy = np.mean(np.abs(y_test.values - predictions) <= threshold) * 100

print(f"模型评价指标：")
print(f"- MSE: {mse:.3f}")
print(f"- RMSE: {rmse:.3f}")
print(f"- MAE: {mae:.3f}")
print(f"- R²: {r2:.3f}")
print(f"- 准确率（±{threshold} 分误差）: {accuracy:.2f}%")

# 保存预测结果到指定路径
output_path = r"D:\Desktop\predicted_results.csv"  # 替换为可写路径
try:
    output = pd.DataFrame({
        "真实值": y_test.values,
        "预测值": predictions
    })
    output.to_csv(output_path, index=False)
    print(f"预测结果已保存至 {output_path}")
except PermissionError:
    print(f"无法保存文件，请检查路径 {output_path} 是否有写权限或文件是否被占用。")

# 绘制训练和验证损失曲线
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='训练集损失')
plt.plot(history.history['val_loss'], label='验证集损失')
plt.title('模型训练损失曲线')
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.legend()
plt.grid()
plt.show()

# 绘制预测值与真实值的对比图
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test.values, label='真实值', marker='o', linestyle='-', color='blue')
plt.plot(range(len(predictions)), predictions, label='预测值', marker='x', linestyle='--', color='orange')
plt.title('预测值与真实值对比')
plt.xlabel('样本索引')
plt.ylabel('得分')
plt.legend()
plt.grid()
plt.show()