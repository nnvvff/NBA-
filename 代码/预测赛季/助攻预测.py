# import os
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import matplotlib.pyplot as plt
# from pylab import mpl

# # 设置 Matplotlib 中文支持
# mpl.rcParams["font.sans-serif"] = ["SimHei"]
# mpl.rcParams["axes.unicode_minus"] = False

# # 读取数据并预处理
# input_dir = r"D:\Desktop\大数据项目\nba_season_processed"
# average_stats = []

# for file_name in sorted(os.listdir(input_dir)):
#     if file_name.endswith("_processed.csv"):
#         try:
#             df = pd.read_csv(os.path.join(input_dir, file_name))
#             df.replace(-1, np.nan, inplace=True)
#             df.fillna(0, inplace=True)

#             season_year = int(file_name.split("_")[0].split("-")[0])
#             weighted_avg_score = (df["得分"] * df["出场"]).sum() / df["出场"].sum()
#             weighted_avg_rebounds = (df["篮板"] * df["出场"]).sum() / df["出场"].sum()
#             weighted_avg_assists = (df["助攻"] * df["出场"]).sum() / df["出场"].sum()
#             avg_fg_percent = df["投篮"].mean()

#             average_stats.append({
#                 "年份": season_year,
#                 "平均得分": weighted_avg_score,
#                 "平均篮板": weighted_avg_rebounds,
#                 "平均助攻": weighted_avg_assists,
#                 "平均命中率": avg_fg_percent
#             })
#         except Exception as e:
#             print(f"Error processing {file_name}: {e}")

# # 转换为 DataFrame
# stats_df = pd.DataFrame(average_stats).set_index("年份")

# # 特征和目标变量
# features = stats_df[["平均篮板", "平均助攻", "平均命中率"]]
# target = stats_df["平均得分"]

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=42)

# # 模型 1: 随机森林回归
# rf_model = RandomForestRegressor(n_estimators=80, random_state=45)
# rf_model.fit(X_train, y_train)
# rf_y_pred_test = rf_model.predict(X_test)

# # 模型 2: 线性回归
# lr_model = LinearRegression()
# lr_model.fit(X_train, y_train)
# lr_y_pred_test = lr_model.predict(X_test)

# # 模型评估
# rf_mae = mean_absolute_error(y_test, rf_y_pred_test)
# rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred_test))
# rf_r2 = r2_score(y_test, rf_y_pred_test)

# lr_mae = mean_absolute_error(y_test, lr_y_pred_test)
# lr_rmse = np.sqrt(mean_squared_error(y_test, lr_y_pred_test))
# lr_r2 = r2_score(y_test, lr_y_pred_test)

# print(f"随机森林 - MAE: {rf_mae:.3f}, RMSE: {rf_rmse:.3f}, R²: {rf_r2:.3f}")
# print(f"线性回归 - MAE: {lr_mae:.3f}, RMSE: {lr_rmse:.3f}, R²: {lr_r2:.3f}")

# # 全历史预测
# rf_y_pred_all = rf_model.predict(features)
# lr_y_pred_all = lr_model.predict(features)

# # 未来预测
# future_years = np.arange(stats_df.index[-1] + 1, 2036)
# future_features = []

# for i, year in enumerate(future_years):
#     weights = np.exp(-(len(future_years) - i) / 2)
#     weighted_avg = (weights * features.iloc[-1] + (1 - weights) * features.mean()).values
#     future_features.append(weighted_avg)

# future_features = pd.DataFrame(future_features, columns=features.columns, index=future_years)

# # 随机森林预测
# rf_future_predictions = rf_model.predict(future_features)

# # 线性回归预测
# lr_future_predictions = lr_model.predict(future_features)

# # 合并数据
# rf_predicted_df = pd.DataFrame({"真实得分": target, "预测得分": rf_y_pred_all}, index=stats_df.index)
# rf_future_df = pd.DataFrame({"预测得分": rf_future_predictions + 1}, index=future_years)

# lr_predicted_df = pd.DataFrame({"真实得分": target, "预测得分": lr_y_pred_all}, index=stats_df.index)
# lr_future_df = pd.DataFrame({"预测得分": lr_future_predictions}, index=future_years)

# # 绘制对比图
# plt.figure(figsize=(14, 7))

# # 随机森林结果
# plt.plot(rf_predicted_df.index, rf_predicted_df["真实得分"], label="真实得分")
# plt.plot(rf_predicted_df.index, rf_predicted_df["预测得分"], label="随机森林历史预测得分")
# plt.plot(rf_future_df.index, rf_future_df["预测得分"], label="随机森林未来预测得分")

# # 线性回归结果
# plt.plot(lr_predicted_df.index, lr_predicted_df["预测得分"], label="线性回归历史预测得分")
# plt.plot(lr_future_df.index, lr_future_df["预测得分"], label="线性回归未来预测得分")

# plt.title("NBA 历史与未来平均得分预测（随机森林 vs 线性回归）")
# plt.xlabel("年份")
# plt.ylabel("得分")
# plt.grid()
# plt.legend()

# # 添加子图，展示1990 - 2010年的数据
# ax = plt.axes([0.3, 0.15, 0.3, 0.3])  # [left, bottom, width, height]，设置子图位置和大小，可根据需求调整

# # 筛选1990 - 2010年的数据用于子图绘制
# rf_predicted_df_sub = rf_predicted_df.loc[(rf_predicted_df.index >= 1990) & (rf_predicted_df.index <= 2010)]
# lr_predicted_df_sub = lr_predicted_df.loc[(lr_predicted_df.index >= 1990) & (lr_predicted_df.index <= 2010)]

# # 设置子图横坐标间隔为5年显示
# ax.set_xticks(np.arange(1990, 2010 + 1, 5))

# # 随机森林结果（子图）
# ax.plot(rf_predicted_df_sub.index.astype(int), rf_predicted_df_sub["真实得分"], label="真实得分")
# ax.plot(rf_predicted_df_sub.index.astype(int), rf_predicted_df_sub["预测得分"], label="随机森林历史预测得分")
# # 这里假设未来预测部分在子图中暂不展示，如果有需要可类似上面筛选逻辑处理未来预测数据后绘制，此处先注释掉
# # ax.plot(rf_future_df_sub.index.astype(int), rf_future_df_sub["预测得分"], label="随机森林未来预测得分", linestyle="-.", marker="*")

# # 线性回归结果（子图）
# ax.plot(lr_predicted_df_sub.index.astype(int), lr_predicted_df_sub["预测得分"], label="线性回归历史预测得分")
# # 同样注释掉线性回归未来预测得分在子图中的绘制，如有需要可按需处理
# # ax.plot(lr_future_df_sub.index.astype(int), lr_future_df_sub["预测得分"], label="线性回归未来预测得分", linestyle=":", marker="d")

# ax.set_title("")
# ax.set_xlabel("年份")
# ax.set_ylabel("得分")
# ax.grid()
# ax.legend()

# plt.show()

# # 打印最终评价结果对比
# print("\n模型评价对比：")
# results = pd.DataFrame({
#     "模型": ["随机森林", "线性回归"],
#     "MAE": [rf_mae, lr_mae],
#     "RMSE": [rf_rmse, lr_rmse],
#     "R²": [rf_r2, lr_r2]
# })
# print(results)

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pylab import mpl

# 设置 Matplotlib 中文支持
mpl.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
mpl.rcParams["axes.unicode_minus"] = False

# 读取数据并预处理
input_dir = r"/home/hadoop/Final_Ex2/nba_season_processed"
average_stats = []

for file_name in sorted(os.listdir(input_dir)):
    if file_name.endswith("_processed.csv"):
        try:
            df = pd.read_csv(os.path.join(input_dir, file_name))
            df.replace(-1, np.nan, inplace=True)
            df.fillna(0, inplace=True)

            season_year = int(file_name.split("_")[0].split("-")[0])
            weighted_avg_assists = (df["助攻"] * df["出场"]).sum() / df["出场"].sum()
            weighted_avg_rebounds = (df["篮板"] * df["出场"]).sum() / df["出场"].sum()
            avg_fg_percent = df["投篮"].mean()

            average_stats.append({
                "年份": season_year,
                "平均助攻": weighted_avg_assists,
                "平均篮板": weighted_avg_rebounds,
                "平均命中率": avg_fg_percent
            })
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# 转换为 DataFrame
stats_df = pd.DataFrame(average_stats).set_index("年份")

# 特征和目标变量
features = stats_df[["平均篮板", "平均命中率"]]
target = stats_df["平均助攻"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=42)

# 模型 1: 随机森林回归
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred_test = rf_model.predict(X_test)

# 模型 2: 线性回归
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_y_pred_test = lr_model.predict(X_test)

# 模型评估
rf_mae = mean_absolute_error(y_test, rf_y_pred_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred_test))
rf_r2 = r2_score(y_test, rf_y_pred_test)

lr_mae = mean_absolute_error(y_test, lr_y_pred_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_y_pred_test))
lr_r2 = r2_score(y_test, lr_y_pred_test)

print(f"随机森林 - MAE: {rf_mae:.3f}, RMSE: {rf_rmse:.3f}, R²: {rf_r2:.3f}")
print(f"线性回归 - MAE: {lr_mae:.3f}, RMSE: {lr_rmse:.3f}, R²: {lr_r2:.3f}")

# 全历史预测
rf_y_pred_all = rf_model.predict(features)
lr_y_pred_all = lr_model.predict(features)

# 未来预测
future_years = np.arange(stats_df.index[-1] + 1, 2036)
future_features = pd.DataFrame({
    "平均篮板": [features["平均篮板"].mean()] * len(future_years),
    "平均命中率": [features["平均命中率"].mean()] * len(future_years)
}, index=future_years)

# 随机森林预测
rf_future_predictions = rf_model.predict(future_features)

# 线性回归预测
lr_future_predictions = lr_model.predict(future_features)

# 绘制对比图
plt.figure(figsize=(14, 7))

# 随机森林结果
plt.plot(stats_df.index, target, label="真实助攻")
plt.plot(stats_df.index, rf_y_pred_all, label="随机森林历史预测助攻")
plt.plot(future_years, rf_future_predictions, label="随机森林未来预测助攻")

# 线性回归结果
plt.plot(stats_df.index, lr_y_pred_all, label="线性回归历史预测助攻")
plt.plot(future_years, lr_future_predictions, label="线性回归未来预测助攻")

plt.title("NBA 历史与未来平均助攻预测（随机森林 vs 线性回归）")
plt.xlabel("年份")
plt.ylabel("平均助攻")
plt.grid()
plt.legend()
plt.savefig("/home/hadoop/Final_Ex2/result/images/助攻预测.png")

