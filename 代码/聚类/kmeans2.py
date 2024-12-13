import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 创建保存路径
output_dir = '/home/hadoop/Final_Ex2/result/images'
os.makedirs(output_dir, exist_ok=True)

# 读取数据
player_stats = pd.read_csv('/home/hadoop/Final_Ex2/merged_cleaned_data.csv')

# 选择需要进行聚类的特征
required_columns = ['fg%', '3p%']

# 检查是否存在 NaN 值并跳过有空值的行
if player_stats[required_columns].isna().any().any():
    print("Data contains missing values. Skipping rows with NaN values.")
    player_stats = player_stats.dropna(subset=required_columns)

# 如果数据为空，停止程序
if player_stats.empty:
    print("No valid data available after removing NaN rows.")
else:
    # 聚类特征
    new_features = player_stats[required_columns]

    # 数据标准化
    scaler = StandardScaler()
    new_scaled_features = scaler.fit_transform(new_features)

    # 使用 K-means 聚类
    new_kmeans = KMeans(n_clusters=3, random_state=42)
    new_clusters = new_kmeans.fit_predict(new_scaled_features)

    # 将聚类结果添加到数据集中
    player_stats['new_cluster'] = new_clusters

    # 可视化聚类结果
    plt.figure(figsize=(14, 7))
    sns.scatterplot(x='fg%', y='3p%', hue='new_cluster', palette='viridis', data=player_stats, s=100)
    plt.title('K-means Clustering of Players based on FG% and 3P%')
    plt.xlabel('Field Goal Percentage')
    plt.ylabel('3-Point Percentage')
    plt.legend(title='Cluster')

    # 保存图像
    plt.savefig(f'{output_dir}/kmeans3.png')
    plt.show()
