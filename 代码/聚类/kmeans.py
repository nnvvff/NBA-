import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 读取处理后的数据
player_stats = pd.read_csv('/home/hadoop/Final_Ex2/merged_cleaned_data.csv')

# 选择特征进行聚类
features = player_stats[['PTS', 'ast', 'reb']]

# 数据标准化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 使用 K-means 进行聚类
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# 将聚类结果添加到数据集中
player_stats['cluster'] = clusters

# 可视化聚类结果
plt.figure(figsize=(14, 7))
sns.scatterplot(x='PTS', y='reb', hue='cluster', palette='viridis', data=player_stats, s=100)
plt.title('K-means Clustering of Players based on Points and Rebounds')
plt.xlabel('Average Points')
plt.ylabel('Average Rebounds')
plt.legend(title='Cluster')
plt.show()
plt.savefig('/home/hadoop/Final_Ex2/result/images/kmeans1.png')
plt.figure(figsize=(14, 7))
sns.scatterplot(x='PTS', y='ast', hue='cluster', palette='viridis', data=player_stats, s=100)
plt.title('K-means Clustering of Players based on Points and Assists')
plt.xlabel('Average Points')
plt.ylabel('Average Assists')
plt.legend(title='Cluster')
plt.savefig('/home/hadoop/Final_Ex2/result/images/kmeans2.png')
plt.show()

