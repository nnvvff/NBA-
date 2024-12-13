import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('NBAPlayerStatsAnalysis2').getOrCreate()
df = spark.read.option("header", True).option("inferSchema", True).csv("hdfs://master:9000/usr/local/hadoop/merged_cleaned_data.csv")
# Load the data
data_path = "hdfs://master:9000/usr/local/hadoop/merged_cleaned_data.csv"
data = spark.read.csv(data_path, header=True, inferSchema=True)
#计算属性间的相似性矩阵
import pandas as pd

# Convert Spark DataFrame to Pandas DataFrame for correlation computation
df = data.toPandas()

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig("/home/hadoop/Final_Ex2/result/images/pearson.png")
