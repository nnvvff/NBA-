import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName('NBAPlayerStatsAnalysis2').getOrCreate()
df = spark.read.option("header", True).option("inferSchema", True).csv("hdfs://master:9000/usr/local/hadoop/merge_process.csv")#修改
# Load the data
data_path = "hdfs://master:9000/usr/local/hadoop/merge_process.csv"#修改
data = spark.read.csv(data_path, header=True, inferSchema=True)

# NBA球员表现的预测模型
# 创建新特征（例如，每场比赛得分）。
# 删除 PName 列并对分类变量进行编码。
# 将数据分为训练集和测试集。
# 选择一个合适的回归模型（线性回归）。
# 在训练集上训练模型。
# 使用测试集进行预测。
# 计算评估指标（MAE、MSE 和 R-squared）。
# 定义一个函数 predict_points，用于预测特定球员的得分。
# 预测并输出特定球员（例如，Nikola Jokic）在下赛季的得分。
# Create new features



# Index categorical columns
team_indexer = StringIndexer(inputCol="Tm", outputCol="team_index")
pos_indexer = StringIndexer(inputCol="Pos", outputCol="pos_index")

# Assemble features into a feature vector
assembler = VectorAssembler(inputCols=[col for col in data.columns if col not in ['Player', 'Age', 'Tm', 'Pos','Rk','pre_pts']],
                            outputCol="features")

# Define the Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="pre_pts")
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="pre_pts",
    numTrees=80,
    maxDepth=30,
    maxBins=64,
    subsamplingRate=0.8,
    featureSubsetStrategy="sqrt",
    seed=42
)

# Split the data into a training set and a test set
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Create a Pipeline
pipeline_lr = Pipeline(stages=[team_indexer, pos_indexer, assembler, lr])
pipeline_rf = Pipeline(stages=[team_indexer, pos_indexer, assembler, rf])
# Train the model
# 模型训练
model_lr = pipeline_lr.fit(train_data)
model_rf = pipeline_rf.fit(train_data)

# 模型预测
predictions_lr = model_lr.transform(test_data)
predictions_rf = model_rf.transform(test_data)

# 转换预测结果为 Pandas DataFrame
predictions_lr_pd = predictions_lr.select("pre_pts", "prediction").withColumnRenamed("prediction", "lr_prediction").toPandas()
predictions_rf_pd = predictions_rf.select("pre_pts", "prediction").withColumnRenamed("prediction", "rf_prediction").toPandas()

# 合并结果用于比较
predictions_combined = predictions_lr_pd.merge(predictions_rf_pd, on="pre_pts")

# 绘制对比图
plt.figure(figsize=(14, 7))
sns.scatterplot(x=predictions_combined["pre_pts"], y=predictions_combined["lr_prediction"], label="Linear Regression Prediction", alpha=0.6)
sns.scatterplot(x=predictions_combined["pre_pts"], y=predictions_combined["rf_prediction"], label="Random Forest Prediction", alpha=0.6)
sns.lineplot(x=predictions_combined["pre_pts"], y=predictions_combined["pre_pts"], color="red", linestyle="--", label="Actual Points")
plt.xlabel("Actual Points")
plt.ylabel("Predicted Points")
plt.title("Linear Regression vs Random Forest Predictions")
plt.legend()
plt.savefig("/home/hadoop/Final_Ex2/result/images/comparison_predictions.png")

# 模型评估：线性回归
evaluator = RegressionEvaluator(labelCol="pre_pts", predictionCol="prediction", metricName="mae")
mae_lr = evaluator.evaluate(predictions_lr)
mse_lr = evaluator.evaluate(predictions_lr, {evaluator.metricName: "mse"})
r2_lr = evaluator.evaluate(predictions_lr, {evaluator.metricName: "r2"})

# 模型评估：随机森林
mae_rf = evaluator.evaluate(predictions_rf)
mse_rf = evaluator.evaluate(predictions_rf, {evaluator.metricName: "mse"})
r2_rf = evaluator.evaluate(predictions_rf, {evaluator.metricName: "r2"})

# 输出评价指标
print("Linear Regression Metrics:")
print(f"  Mean Absolute Error (MAE): {mae_lr}")
print(f"  Mean Squared Error (MSE): {mse_lr}")
print(f"  R-squared: {r2_lr}")

print("\nRandom Forest Metrics:")
print(f"  Mean Absolute Error (MAE): {mae_rf}")
print(f"  Mean Squared Error (MSE): {mse_rf}")
print(f"  R-squared: {r2_rf}")
