import pandas as pd

# 定义文件路径
input_file =  r"proceed\merge_process.csv" # 替换为你的输入文件路径
output_file = "merged_process.csv"    # 替换为你的输出文件路径

# 加载 CSV 文件
df = pd.read_csv(input_file)

# 检查数据
print("Original DataFrame:")
print(df.info())  # 显示数据结构
print(df.head())  # 显示前几行

# 删除包含空值的行
cleaned_df = df.dropna()

# 检查清洗后的数据
print("\nCleaned DataFrame:")
print(cleaned_df.info())
print(cleaned_df.head())

# 保存清洗后的数据到新文件
cleaned_df.to_csv(output_file, index=False)

print(f"\nCleaned file has been saved to: {output_file}")
