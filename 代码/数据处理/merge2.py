import pandas as pd
import os

# 文件路径设置
output_folder = "proceed"
merged_file = os.path.join(output_folder, "merge_process.csv")

# 初始化空的 DataFrame 用于合并
merged_data = pd.DataFrame()

# 遍历 proceed 文件夹中的文件
for year in range(1970, 2015):
    proceed_file = os.path.join(output_folder, f"proceed_{year}.csv")

    if os.path.exists(proceed_file):
        # 读取每年的 proceed 文件
        yearly_data = pd.read_csv(proceed_file)

        # 添加一个新列标记年份
        yearly_data["Year"] = year

        # 合并到总的 DataFrame 中
        merged_data = pd.concat([merged_data, yearly_data], ignore_index=True)
    else:
        print(f"File not found: {proceed_file}")

# 将合并后的数据保存为一个文件
merged_data.to_csv(merged_file, index=False)
print(f"All files merged and saved to {merged_file}.")
