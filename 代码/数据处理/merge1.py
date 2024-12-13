import pandas as pd
import os

# 设置文件夹路径，假设这些文件与代码运行文件夹相同
input_dir = os.getcwd()  # 当前代码运行目录
output_file = os.path.join(input_dir, 'merged_cleaned_data.csv')  # 合并后的文件路径

# 查找所有需要合并的文件
files_to_merge = [f for f in os.listdir(input_dir) if f.startswith("cleaned_") and f.endswith(".csv")]

# 初始化一个空列表，用于存放所有数据
data_frames = []

# 遍历所有文件并读取
for file_name in sorted(files_to_merge):  # 按年份顺序合并
    file_path = os.path.join(input_dir, file_name)
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 添加一个年份列，从文件名中提取年份
        year = file_name.split("_")[1].split(".")[0]  # 提取 "cleaned_1977.csv" 中的 "1977"
        df["year"] = int(year)

        # 将数据加入列表
        data_frames.append(df)
        print(f"已读取文件: {file_name}")
    except Exception as e:
        print(f"读取文件 {file_name} 时发生错误: {e}")

# 合并所有数据
if data_frames:
    merged_data = pd.concat(data_frames, ignore_index=True)
    # 保存合并后的文件
    merged_data.to_csv(output_file, index=False)
    print(f"所有文件已合并并保存为: {output_file}")
else:
    print("没有找到需要合并的文件。")
