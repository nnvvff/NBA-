import pandas as pd
import os

# 文件路径设置
file_prefix = "cleaned_"
output_folder = "proceed"
start_year = 1970
end_year = 2015

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

for year in range(start_year, end_year):
    # 当前年份文件
    current_file = f"{file_prefix}{year}.csv"
    next_file = f"{file_prefix}{year + 1}.csv"
    output_file = os.path.join(output_folder, f"proceed_{year}.csv")

    if not os.path.exists(current_file) or not os.path.exists(next_file):
        print(f"Skipping missing file: {current_file} or {next_file}")
        continue

    # 加载数据
    current_data = pd.read_csv(current_file)
    next_data = pd.read_csv(next_file)

    # 添加新列 pre_pts 并初始化为空
    current_data["pre_pts"] = None

    # 遍历当前年份文件的每一行（从第二行，包括第二行）
    rows_to_keep = []
    for index, row in current_data.iterrows():
        player_name = row["Player"]
        # 查找下一年是否有相同的 Player
        next_row = next_data[next_data["Player"] == player_name]

        if not next_row.empty:
            # 如果找到，复制下一年的 PTS 到当前年的 pre_pts
            current_data.at[index, "pre_pts"] = next_row.iloc[0]["PTS"]
            rows_to_keep.append(True)
        else:
            # 如果没找到，标记为需要删除
            rows_to_keep.append(False)

    # 删除没有匹配的行
    processed_data = current_data[rows_to_keep]

    # 保存处理后的数据到新的文件
    processed_data.to_csv(output_file, index=False)
    print(f"Processed {current_file}, saved to {output_file}.")
