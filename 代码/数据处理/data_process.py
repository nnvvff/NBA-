import pandas as pd
import os

# 指定数据文件的路径模板
file_template = r'D:\Dekstop\big_data\basketball\leagues_NBA_{}_per_game_per_game.csv'

# 输出文件保存的文件夹
output_dir = os.getcwd()  # 当前代码运行的文件夹

# 遍历年份从 1970 到 2015
for year in range(1970, 2016):  # 包括2015
    try:
        # 构造当前年份的文件路径
        input_file_path = file_template.format(year)

        # 检查文件是否存在
        if not os.path.exists(input_file_path):
            print(f"文件不存在：{input_file_path}")
            continue

        # 读取文件
        df = pd.read_csv(input_file_path)

        # 删除从第三行开始，第一列值为 'RK' 的行
        df_cleaned = df[~((df.index >= 2) & (df.iloc[:, 0] == 'Rk'))]

        # 重置索引
        df_cleaned.reset_index(drop=True, inplace=True)

        # 保存清理后的文件到代码运行文件夹
        output_file_path = os.path.join(output_dir, f'cleaned_{year}.csv')
        df_cleaned.to_csv(output_file_path, index=False)

        print(f"年份 {year} 的数据处理完成，保存为：{output_file_path}")

    except Exception as e:
        print(f"处理年份 {year} 时发生错误：{e}")

