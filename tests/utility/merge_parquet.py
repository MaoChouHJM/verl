import pandas as pd
import pyarrow.parquet as pq
import sys

# 定义文件路径
file_path1 = sys.argv[1]
file_path2 = sys.argv[2]
output_file_path = './merged.parquet'

try:
    # 读取第一个 Parquet 文件
    df_file1 = pd.read_parquet(file_path1)
    print(f"\n成功读取 {file_path1}。")
    print("文件1的摘要信息:")
    df_file1.info()
    print("\n文件1的前5行数据:")
    print(df_file1.head())

    # 读取第二个 Parquet 文件
    df_file2 = pd.read_parquet(file_path2)
    print(f"\n成功读取 {file_path2}。")
    print("文件2的摘要信息:")
    df_file2.info()
    print("\n文件2的前5行数据:")
    print(df_file2.head())

    # 从每个文件中选择前4行
    selected_rows_df1 = df_file1.head(1)
    selected_rows_df2 = df_file2.head(7)

    print("\n文件1中选择的前4行:")
    print(selected_rows_df1)
    print("\n文件2中选择的前4行:")
    print(selected_rows_df2)

    # 合并两个 DataFrame
    # 由于 schema 不同，直接 concat 会保留所有列，并在缺失值处填充 NaN
    merged_df = pd.concat([selected_rows_df1, selected_rows_df2], ignore_index=True)
    #merged_df = pd.concat([selected_rows_df1], ignore_index=True)

    print("\n合并后的 DataFrame 摘要信息:")
    merged_df.info()
    print("\n合并后的 DataFrame 前10行数据:")
    print(merged_df.head(10))

    # 将合并后的 DataFrame 存储到新的 Parquet 文件中
    merged_df.to_parquet(output_file_path)
    print(f"\n合并后的数据已成功存储到 {output_file_path}。")

    # 验证新文件是否可读
    df_verify = pd.read_parquet(output_file_path)
    print(f"\n验证 {output_file_path} 的内容:")
    print(df_verify)

except FileNotFoundError:
    print("错误：Parquet 文件未找到。请检查文件路径是否正确。")
except Exception as e:
    print(f"发生错误：{e}")


