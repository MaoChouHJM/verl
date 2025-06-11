import pandas as pd
import pyarrow.parquet as pq
import random
import os

def select_random_row_and_save(input_parquet_path, output_parquet_path):
    """
    读取一个Parquet文件，从中随机选择一行，并将其保存为另一个Parquet文件。

    Args:
        input_parquet_path (str): 输入Parquet文件的路径。
        output_parquet_path (str): 输出Parquet文件的路径，将保存随机选择的行。
    """
    if not os.path.exists(input_parquet_path):
        print(f"错误：输入文件 '{input_parquet_path}' 不存在。")
        return

    try:
        # 1. 读取Parquet文件到Pandas DataFrame
        print(f"正在读取Parquet文件：{input_parquet_path}")
        df = pd.read_parquet(input_parquet_path)
        print(f"文件读取成功。总行数：{len(df)}")

        if df.empty:
            print("警告：输入Parquet文件为空，无法选择行。")
            return

        # 2. 随机选择一行
        # 使用df.sample(n=1) 是最简洁和推荐的方式
        random_row_df = df.sample(n=1)
        print("\n随机选择的行数据：")
        print(random_row_df)

        # 3. 将随机选择的行保存为新的Parquet文件
        print(f"\n正在将随机选择的行保存到：{output_parquet_path}")
        random_row_df.to_parquet(output_parquet_path, index=False) # index=False 避免将DataFrame索引写入文件
        print(f"随机选择的行已成功保存到 '{output_parquet_path}'。")

    except Exception as e:
        print(f"处理Parquet文件时发生错误：{e}")

# --- 示例用法 ---
if __name__ == "__main__":
    # 1. 创建一个示例Parquet文件（如果不存在）
    # 这是一个辅助函数，用于生成一个用于测试的Parquet文件
    def create_sample_parquet(file_path, num_rows=10):
        if not os.path.exists(file_path):
            print(f"正在创建示例Parquet文件：{file_path}")
            data = {
                'id': range(num_rows),
                'name': [f'Item_{i}' for i in range(num_rows)],
                'value': [random.randint(100, 1000) for _ in range(num_rows)],
                'category': [random.choice(['A', 'B', 'C']) for _ in range(num_rows)]
            }
            sample_df = pd.DataFrame(data)
            sample_df.to_parquet(file_path, index=False)
            print("示例Parquet文件创建成功。")
        else:
            print(f"示例Parquet文件 '{file_path}' 已存在，跳过创建。")

    input_file = "/nlp_group/huangjiaming/data/gsm8k/test.parquet"
    output_file = "random_row.parquet"

    # 创建一个示例文件，以便你可以运行脚本
    create_sample_parquet(input_file, num_rows=20)

    # 运行主函数
    select_random_row_and_save(input_file, output_file)

    # 你可以再次运行，看看是否选择了不同的行
    # print("\n--- 再次运行以选择另一行 ---")
    # select_random_row_and_save(input_file, "another_random_row.parquet")

