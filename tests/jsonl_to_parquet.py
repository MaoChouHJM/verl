import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def jsonl_to_parquet(jsonl_file_path, parquet_file_path):
    """
    将 JSONL 文件转换为 Parquet 文件。

    Args:
        jsonl_file_path (str): 输入的 JSONL 文件路径。
        parquet_file_path (str): 输出的 Parquet 文件路径。
    """
    try:
        # 使用 pandas 读取 JSONL 文件
        # lines=True 参数告诉 pandas 每一行是一个独立的 JSON 对象
        df = pd.read_json(jsonl_file_path, lines=True)

        # 将 pandas DataFrame 转换为 PyArrow Table
        table = pa.Table.from_pandas(df)

        # 将 PyArrow Table 写入 Parquet 文件
        pq.write_table(table, parquet_file_path)

        print(f"成功将 {jsonl_file_path} 转换为 {parquet_file_path}")

    except FileNotFoundError:
        print(f"错误：文件未找到 - {jsonl_file_path}")
    except Exception as e:
        print(f"转换过程中发生错误：{e}")

if __name__ == "__main__":
    # 示例用法：
    input_jsonl_file = sys.argv[1]  # 替换为你的 JSONL 文件路径
    output_parquet_file = sys.argv[2] # 替换为你希望保存的 Parquet 文件路径
    
    jsonl_to_parquet(input_jsonl_file, output_parquet_file)
