import sys
import pyarrow.parquet as pq
import pandas as pd # 通常还是会用到 pandas 来方便遍历

# 假设你有一个名为 'example.parquet' 的 Parquet 文件
# 如果没有，可以先创建一个用于测试
# data = {'col1': [1, 2, 3], 'col2': ['A', 'B', 'C']}
# df_test = pd.DataFrame(data)
# df_test.to_parquet('example.parquet', index=False)

file_path = sys.argv[1]


parquet_file = pq.ParquetFile(file_path)

print(f"成功打开文件: {file_path}")
print("文件包含以下列:", parquet_file.schema.names)
#print("文件总行数:", parquet_file.num_rows)
print("文件包含行组数:", parquet_file.num_row_groups)
#exit(0)
print("\n开始遍历每一条数据:")

# 方法 1: 读取整个文件到 pandas DataFrame 并遍历
# print("使用 pandas DataFrame 遍历:")
# df = parquet_file.read().to_pandas()
# for index, row in df.iterrows():
#     print(f"行索引: {index}")
#     print("数据:", row.to_dict())
#     print("-" * 30)

# 方法 2: 按行组读取并使用 pandas DataFrame 遍历
print("按行组读取并使用 pandas DataFrame 遍历:")
total_rows_processed = 0
for i in range(parquet_file.num_row_groups):
    print(f"\n正在处理行组 {i}...")
    row_group_table = parquet_file.read_row_group(i)
    df_row_group = row_group_table.to_pandas()

    for index, row in df_row_group.iterrows():
        # 注意这里的 index 是行组内的索引
        global_index = total_rows_processed + index
        if len(row.to_dict()['images']) != 1:
            print("数据:", len(row.to_dict()['images']))
    total_rows_processed += len(df_row_group)

print(f'{total_rows_processed=}')


