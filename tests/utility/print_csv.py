import csv
import sys

def parse_csv_and_print_column(file_path, column_name):
    """
    解析 CSV 文件并打印指定列的所有值。

    Args:
        file_path (str): CSV 文件的路径。
        column_name (str): 要打印的列的名称。
    """
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            # 使用 csv.DictReader 可以将每一行读取为字典，
            # 字典的键是 CSV 文件的头部（header）。
            reader = csv.DictReader(csvfile)

            # 检查指定的列名是否存在于 CSV 文件的头部
            if column_name not in reader.fieldnames:
                print(f"错误：CSV 文件中不存在名为 '{column_name}' 的列。")
                print(f"可用的列有：{', '.join(reader.fieldnames)}")
                return

            print(f"--- '{column_name}' 列的值 ---")
            for row in reader:
                # 每一行都是一个字典，通过列名作为键来访问值
                #print(row[column_name])
                name = row[column_name]
                split_tokens = name.split('.')
                if len(split_tokens) > 3:
                    print(split_tokens[2])
                else:
                    print(name)
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。请检查文件路径是否正确。")
    except Exception as e:
        print(f"处理 CSV 文件时发生错误：{e}")

# --- 使用示例 ---
if __name__ == "__main__":
    csv_file = sys.argv[1]  # 你的 CSV 文件名
    target_column = 'name' # 你想打印的列名

    # 调用函数解析并打印 'name' 列
    parse_csv_and_print_column(csv_file, target_column)

    #print("\n--- 尝试打印不存在的列 ---")
    #parse_csv_and_print_column(csv_file, 'email') # 尝试打印一个不存在的列

