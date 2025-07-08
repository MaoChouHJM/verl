import pandas as pd
import pickle
import sys

pkl_file = sys.argv[1]

output_csv_file = pkl_file.split('.')[0] + '.csv'

def process_and_save_to_csv(data_dict, output_filename=output_csv_file):
    """
    将包含特定模式键的字典转换为DataFrame并保存为CSV。
    特别处理 sglang_cost_time 字段，将其内部的键值对提取为新的列。

    Args:
        data_dict (dict): 包含 generate_weight_{name}, update_tensor_{name} 等键的字典。
        output_filename (str): 输出CSV文件的名称。
    """
    processed_data = {}
    
    # 遍历字典，提取name和对应的值
    for key, value in data_dict.items():
        if key.startswith("generate_weight_"):
            name = key.replace("generate_weight_", "")
            if name not in processed_data:
                processed_data[name] = {}
            processed_data[name]["generate_weight"] = value
        elif key.startswith("weight_shape_"):
            name = key.replace("weight_shape_", "")
            if name not in processed_data:
                processed_data[name] = {}
            processed_data[name]["weight_shape"] = value
        elif key.startswith("weight_dtype"):
            name = key.replace("weight_dtype_", "")
            if name not in processed_data:
                processed_data[name] = {}
            processed_data[name]["weight_dtype"] = value
        elif key.startswith("update_tensor_"):
            name = key.replace("update_tensor_", "")
            if name not in processed_data:
                processed_data[name] = {}
            processed_data[name]["update_tensor"] = value
        elif key.startswith("sglang_cost_time_"):
            name = key.replace("sglang_cost_time_", "")
            if name not in processed_data:
                processed_data[name] = {}
            
            # 原始的 sglang_cost_time 字段值
            #processed_data[name]["sglang_cost_time_raw"] = value # 可以保留原始字符串，或者不保留
            
            # 解析 sglang_cost_time 字符串
            if isinstance(value, str): # 确保值是字符串类型
                parts = value.strip().split(' ') # 按空格分割
                for part in parts:
                    if '=' in part:
                        sub_key, sub_value_str = part.split('=', 1) # 只分割一次
                        try:
                            # 尝试将值转换为浮点数，如果失败则保留为字符串
                            processed_data[name][f"{sub_key}"] = float(sub_value_str)
                        except ValueError:
                            processed_data[name][f"{sub_key}"] = sub_value_str
            else:
                # 如果 sglang_cost_time 不是字符串，可以根据需要处理，例如设置为None或NaN
                pass # 或者 raise ValueError("sglang_cost_time value is not a string")

        elif key.startswith("flush_cache_"):
            name = key.replace("flush_cache_", "")
            if name not in processed_data:
                processed_data[name] = {}
            processed_data[name]["flush_cache"] = value

    # 将处理后的数据转换为DataFrame
    df = pd.DataFrame.from_dict(processed_data, orient='index')
    
    # 设置索引的名称为 'name'
    df.index.name = 'name'
    
    # 将DataFrame保存为CSV文件
    df.to_csv(output_filename)
    print(f"数据已成功保存到 {output_filename}")
    print("\nCSV文件内容预览：")
    print(df.head())
    print("\n所有列名：")
    print(df.columns.tolist())


# 示例数据 (包含 sglang_cost_time 的字符串格式)
data = {
    "generate_weight_modelA": 10.5,
    "update_tensor_modelA": 2.1,
    "sglang_cost_time_modelA": 'serial_time=0.004073381423950195 tokenizer_to_scheduler_time=0.0004062652587890625 scheduler_broadcast_time=0.0003037452697753906  desrial_time=0.006530284881591797 load_model_time=0.0026128292083740234 broadcast_tensor_time=1.409773588180542 broadcast_meta_time=0.000545501708984375',
    "flush_cache_modelA": 0.1,
    "generate_weight_modelB": 12.3,
    "update_tensor_modelB": 2.5,
    "sglang_cost_time_modelB": 'serial_time=0.005 tokenizer_to_scheduler_time=0.0005 desrial_time=0.007 load_model_time=0.003 broadcast_tensor_time=1.5 broadcast_meta_time=0.0006',
    "flush_cache_modelB": 0.15,
    "generate_weight_modelC": 9.8,
    "update_tensor_modelC": 1.9,
    "sglang_cost_time_modelC": 'serial_time=0.0035 tokenizer_to_scheduler_time=0.00035 desrial_time=0.006 load_model_time=0.0025 broadcast_tensor_time=1.3 broadcast_meta_time=0.0005',
    "flush_cache_modelC": 0.09,
}

with open(pkl_file, 'rb') as f:
    loaded = pickle.load(f)


process_and_save_to_csv(loaded)


