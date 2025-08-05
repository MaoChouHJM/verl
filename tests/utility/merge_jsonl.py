import json
import os

def merge_jsonl_files(file1_path, file2_path, output_file_path, encoding='utf-8'):
    """
    合并两个JSONL文件到一个新的JSONL文件。

    Args:
        file1_path (str): 第一个JSONL文件的路径。
        file2_path (str): 第二个JSONL文件的路径。
        output_file_path (str): 合并后JSONL文件的输出路径。
        encoding (str): 文件编码，默认为'utf-8'。
    """
    merged_data = []

    # 读取第一个JSONL文件
    try:
        with open(file1_path, 'r', encoding=encoding) as infile1:
            for line in infile1:
                line = line.strip()
                if line:  # 确保行不为空
                    try:
                        merged_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"警告: 文件 {file1_path} 中发现无效JSON行: {line} - 错误: {e}")
    except FileNotFoundError:
        print(f"错误: 文件 {file1_path} 未找到。")
        return
    except Exception as e:
        print(f"读取文件 {file1_path} 时发生错误: {e}")
        return

    # 读取第二个JSONL文件
    try:
        with open(file2_path, 'r', encoding=encoding) as infile2:
            for line in infile2:
                line = line.strip()
                if line:  # 确保行不为空
                    try:
                        merged_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"警告: 文件 {file2_path} 中发现无效JSON行: {line} - 错误: {e}")
    except FileNotFoundError:
        print(f"错误: 文件 {file2_path} 未找到。")
        return
    except Exception as e:
        print(f"读取文件 {file2_path} 时发生错误: {e}")
        return

    # 将合并后的数据写入新的JSONL文件
    try:
        with open(output_file_path, 'w', encoding=encoding) as outfile:
            for item in merged_data:
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"成功将 {file1_path} 和 {file2_path} 合并到 {output_file_path}")
    except Exception as e:
        print(f"写入文件 {output_file_path} 时发生错误: {e}")

# 示例用法
if __name__ == "__main__":
    ## 创建示例JSONL文件
    ## one.jsonl
    #with open('one.jsonl', 'w', encoding='utf-8') as f:
    #    f.write('{"name": "one", "description": "testDescription...", "comment": "1"}\n')
    #    f.write('{"name": "two", "description": "testDescription2...", "comment": "2"}\n')

    ## second.jsonl
    #with open('second.jsonl', 'w', encoding='utf-8') as f:
    #    f.write('{"name": "eleven", "description": "testDescription11...", "comment": "11"}\n')
    #    f.write('{"name": "twelve", "description": "testDescription12...", "comment": "12"}\n')
    #    f.write('{"name": "thirteen", "description": "testDescription13...", "comment": "13"}\n')

    file1 = '/hetu_group/jky/misc/tools/swift_20250508/playground/keye_8b_20250613/rl/20250704.1.r1reward_tpl_v4__fixanswerblank/tools/verl_dataset_debug_text_img.jsonl'
    file2 = '/hetu_group/jky/misc/tools/swift_20250508/playground/keye_8b_20250613/rl/20250704.1.r1reward_tpl_v4__fixanswerblank/tools/verl_dataset_debug_video.jsonl'
    output_file = '/nlp_group/huangjiaming/logits-distill/merged_file.jsonl'

    merge_jsonl_files(file1, file2, output_file)

    # 验证合并后的文件内容
    print("\n合并文件内容:")
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                print(line.strip())
    except FileNotFoundError:
        print(f"错误: 输出文件 {output_file} 未找到。")

    # 清理示例文件
    # os.remove('one.jsonl')
    # os.remove('second.jsonl')
    # os.remove('merged_file.jsonl')

