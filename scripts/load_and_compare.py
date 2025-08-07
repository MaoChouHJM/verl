import torch
from pathlib import Path
from verl import DataProto
import pickle
import csv
import pandas as pd
import argparse
import re


def clean_excel_data(df):
    # remove characters excel does not support
    illegal_chars = [chr(i) for i in range(0, 9)] + [chr(i) for i in range(11, 13)] + [chr(i) for i in range(14, 32)]
    
    pattern = '[' + re.escape(''.join(illegal_chars)) + ']'
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(pattern, '', regex=True)
    
    return df

def parse_args():
    parser = argparse.ArgumentParser(description="Compare SWIFT and VERL data outputs")
    
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to SWIFT data directory"
    )
    
    parser.add_argument(
        "--verl-data-path", 
        type=str,
        required=True,
        help="Path to VERL train data (.pt file)"
    )
    
    parser.add_argument(
        "--verl-proto-path",
        type=str, 
        required=True,
        help="Path to VERL protocol batch (.pkl file)"
    )
    
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Output path for comparison results (.pkl file)"
    )
    
    parser.add_argument(
        "--save-xlsx-path",
        type=str,
        required=True, 
        help="Output path for Excel comparison results (.xlsx file)"
    )
    
    parser.add_argument(
        "--num-ranks",
        type=int,
        default=8,
        help="Number of ranks for data loading (default: 8)"
    )
    
    return parser.parse_args()


def load_and_concat(base_path, num_ranks=8):
    base_path = Path(base_path)
    all_messages = []
    all_completions = []
    all_reward_MyBaseAccuracy = []
    all_reward_MyFormat = []
    all_completions_serve = []
    all_completions_serve_ModelAccuracyV2 = []
    all_completions_serve_MyFormat = []
    
    for i in range(num_ranks):
        rank_file = base_path / f"rank{i}_globalstep0_buffered_inputs.pt"
        
        if not rank_file.exists():
            print(f"Warning: {rank_file} does not exist")
            continue
        
        data = torch.load(rank_file, map_location='cpu')
        
        # if 'reward_kwargs' not in data:
        #     print(f"Warning: {rank_file} does not have 'reward_kwargs'")
        #     continue
        
        reward_kwargs = data['reward_kwargs']['messages']
        
        # if not isinstance(reward_kwargs, list):
        #     print(f"Warning: {rank_file} messages is not list format")
        #     continue

        completions = data['completions']

        completions_serve = data['completions_serve']
        completions_serve_ModelAccuracyV2 = data['completions_serve_modelaccuracyv2']
        completions_serve_MyFormat = data["completions_serve_myformat"]
        
        rewards = data['rewards_per_func']
        mybaseaccuracy = rewards[:, 0].tolist()
        myformat = rewards[:, 1].tolist()

        all_completions.extend(completions)
        all_reward_MyBaseAccuracy.extend(mybaseaccuracy)
        all_reward_MyFormat.extend(myformat)

        all_completions_serve.extend(completions_serve)
        all_completions_serve_ModelAccuracyV2.extend(completions_serve_ModelAccuracyV2)
        all_completions_serve_MyFormat.extend(completions_serve_MyFormat)

        for i in range(len(reward_kwargs)):
            all_messages.append(reward_kwargs[i][0]['content'])
        print(f"loading {rank_file.name}...")
    
    if not all_messages:
        print("Warning: not messages found!")
        return None
    
    print(f"Successful concat! Total {len(all_messages)} messages, {len(all_completions)} completions, {len(all_reward_MyBaseAccuracy)} reward scores for MyBaseAccuracy, {len(all_reward_MyFormat)} reward scores for MyFormat!")
    
    return all_messages, all_completions, all_completions_serve, all_reward_MyBaseAccuracy, all_completions_serve_ModelAccuracyV2, all_reward_MyFormat, all_completions_serve_MyFormat

def load_verl(path):
    data = torch.load(path, weights_only=False)
    return data['completions'][0], data['rewards_per_func'][0], data['rewards_per_func'][1]

def get_repeat_keys(path):
    batch = DataProto.load_from_disk(path)
    if 'messages' in batch.non_tensor_batch:
        messages = batch.non_tensor_batch['messages']
    else:
        messages = batch.non_tensor_batch['raw_prompt']
    content_list = [item[0]['content'] for item in messages]
    return content_list

def compare_string_lists(list1, list2):
    print(f"Comparing messages for swift and verl...")
    if len(list1) != len(list2):
        print(f"len diff: {len(list1)} vs {len(list2)}")
        return False
    
    differences = []
    for i, (str1, str2) in enumerate(zip(list1, list2)):
        if str1 != str2:
            differences.append(i)
    
    if differences:
        print(f"Spot {len(differences)} differences: {differences}")
        return False
    else:
        print("All messages are the same!")
        return True

if __name__ == "__main__":
    # data_path = "/hetu_group/jky/playground_hhd_2/2025/20250811_verl_rl/1.reward/output_max_tokens_5120/"
    # verl_data_path = "/nlp_group/yuanjiawei05/new_logits_distill/train_dir/train_data.pt"
    # verl_proto_path = "/nlp_group/yuanjiawei05/new_logits_distill/train_dir/train_batch.pkl"

    # save_path = "/nlp_group/yuanjiawei05/new_logits_distill/train_dir/compare_res.pkl"
    # save_xlsx_path = "/nlp_group/yuanjiawei05/new_logits_distill/train_dir/compare_res.xlsx"

    args = parse_args()
    
    # Validate input paths exist
    data_path = Path(args.data_path)
    verl_data_path = Path(args.verl_data_path)
    verl_proto_path = Path(args.verl_proto_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    if not verl_data_path.exists():
        raise FileNotFoundError(f"VERL data path does not exist: {verl_data_path}")
    if not verl_proto_path.exists():
        raise FileNotFoundError(f"VERL protocol path does not exist: {verl_proto_path}")
    
    # Create output directories if they don't exist
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_xlsx_path).parent.mkdir(parents=True, exist_ok=True)

    swift_messages, swift_completions, swift_s_completions, swift_fn1, swift_s_fn1, swift_fn2, swift_s_fn2 = load_and_concat(data_path, num_ranks=8)
    
    verl_messages = get_repeat_keys(verl_proto_path)
    verl_completions, verl_fn1, verl_fn2 = load_verl(verl_data_path)

    compare_string_lists(swift_messages, verl_messages)
    diff_type = ["DIFF" if (ss_fn1 != v_fn1 or ss_fn2 != v_fn2) else "MATCH" 
               for ss_fn1, v_fn1, ss_fn2, v_fn2 in zip(swift_s_fn1, verl_fn1, swift_s_fn2, verl_fn2)]
    
    
    # bad_case_id = [222, 223, 263, 386, 408, 499, 626, 676, 714, 716, 762, 784, 809, 813, 1521, 1544, 1551, 1574, 1575, 1670, 1689, 1690, 1693, 1812, 1814, 1887, 1896, 1897, 1898, 1899, 1900, 1909]
    # compare_dict = {
    #     i: {
    #         'verl_message': verl_msg,
    #         'swift_completion': swift_comp,
    #         'swift_s_completion': swift_s_comp,
    #         'verl_completion': verl_comp,
    #         'swift_fn1': s_fn1,
    #         'swift_s_fn1': ss_fn1,
    #         'verl_fn1': v_fn1,
    #         'swift_fn2': s_fn2,
    #         'swift_s_fn2': ss_fn2,
    #         'verl_fn2': v_fn2,
    #         'diff_type': dtype
    #     }
    #     for i, (verl_msg, swift_comp, swift_s_comp, verl_comp, s_fn1, ss_fn1, v_fn1, s_fn2, ss_fn2, v_fn2, dtype) 
    #     in enumerate(zip(verl_messages, swift_completions, swift_s_completions, verl_completions, swift_fn1, swift_s_fn1, verl_fn1, swift_fn2, swift_s_fn2, verl_fn2, diff_type))
    # }
    # with open(args.save_path, 'wb') as f:
    #     pickle.dump(compare_dict, f)

    # # with open(save_csv_path, 'w', newline='', encoding='utf-8') as f:
    # #     writer = csv.writer(f)
    # #     writer.writerow(['index', 'messages', 'swift_completions', 'verl_completions', 'swift_fn1', 'verl_fn1', 'swift_fn2', 'verl_fn2'])  # 表头
    # #     for key, values in compare_dict.items():
    # #         writer.writerow([key] + values)

    # data = [list(values.values()) for key, values in compare_dict.items()]
    # df = pd.DataFrame(data, columns=['messages', 'swift_completions', 'swift_s_completions', 'verl_completions', 'swift_fn1', 'swift_s_fn1', 'verl_fn1', 'swift_fn2', 'swift_s_fn2', 'verl_fn2', 'diff'])
    # df = clean_excel_data(df)
    # df.to_excel(args.save_xlsx_path, index=False)

    compare_dict = {
        i: {
            'verl_message': verl_msg,
            'verl_completion': verl_comp,
            'verl_fn1': v_fn1,
            'verl_fn2': v_fn2,
        }
        for i, (verl_msg, verl_comp, v_fn1, v_fn2) 
        in enumerate(zip(verl_messages, verl_completions, verl_fn1, verl_fn2))
    }

    with open(args.save_path, 'wb') as f:
        pickle.dump(compare_dict, f)

    # with open(save_csv_path, 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['index', 'messages', 'verl_completions', 'verl_fn1', 'verl_fn2'])  # 表头
    #     for key, values in compare_dict.items():
    #         writer.writerow([key] + values)

    data = [list(values.values()) for key, values in compare_dict.items()]
    df = pd.DataFrame(data, columns=['messages', 'verl_completions', 'verl_fn1', 'verl_fn2'])
    df = clean_excel_data(df)
    df.to_excel(args.save_xlsx_path, index=False)

