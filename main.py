import time
from conf.recommend_parser import recommend_parse_args
from util.DataLoader import DataLoader
from util.tool import seedSet
import os
import torch
import numpy as np
import random
from llm_dqaf.llm_utils import QwenLLM
from llm_dqaf.pipeline import llm_dqaf_pipeline
import pandas as pd
import json
import copy

def inject_label_noise(raw_data, noise_ratio=0.1):
    noisy_data = copy.deepcopy(raw_data)
    n = len(noisy_data)
    num_noisy = int(n * noise_ratio)
    noisy_indices = random.sample(range(n), num_noisy)
    for idx in noisy_indices:
        rating = noisy_data[idx]['rating']
        noisy_data[idx]['rating'] = 6 - rating
        noisy_data[idx]['is_noisy'] = True
    print(f"已注入{num_noisy}条标签噪声")
    return noisy_data

def inject_behavior_noise(raw_data, user_ratio=0.05, burst_size=10, burst_interval=10):
    noisy_data = copy.deepcopy(raw_data)
    user_ids = list(set([r['user_id'] for r in noisy_data]))
    num_noisy_users = max(1, int(len(user_ids) * user_ratio))
    noisy_users = random.sample(user_ids, num_noisy_users)
    max_timestamp = max(r['timestamp'] for r in noisy_data)
    new_records = []
    for user_id in noisy_users:
        for i in range(burst_size):
            new_records.append({
                'user_id': user_id,
                'item_id': random.randint(1, 1700),
                'rating': random.randint(1, 5),
                'timestamp': max_timestamp + i * burst_interval,
                'is_noisy': True
            })
    noisy_data.extend(new_records)
    print(f"为{num_noisy_users}个用户注入{len(new_records)}条行为噪声")
    return noisy_data

def inject_content_noise(raw_data, item_ratio=0.05, boost_rating=5):
    noisy_data = copy.deepcopy(raw_data)
    item_ids = list(set([r['item_id'] for r in noisy_data]))
    num_noisy_items = max(1, int(len(item_ids) * item_ratio))
    noisy_items = random.sample(item_ids, num_noisy_items)
    user_ids = list(set([r['user_id'] for r in noisy_data]))
    max_timestamp = max(r['timestamp'] for r in noisy_data)
    new_records = []
    for item_id in noisy_items:
        for i in range(10):
            new_records.append({
                'user_id': random.choice(user_ids),
                'item_id': item_id,
                'rating': boost_rating,
                'timestamp': max_timestamp + i + random.randint(1, 100),
                'is_noisy': True
            })
    noisy_data.extend(new_records)
    print(f"为{num_noisy_items}个物品注入{len(new_records)}条内容噪声")
    return noisy_data

def inject_mixed_noise(raw_data, total_noise_ratio=0.1):
    noisy_data = copy.deepcopy(raw_data)
    n = len(noisy_data)
    num_noisy = int(n * total_noise_ratio)
    indices = list(range(n))
    random.shuffle(indices)
    noise_types = ['label', 'behavior', 'content']
    user_ids = list(set([r['user_id'] for r in noisy_data]))
    item_ids = list(set([r['item_id'] for r in noisy_data]))
    max_timestamp = max(r['timestamp'] for r in noisy_data)
    label_count = behavior_count = content_count = 0
    for idx in indices[:num_noisy]:
        noise_type = random.choice(noise_types)
        if noise_type == 'label':
            rating = noisy_data[idx]['rating']
            noisy_data[idx]['rating'] = 6 - rating
            noisy_data[idx]['is_noisy'] = 'label'
            label_count += 1
        elif noise_type == 'behavior':
            user_id = noisy_data[idx]['user_id']
            noisy_data.append({
                'user_id': user_id,
                'item_id': random.randint(1, 1700),
                'rating': random.randint(1, 5),
                'timestamp': max_timestamp + random.randint(1, 100),
                'is_noisy': 'behavior'
            })
            behavior_count += 1
        elif noise_type == 'content':
            item_id = noisy_data[idx]['item_id']
            noisy_data.append({
                'user_id': random.choice(user_ids),
                'item_id': item_id,
                'rating': 5,
                'timestamp': max_timestamp + random.randint(1, 100),
                'is_noisy': 'content'
            })
            content_count += 1
    print(f"已注入混合噪声：标签{label_count}条，行为{behavior_count}条，内容{content_count}条，总计{num_noisy}条")
    return noisy_data

def inject_high_suspicion_noise(raw_data, total_noise_ratio=0.1, burst_size=10, time_span=20):
    """
    注入高可疑度混合噪声：
    - 随机选N个用户，每个用户对同一物品在极短时间内反复评分（极端分），
    - 每条新噪声记录能命中4个及以上规则。
    """
    noisy_data = copy.deepcopy(raw_data)
    n = len(noisy_data)
    num_noisy = int(n * total_noise_ratio)
    user_ids = list(set([r['user_id'] for r in noisy_data]))
    item_ids = list(set([r['item_id'] for r in noisy_data]))
    max_timestamp = max(r['timestamp'] for r in noisy_data)
    # 每个用户注入burst_size条高可疑噪声
    num_users = max(1, num_noisy // burst_size)
    selected_users = random.sample(user_ids, num_users)
    noise_count = 0
    for user_id in selected_users:
        item_id = random.choice(item_ids)
        base_time = max_timestamp + random.randint(1, 10000)
        for i in range(burst_size):
            noisy_data.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': random.choice([1, 5]),
                'timestamp': base_time + (i * (time_span // burst_size)),
                'is_noisy': 'high_suspicion'
            })
            noise_count += 1
            if noise_count >= num_noisy:
                break
        if noise_count >= num_noisy:
            break
    print(f"已注入高可疑度混合噪声：共{noise_count}条，每条可命中多重异常规则")
    return noisy_data

def run_experiment(raw_data, llm_model, recommend_args, noise_type=None, noise_ratio=0.1):
    # 1. 注入噪声
    if noise_type == 'label':
        noisy_data = inject_label_noise(raw_data, noise_ratio)
    elif noise_type == 'behavior':
        noisy_data = inject_behavior_noise(raw_data, user_ratio=noise_ratio)
    elif noise_type == 'content':
        noisy_data = inject_content_noise(raw_data, item_ratio=noise_ratio)
    else:
        noisy_data = copy.deepcopy(raw_data)
        print("未注入噪声，使用原始数据")
    # 2. LLM-DQAF清洗
    clean_data, quality_report = llm_dqaf_pipeline(noisy_data, llm_model, strategy='hard', threshold=0.5)
    print(f"LLM-DQAF清洗后数据条数: {len(clean_data)}")
    # 3. 用clean_data训练推荐模型，评估指标
    data = DataLoader(recommend_args)
    if hasattr(data, 'train_data'):
        data.train_data = pd.DataFrame(clean_data)
    recommend_model = eval(recommend_args.model_name)(recommend_args, data)
    recommend_model.train()
    _, metrics = recommend_model.test()
    # 4. 返回实验结果
    return {
        'noise_type': noise_type or 'none',
        'noise_ratio': noise_ratio,
        'raw_data_size': len(raw_data),
        'noisy_data_size': len(noisy_data),
        'clean_data_size': len(clean_data),
        'metrics': metrics
    }

def run_mixed_noise_experiment(raw_data, llm_model, recommend_args, total_noise_ratio=0.1):
    noisy_data = inject_mixed_noise(raw_data, total_noise_ratio)
    clean_data, quality_report = llm_dqaf_pipeline(noisy_data, llm_model, strategy='hard', threshold=0.5)
    print(f"LLM-DQAF清洗后数据条数: {len(clean_data)}")
    data = DataLoader(recommend_args)
    if hasattr(data, 'train_data'):
        data.train_data = pd.DataFrame(clean_data)
    recommend_model = eval(recommend_args.model_name)(recommend_args, data)
    recommend_model.train()
    _, metrics = recommend_model.test()
    result = {
        'noise_type': 'mixed_random_ratio',
        'total_noise_ratio': total_noise_ratio,
        'raw_data_size': len(raw_data),
        'noisy_data_size': len(noisy_data),
        'clean_data_size': len(clean_data),
        'metrics': metrics
    }
    with open('results_llmdqaf_mixed.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("混合噪声实验结果已保存到 results_llmdqaf_mixed.json")
    return result

def run_high_suspicion_noise_experiment(raw_data, llm_model, recommend_args, total_noise_ratio=0.1):
    noisy_data = inject_high_suspicion_noise(raw_data, total_noise_ratio)
    clean_data, quality_report = llm_dqaf_pipeline(noisy_data, llm_model, strategy='hard', threshold=0.5)
    print(f"LLM-DQAF清洗后数据条数: {len(clean_data)}")
    data = DataLoader(recommend_args)
    if hasattr(data, 'train_data'):
        data.train_data = pd.DataFrame(clean_data)
    recommend_model = eval(recommend_args.model_name)(recommend_args, data)
    recommend_model.train()
    _, metrics = recommend_model.test()
    result = {
        'noise_type': 'high_suspicion_mixed',
        'total_noise_ratio': total_noise_ratio,
        'raw_data_size': len(raw_data),
        'noisy_data_size': len(noisy_data),
        'clean_data_size': len(clean_data),
        'metrics': metrics
    }
    with open('results_llmdqaf_highsuspicion.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("高可疑度混合噪声实验结果已保存到 results_llmdqaf_highsuspicion.json")
    return result

if __name__ == '__main__':
    # 1. Load configuration
    recommend_args = recommend_parse_args()
    # 2. Import recommend model and attack model
    os.environ['CUDA_VISIBLE_DEVICES'] = recommend_args.gpu_id
    seed = recommend_args.seed
    seedSet(seed)

    import_str = 'from recommender.' + recommend_args.model_name + ' import ' + recommend_args.model_name
    exec(import_str)


    # 3. Load train.txt为raw_data
    raw_data = []
    with open('data/clean/ml-100k/train.txt', encoding='utf-8') as f:
        for line in f:
            user_id, item_id, rating, timestamp = line.strip().split()
            raw_data.append({
                'user_id': int(user_id),
                'item_id': int(item_id),
                'rating': int(rating),
                'timestamp': int(timestamp)
            })

    # 4. LLM-DQAF清洗（Qwen-Plus）
    # 获取API密钥，优先使用环境变量
    QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    # 如果环境变量未设置，可以在这里直接设置API密钥
    QWEN_API_KEY = "sk-bb190fb4ee0845a8bcb5fbd1a83bbb95"
    llm_model = QwenLLM(api_key=QWEN_API_KEY)
    # 实验
    run_high_suspicion_noise_experiment(raw_data, llm_model, recommend_args, total_noise_ratio=0.1)
