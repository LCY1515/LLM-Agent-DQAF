import random
import pandas as pd
import json
import sklearn.metrics
from main import  inject_mixed_noise, llm_dqaf_pipeline
from util.DataLoader import DataLoader
from conf.recommend_parser import recommend_parse_args
from llm_dqaf.llm_utils import QwenLLM

def sample_raw_data(raw_data, n=2000):
    return random.sample(raw_data, min(n, len(raw_data)))

def mixed_noise_experiment(raw_data, llm_model, recommend_args, sample_size=2000):
    raw_data_small = sample_raw_data(raw_data, sample_size)
    # 一次性注入混合噪声
    noisy_data = inject_mixed_noise(raw_data_small, total_noise_ratio=0.1)
    y_true = [(d.get('is_noisy', False) is not False) for d in noisy_data]
    clean_data, quality_report = llm_dqaf_pipeline(noisy_data, llm_model, strategy='hard', threshold=0.5)
    # 多阈值评估
    thresholds = [0.7]
    eval_results = []
    for threshold in thresholds:
        y_pred = [info['quality'] < threshold for info in quality_report]
        min_len = min(len(y_true), len(y_pred))
        y_true_cut = y_true[:min_len]
        y_pred_cut = y_pred[:min_len]
        acc = sklearn.metrics.accuracy_score(y_true_cut, y_pred_cut)
        recall = sklearn.metrics.recall_score(y_true_cut, y_pred_cut)
        f1 = sklearn.metrics.f1_score(y_true_cut, y_pred_cut)
        print(f"阈值: {threshold:.2f} | 准确率: {acc:.3f} 召回率: {recall:.3f} F1: {f1:.3f}")
        eval_results.append({'threshold': threshold, 'accuracy': acc, 'recall': recall, 'f1': f1})
    # 主评测阈值设为0.7
    y_pred = [info['quality'] < 0.7 for info in quality_report]
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    recall = sklearn.metrics.recall_score(y_true, y_pred)
    f1 = sklearn.metrics.f1_score(y_true, y_pred)
    print(f"清洗前数据量: {len(noisy_data)} 清洗后: {len(clean_data)}")
    data = DataLoader(recommend_args)
    if hasattr(data, 'train_data'):
        data.train_data = pd.DataFrame(clean_data)
    import_str = f'from recommender.{recommend_args.model_name} import {recommend_args.model_name}'
    exec(import_str)
    recommend_model = eval(recommend_args.model_name)(recommend_args, data)
    recommend_model.train()
    _, metrics = recommend_model.test()
    result = {
        'accuracy': acc,
        'recall': recall,
        'f1': f1,
        'raw_data_size': len(raw_data_small),
        'noisy_data_size': len(noisy_data),
        'clean_data_size': len(clean_data),
        'metrics': metrics,
        'threshold_eval': eval_results
    }
    with open('results_mixed.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("混合噪声实验结果已保存到 results_mixed.json")
    return noisy_data  # 返回注入噪声后的数据，便于直接对比

# 直接用注入噪声数据训练推荐系统（不做去噪）
def run_no_dqaf_experiment(noisy_data, recommend_args):
    print(f"未去噪直接用噪声数据训练，数据量: {len(noisy_data)}")
    data = DataLoader(recommend_args)
    if hasattr(data, 'train_data'):
        data.train_data = pd.DataFrame(noisy_data)
    import_str = f'from recommender.{recommend_args.model_name} import {recommend_args.model_name}'
    exec(import_str)
    recommend_model = eval(recommend_args.model_name)(recommend_args, data)
    recommend_model.train()
    _, metrics = recommend_model.test()
    result = {
        'raw_data_size': len(noisy_data),
        'metrics': metrics
    }
    with open('results_no_dqaf_mixed.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("未去噪实验结果已保存到 results_no_dqaf_mixed.json")

def run_clean_experiment(raw_data, recommend_args):
    print(f"直接用干净数据训练，数据量: {len(raw_data)}")
    data = DataLoader(recommend_args)
    if hasattr(data, 'train_data'):
        data.train_data = pd.DataFrame(raw_data)
    import_str = f'from recommender.{recommend_args.model_name} import {recommend_args.model_name}'
    exec(import_str)
    recommend_model = eval(recommend_args.model_name)(recommend_args, data)
    recommend_model.train()
    _, metrics = recommend_model.test()
    result = {
        'raw_data_size': len(raw_data),
        'metrics': metrics
    }
    with open('results_clean.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("干净数据实验结果已保存到 results_clean.json")

if __name__ == '__main__':
    # 加载原始数据
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
    # 推荐参数和LLM
    recommend_args = recommend_parse_args()
    QWEN_API_KEY = "sk-bb190fb4ee0845a8bcb5fbd1a83bbb95"  # 或用os.getenv
    llm_model = QwenLLM(api_key=QWEN_API_KEY)
    # 动态导入推荐模型
    import_str = f'from recommender.{recommend_args.model_name} import {recommend_args.model_name}'
    exec(import_str)
    # 快速预研实验
    noisy_data = mixed_noise_experiment(raw_data, llm_model, recommend_args, sample_size=2000)
    # 直接用噪声数据训练推荐系统（不做去噪）
    run_no_dqaf_experiment(noisy_data, recommend_args)
    # 用干净数据训练推荐系统
    run_clean_experiment(raw_data, recommend_args)