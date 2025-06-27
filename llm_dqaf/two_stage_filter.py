import numpy as np
import bisect
from collections import defaultdict
from .SND import semantic_noise_detection
import json
import os

CACHE_FILE = "llm_noise_cache.json"

def load_llm_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_llm_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def get_record_key(record):
    # 用user_id, item_id, rating, timestamp唯一标识一条数据
    return f"{record['user_id']}_{record['item_id']}_{record['rating']}_{record['timestamp']}"

def is_rating_anomaly(record, user_history):
    """检测评分异常"""
    if not user_history:
        return False
    
    user_ratings = [r['rating'] for r in user_history]
    mean_rating = np.mean(user_ratings)
    std_rating = np.std(user_ratings)
    
    # 评分偏离均值超过2个标准差
    z_score = abs(record['rating'] - mean_rating) / (std_rating + 1e-8)
    return z_score > 2.0

def is_temporal_anomaly(record, user_history):
    """检测时间异常"""
    if len(user_history) < 2:
        return False
    
    user_times = sorted([r['timestamp'] for r in user_history])
    current_time = record['timestamp']
    
    # 找到当前时间在历史时间中的位置
    insert_idx = bisect.bisect_left(user_times, current_time)
    
    # 检查时间间隔是否异常
    if insert_idx > 0:
        time_diff_prev = current_time - user_times[insert_idx - 1]
        if time_diff_prev < 20:  # 20秒内连续评分
            return True
    
    if insert_idx < len(user_times):
        time_diff_next = user_times[insert_idx] - current_time
        if time_diff_next < 20:  # 20秒内连续评分
            return True
    
    return False

def is_frequency_anomaly(record, user_history, time_window=3600):
    """检测行为频率异常"""
    current_time = record['timestamp']
    
    # 统计1小时内的评分数量
    recent_ratings = [r for r in user_history 
                     if abs(r['timestamp'] - current_time) <= time_window]
    
    # 如果1小时内评分超过30次，可能是异常
    return len(recent_ratings) > 30

def is_duplicate_rating(record, user_history):
    """检测重复评分"""
    item_id = record['item_id']
    
    # 检查用户是否已经对该物品评过分
    for r in user_history:
        if r['item_id'] == item_id:
            return True
    
    return False

def is_extreme_rating(record):
    """检测极端评分"""
    rating = record['rating']
    
    # 评分1分或5分且没有其他信息支持
    return rating == 1 or rating == 5

def is_suspicious_by_rules(record, user_history):
    """综合快速规则判断"""
    suspicious_flags = []
    
    # 评分异常
    if is_rating_anomaly(record, user_history):
        suspicious_flags.append("rating_anomaly")
    
    # 时间异常
    if is_temporal_anomaly(record, user_history):
        suspicious_flags.append("temporal_anomaly")
    
    # 频率异常
    if is_frequency_anomaly(record, user_history):
        suspicious_flags.append("frequency_anomaly")
    
    # 重复评分
    if is_duplicate_rating(record, user_history):
        suspicious_flags.append("duplicate_rating")
    
    # 极端评分
    if is_extreme_rating(record):
        suspicious_flags.append("extreme_rating")
    
    # 如果有2个或以上异常标志，认为是可疑数据
    return len(suspicious_flags) >= 2, suspicious_flags

def two_stage_noise_detection(records, llm_model, max_records=None):
    """两阶段噪声检测"""
    llm_cache = load_llm_cache()
    user_history = defaultdict(list)
    for record in records:
        user_history[record['user_id']].append(record)
    
    # 阶段1：快速规则过滤
    suspicious_records = []
    suspicious_indices = []
    normal_results = []
    suspicious_keys = []
    
    for i, record in enumerate(records):
        # 获取用户历史（排除当前记录）
        history = [r for r in user_history[record['user_id']] if r != record]
        
        # 快速规则判断
        is_suspicious, flags = is_suspicious_by_rules(record, history)
        
        if is_suspicious:
            suspicious_records.append(record)
            suspicious_indices.append(i)
            suspicious_keys.append(get_record_key(record))
            print(f"可疑数据 {i}: {flags}")
        else:
            # 正常数据直接赋低噪声分数
            normal_results.append({
                "semantic_noise": 0.0,
                "behavior_noise": 0.0,
                "content_noise": 0.0,
                "explanation": "快速规则判断为正常数据"
            })
    
    print(f"快速过滤结果: {len(records)}条数据中，{len(suspicious_records)}条可疑，{len(normal_results)}条正常")
    
    # 阶段2：LLM精筛可疑数据（查缓存）
    llm_results = []
    uncached_records = []
    uncached_keys = []
    for rec, key in zip(suspicious_records, suspicious_keys):
        if key in llm_cache:
            llm_results.append(llm_cache[key])
        else:
            uncached_records.append(rec)
            uncached_keys.append(key)
    if uncached_records:
        print(f"开始LLM精筛 {len(uncached_records)} 条未缓存可疑数据...")
        new_results = semantic_noise_detection(uncached_records, llm_model)
        for key, result in zip(uncached_keys, new_results):
            llm_cache[key] = result
            llm_results.append(result)
        save_llm_cache(llm_cache)
    else:
        print("所有可疑数据均已缓存，无需LLM调用。")
    
    # 合并结果
    all_results = [None] * len(records)
    
    # 填充正常数据结果
    normal_idx = 0
    llm_idx = 0
    for i in range(len(records)):
        if i not in suspicious_indices:
            all_results[i] = normal_results[normal_idx]
            normal_idx += 1
        else:
            all_results[i] = llm_results[llm_idx]
            llm_idx += 1
    
    return all_results 