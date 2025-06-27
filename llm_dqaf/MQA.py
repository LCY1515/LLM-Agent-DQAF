import numpy as np
from collections import defaultdict, Counter
import math

def entropy(arr):
    arr = np.array(arr)
    probs = np.bincount(arr) / len(arr) if len(arr) > 0 else np.array([1.0])
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs + 1e-8)) if len(probs) > 0 else 1.0

def multi_dim_quality_assessment(records, noise_scores, llm_model, item2cat=None, alpha=0.4, beta=0.4, gamma=0.2, delta=0.25):
    """
    Q_semantic: LLM语义分数反向
    Q_behavior: 统计行为一致性+LLM行为分融合
    Q_temporal: 类别转移概率近似

    """
    user_ratings = defaultdict(list)
    user_items = defaultdict(list)
    user_times = defaultdict(list)
    user_cats = defaultdict(list)
    all_cats = []
    all_ratings = []
    all_users = []
    for r in records:
        user_ratings[r['user_id']].append(r['rating'])
        user_items[r['user_id']].append(r['item_id'])
        user_times[r['user_id']].append(r['timestamp'])
        all_ratings.append(r['rating'])
        all_users.append(r['user_id'])
        if item2cat is not None:
            cat = item2cat.get(r['item_id'], None)
            if cat is not None:
                user_cats[r['user_id']].append(cat)
                all_cats.append(cat)
    # 多样性
    #H_cat = entropy(all_cats) if all_cats else 1.0
    #H_rating = entropy(all_ratings)
    #H_user = entropy(all_users)
    #Q_diversity = H_cat * H_rating * H_user
    # 逐条计算
    quality_scores = []
    for i, record in enumerate(records):
        # 1. 语义相关性
        Q_semantic = 1 - noise_scores[i].get('semantic_noise', 0.0)
        # 2. 行为一致性（统计+LLM融合）
        ratings = user_ratings[record['user_id']]
        mean_rating = np.mean(ratings) if ratings else 0
        std_rating = np.std(ratings) if ratings else 1
        z = (record['rating'] - mean_rating) / (std_rating + 1e-8)
        C_rating = np.exp(-z**2)
        times = sorted(user_times[record['user_id']])
        if len(times) > 1:
            time_diffs = np.diff(times)
            C_temporal = 1 - (np.std(time_diffs) / (np.mean(time_diffs) + 1e-8))
            C_temporal = max(0, min(1, C_temporal))
        else:
            C_temporal = 1.0
        if item2cat is not None:
            cat = item2cat.get(record['item_id'], None)
            if cat is not None:
                user_cat_hist = user_cats[record['user_id']]
                C_category = user_cat_hist.count(cat) / len(user_cat_hist) if user_cat_hist else 1.0
            else:
                C_category = 1.0
        else:
            C_category = 1.0
        Q_behavior_stat = 0.4*C_rating + 0.3*C_temporal + 0.3*C_category
        behavior_score_llm = 1 - noise_scores[i].get('behavior_noise', 0.0)
        final_behavior_score = 0.3 * Q_behavior_stat + 0.7 * behavior_score_llm
        # 3. 时间逻辑性（类别转移概率近似）
        if item2cat is not None:
            cat_seq = [item2cat.get(it, None) for it in user_items[record['user_id']]]
            cat_seq = [c for c in cat_seq if c is not None]
            if len(cat_seq) > 1:
                trans = 0
                for t in range(len(cat_seq)-1):
                    trans += int(cat_seq[t+1] == cat_seq[t])
                Q_temporal = trans / (len(cat_seq)-1)
            else:
                Q_temporal = 1.0
        else:
            Q_temporal = 1.0
        # 4. 多样性
        # Q_diversity已全局算好
        # 综合
        score = alpha*Q_semantic + beta*final_behavior_score + gamma*Q_temporal
        quality_scores.append(score)
    return quality_scores
