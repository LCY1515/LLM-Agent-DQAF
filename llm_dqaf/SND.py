from .llm_utils import DeepSeekLLM
from collections import defaultdict
import json
import re

noise_detection_prompt = """
你是一个推荐系统的数据质量专家。请分析以下用户行为的合理性：

用户历史评分（含时间戳）：{user_history}
当前行为：物品ID {item_id}，评分 {rating}，时间 {timestamp}
统计信息：
- 历史评分中对该物品的评分次数：{dup_count}
- 当前行为与上一条评分的时间间隔（秒）：{time_gap}
- 该用户1小时内评分总数：{freq_count}
- 当前评分是否为极端分（1或5）：{is_extreme}

请结合上述统计，判断该行为是否为异常刷分、重复评分、极端评分等噪声行为，并从以下维度评估噪声程度（0-1，越高表示越可能是噪声）：
1. 语义一致性：当前物品与用户历史偏好的语义匹配度
2. 行为异常性：评分是否偏离用户正常评分模式，是否存在短时间高频、重复、极端评分等异常
3. 内容匹配性：评分是否与物品客观质量相符（如无内容信息可忽略）

请严格按照以下JSON格式输出：
{{
    "semantic_noise": 0.x,
    "behavior_noise": 0.x,
    "content_noise": 0.x,
    "explanation": "详细解释噪声判断依据"
}}
"""

def semantic_noise_detection(records, llm_model: DeepSeekLLM):
    """
    输入: records - List[dict]，每条包含user_id, item_id, rating, timestamp等
         llm_model - DeepSeekLLM实例
    输出: List[dict]，每条记录的噪声分数 {"semantic_noise": x, "behavior_noise": y, "content_noise": z, "explanation": str}
    """
    # 构建用户历史评分缓存
    user_history = defaultdict(list)
    for record in records:
        user_history[record['user_id']].append(record)

    results = []
    for record in records:
        # 构造用户历史评分（去除当前行为）
        history = [f"(item:{r['item_id']}, rating:{r['rating']}, time:{r['timestamp']})" for r in user_history[record['user_id']] if r != record]
        history_str = ', '.join(history) if history else '无'
        # 统计特征
        dup_count = sum(1 for r in user_history[record['user_id']] if r['item_id'] == record['item_id'] and r != record)
        user_times = sorted([r['timestamp'] for r in user_history[record['user_id']] if r != record])
        if user_times:
            prev_times = [t for t in user_times if t < record['timestamp']]
            if prev_times:
                time_gap = record['timestamp'] - max(prev_times)
            else:
                time_gap = -1
        else:
            time_gap = -1
        freq_count = sum(1 for r in user_history[record['user_id']] if abs(r['timestamp'] - record['timestamp']) <= 3600 and r != record)
        is_extreme = int(record['rating'] == 1 or record['rating'] == 5)
        prompt = noise_detection_prompt.format(
            user_history=history_str,
            item_id=record['item_id'],
            rating=record['rating'],
            timestamp=record['timestamp'],
            dup_count=dup_count,
            time_gap=time_gap,
            freq_count=freq_count,
            is_extreme='是' if is_extreme else '否'
        )
        response = llm_model.chat(prompt)
        try:
            match = re.search(r"\{[\s\S]*\}", response)
            if match:
                json_str = match.group(0)
                score = json.loads(json_str)
            else:
                score = {"semantic_noise": 0.0, "behavior_noise": 0.0, "content_noise": 0.0, "explanation": "LLM输出无JSON，默认低噪声。"}
        except Exception as e:
            print("JSON解析异常:", e)
            score = {"semantic_noise": 0.0, "behavior_noise": 0.0, "content_noise": 0.0, "explanation": "LLM输出解析失败，默认低噪声。"}
        print('LLM prompt:', prompt)
        print('LLM response:', response)
        print('Parsed score:', score)
        results.append(score)
    return results
