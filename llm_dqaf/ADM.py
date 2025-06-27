def adaptive_denoising(records, noise_scores, quality_scores, llm_model, threshold=0.5, strategy='hard'):
    """
    records: 原始数据列表
    noise_scores: SND输出
    quality_scores: MQA输出
    llm_model: DeepSeekLLM实例
    threshold: 过滤阈值
    strategy: 'hard'（硬过滤）或'soft'（软加权）
    返回: clean_data, quality_report
    """
    clean_data = []
    report = []
    for i, record in enumerate(records):
        info = {
            'record': record,
            'quality': quality_scores[i],
            'noise': noise_scores[i],
            'explanation': noise_scores[i].get('explanation', '')
        }
        if strategy == 'hard':
            if quality_scores[i] >= threshold:
                clean_data.append(record)
        elif strategy == 'soft':
            record = record.copy()
            record['weight'] = quality_scores[i]
            clean_data.append(record)
        report.append(info)
    return clean_data, report 