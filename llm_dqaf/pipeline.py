from .two_stage_filter import two_stage_noise_detection
from .MQA import multi_dim_quality_assessment
from .ADM import adaptive_denoising

def llm_dqaf_pipeline(raw_data, llm_model, strategy='hard', threshold=0.5):
    """
    LLM-DQAF主流程（两阶段过滤版本）
    输入: raw_data - List[dict]，llm_model - QwenLLM实例
    返回: clean_data, quality_report
    """
    # 两阶段语义噪声检测
    noise_scores = two_stage_noise_detection(raw_data, llm_model)
    # 质量评估
    quality_scores = multi_dim_quality_assessment(raw_data, noise_scores, llm_model)
    # 自适应去噪
    clean_data, quality_report = adaptive_denoising(raw_data, noise_scores, quality_scores, llm_model, threshold=threshold, strategy=strategy)
    return clean_data, quality_report 