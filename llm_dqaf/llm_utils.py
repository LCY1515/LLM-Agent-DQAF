import os
from openai import OpenAI

class DeepSeekLLM:
    def __init__(self, api_key, base_url="https://api.deepseek.com", model="deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, prompt, system_prompt="You are a helpful assistant"):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content

class QwenLLM:
    def __init__(self, api_key, model="qwen-plus", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, prompt, system_prompt="你是一个推荐系统的数据质量专家"):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1
        )
        return response.choices[0].message.content 