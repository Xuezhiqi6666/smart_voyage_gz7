import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    def __init__(self):
        self.base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        self.api_key = os.getenv('DASHSCOPE_API_KEY','sk-67320312aa3e4f16assdfsess0d7')
        self.model_name = 'qwen3.6-flash'