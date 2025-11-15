import time
import requests
import json
from typing import Dict


class VLLMSender:
    """
    发送请求到 vLLM 本地服务器并返回指标
    至少包括：发送请求的时间戳、第一个token返回的时间、最后一个token返回的时间
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化 vLLM 发送器
        
        Args:
            base_url: vLLM 服务器的地址，默认为 http://localhost:8000
        """
        self.base_url = base_url.rstrip('/')
        self.model_id = None
        self._get_model_id()
        # Instruct 模型使用 chat completions 端点
        self.api_endpoint = f"{self.base_url}/v1/chat/completions"
    
    def _get_model_id(self):
        """从服务器获取模型 ID"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                if 'data' in models and len(models['data']) > 0:
                    self.model_id = models['data'][0]['id']
                    return
        except:
            pass
        # 如果获取失败，使用默认值
        self.model_id = "default"
    
    def send_request(
        self, 
        prompt: str, 
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95
    ) -> Dict:
        """
        发送请求到 vLLM 服务器并记录指标
        
        Args:
            prompt: 输入的提示文本
            max_tokens: 最大生成token数
            temperature: 采样温度
            top_p: top-p 采样参数
            
        Returns:
            包含以下键的字典：
            - request_timestamp: 发送请求的时间戳
            - first_token_time: 第一个token返回的时间（相对于request_timestamp）
            - last_token_time: 最后一个token返回的时间（相对于request_timestamp）
            - total_tokens: 生成的token总数
            - response_text: 完整的响应文本
            - success: 请求是否成功
        """
        request_timestamp = time.time()
        request_start = time.perf_counter()  # 使用高精度计时器测量相对时间
        first_token_time = None
        last_token_time = None
        response_text = ""
        total_tokens = 0
        success = False
        
        try:
            # 使用 chat completions 格式（适合 instruct 模型）
            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": True
            }
            
            # 流式响应：可以更准确地记录首token时间
            response = requests.post(
                self.api_endpoint,
                json=payload,
                stream=True,
                timeout=300
            )
            response.raise_for_status()
            
            first_token_received = False
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # 移除 'data: ' 前缀
                        if data_str.strip() == '[DONE]':
                            break
                        
                        # 在解析JSON之前记录时间，以获得更精确的首token时间
                        current_time = time.perf_counter()
                        line_received_time = current_time - request_start
                        
                        try:
                            data = json.loads(data_str)
                            
                            # 检查是否包含实际内容（非空delta或message）
                            has_content = False
                            if 'choices' in data and len(data['choices']) > 0:
                                choice = data['choices'][0]
                                # 检查delta中的content（流式响应）
                                if 'delta' in choice:
                                    content = choice['delta'].get('content', '')
                                    if content:  # 确保内容非空
                                        has_content = True
                                        response_text += content
                                # 检查message中的content（某些情况下可能使用）
                                elif 'message' in choice:
                                    content = choice['message'].get('content', '')
                                    if content:  # 确保内容非空
                                        has_content = True
                                        response_text += content
                                
                                if 'usage' in data:
                                    total_tokens = data['usage'].get('total_tokens', 0)
                            
                            # 记录首token时间（仅在首次收到实际内容时）
                            if not first_token_received and has_content:
                                first_token_received = True
                                first_token_time = line_received_time
                            
                            # 更新最后token时间（每次收到内容时更新）
                            if has_content:
                                last_token_time = line_received_time
                            
                        except json.JSONDecodeError:
                            continue
            
            # 如果没有收到任何token，设置默认值
            if first_token_time is None:
                first_token_time = last_token_time if last_token_time else 0
            if last_token_time is None:
                last_token_time = 0
            
            success = True
            
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] vLLM 请求失败: {e}")
            first_token_time = first_token_time or 0
            last_token_time = last_token_time or 0
        except Exception as e:
            print(f"[ERROR] 处理响应时出错: {e}")
            first_token_time = first_token_time or 0
            last_token_time = last_token_time or 0
        
        return {
            "request_timestamp": request_timestamp,
            "first_token_time": first_token_time,
            "last_token_time": last_token_time,
            "total_tokens": len(response_text),
            # "total_tokens": total_tokens,
            "response_text": response_text,
            "success": success
        }
