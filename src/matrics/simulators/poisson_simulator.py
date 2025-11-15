import time
import numpy as np
from typing import Callable, List, Dict, Any, Optional


class PoissonSimulator:
    """
    使用泊松过程模拟请求发送
    泊松过程：在时间间隔 [0, T] 内，事件发生的次数服从泊松分布
    事件之间的间隔时间服从指数分布
    """
    
    def __init__(self, lambda_rate: float = 1.0):
        """
        初始化泊松模拟器
        
        Args:
            lambda_rate: 泊松过程的速率参数（每秒请求数）
        """
        self.lambda_rate = lambda_rate
    
    def _generate_arrival_times(self, duration: float = None, max_count: int = None) -> List[float]:
        """
        生成泊松过程的到达时间点
        
        Args:
            duration: 模拟的总时长（秒）
            max_count: 最大生成数量
            
        Returns:
            到达时间点列表（从0开始的绝对时间，单位：秒）
        """
        arrival_times = []
        current_time = 0.0
        
        while True:
            # 指数分布的间隔时间：-ln(U) / lambda，其中 U ~ Uniform(0,1)
            interarrival = np.random.exponential(1.0 / self.lambda_rate)
            current_time += interarrival
            
            if duration is not None and current_time >= duration:
                break
            if max_count is not None and len(arrival_times) >= max_count:
                break
                
            arrival_times.append(current_time)
        
        return arrival_times
    
    def simulate(
        self,
        prompts: List[str],
        sender: Callable[[str], Dict[str, Any]],
        duration: Optional[float] = None,
        max_requests: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        使用泊松过程模拟请求发送，并返回sender提供的结果
        
        Args:
            prompts: 提示文本列表（会循环使用）
            sender: 发送请求的函数，接受一个prompt字符串，返回包含指标的字典
            duration: 模拟的总时长（秒）。如果提供，将使用此参数
            max_requests: 最大请求数。如果提供，将使用此参数。如果两者都提供，以先达到的为准
            
        Returns:
            每个请求的结果列表，每个结果是一个包含sender返回的指标的字典
        """
        if duration is None and max_requests is None:
            raise ValueError("必须提供 duration 或 max_requests 参数之一")
        
        # 生成到达时间
        arrival_times = self._generate_arrival_times(duration, max_requests)
        
        # 执行模拟
        results = []
        prompt_index = 0
        start_time = time.time()
        
        for arrival_time in arrival_times:
            # 等待到到达时间
            elapsed = time.time() - start_time
            if elapsed < arrival_time:
                time.sleep(arrival_time - elapsed)
            
            # 发送请求
            prompt = prompts[prompt_index % len(prompts)]
            result = sender(prompt)
            result['arrival_time'] = arrival_time
            results.append(result)
            
            prompt_index += 1
            
            # 检查是否达到最大请求数限制
            if max_requests is not None and len(results) >= max_requests:
                break
        
        return results
