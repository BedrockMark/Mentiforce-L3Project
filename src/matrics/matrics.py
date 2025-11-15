import time
import matplotlib.pyplot as plt
from senders.vllm_sender import VLLMSender
from simulators.poisson_simulator import PoissonSimulator

# vllm serve "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit" --gpu_memory_utilization 0.8 --max_model_len 2048 --host 127.0.0.1 --port 8000

def record_matrics(
        prompt: list,
        request_sender = VLLMSender, # A class of request sender in @senders folder
        simulator = PoissonSimulator, # a class of simulator in @simulators folder
        max_requests: int = None,
        duration: float = None
    ):
    """
    记录指标：使用 request_sender 和 simulator 发送请求并记录 GPU 使用情况
    
    Args:
        prompt: 提示文本列表
        request_sender: 请求发送器类（来自 @senders 文件夹）
        simulator: 模拟器类（来自 @simulators 文件夹）
        max_requests: 最大请求数（可选，如果未提供则使用 len(prompt)）
        duration: 模拟总时长（秒，可选）
    
    Returns:
        包含所有请求结果的字典，包括：
        - request_results: 每个请求的结果列表
        - gpu_monitor: GPU 监控器实例（包含 times 和 memory_records）
    """
    from monitors.GPUMonitor import GPUMemoryMonitor
    gpu_monitor = GPUMemoryMonitor()
    
    # 启动 GPU 监控
    gpu_monitor.start()
    
    try:
        # 确定 max_requests 参数
        if max_requests is None and duration is None:
            max_requests = len(prompt)
        
        # 使用 simulator 模拟请求发送
        results = simulator.simulate(
            prompt, 
            request_sender.send_request, 
            duration=duration, 
            max_requests=max_requests
        )
        
        # 停止 GPU 监控
        gpu_monitor.stop()
        
        return {
            'request_results': results,
            'gpu_monitor': gpu_monitor
        }
    
    except Exception as e:
        # 确保即使出错也停止监控
        gpu_monitor.stop()
        print(f"[ERROR] 记录指标时出错: {e}")
        raise


def plot_graph(
    title: str,
    x_label: str,
    x_val:list,
    y_label: str,
    y_val:list,
    y_lim:float,
    file_name:str
):
    plt.figure(figsize=(7, 4))
    plt.title(title)
    print(x_val, y_val)
    plt.plot(x_val, y_val, label=title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0, y_lim)
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[✔] {title} have been saved to: {file_name}")


def main():
    """
    测试函数：演示如何使用 record_matrics 函数记录指标
    """
    print("=" * 70)
    print("开始测试指标记录系统")
    print("=" * 70)
    
    # 配置参数
    vllm_base_url = "http://127.0.0.1:8000"
    lambda_rate = 3  # 每秒 0.5 个请求（每2秒一个请求）
    max_requests = None  # 最多发送5个请求
    duration = 10  # 不限制时长，使用 max_requests
    
    # 测试提示列表
    test_prompts = [
        "What is the capital of France?"*5,
        "Explain quantum computing in simple terms."*3,
        "Write a haiku about artificial intelligence."*3,
        "What are the main differences between Python and Java?"*3,
        "Describe the process of photosynthesis."*5,
    ]
    
    print(f"\n[配置]")
    print(f"  vLLM 服务器地址: {vllm_base_url}")
    print(f"  泊松过程速率: {lambda_rate} 请求/秒")
    print(f"  最大请求数: {max_requests}")
    print(f"  测试提示数量: {len(test_prompts)}")
    
    # 创建请求发送器
    print(f"\n[初始化] 创建 vLLM 发送器...")
    sender = VLLMSender(base_url=vllm_base_url)
    
    # 创建泊松模拟器
    print(f"[初始化] 创建泊松模拟器...")
    simulator = PoissonSimulator(lambda_rate=lambda_rate)
    
    # 记录指标
    print(f"\n[开始] 开始记录指标...")
    print(f"  使用模型: {sender.model_id}")
    print(f"  使用端点: {sender.api_endpoint}\n")
    
    try:
        start_time = time.time()
        results = record_matrics(
            prompt=test_prompts,
            request_sender=sender,
            simulator=simulator,
            max_requests=max_requests,
            duration=duration
        )
        end_time = time.time()
        
        request_results = results['request_results']
        gpu_monitor = results['gpu_monitor']
        
        print(f"\n[完成] 指标记录完成！")
        print(f"  总耗时: {end_time - start_time:.2f} 秒")
        print(f"  成功请求数: {sum(1 for r in request_results if r.get('success', False))}")
        print(f"  失败请求数: {sum(1 for r in request_results if not r.get('success', False))}")
        
        # 打印详细统计信息
        if request_results:
            print(f"\n[统计信息]")
            successful_results = [r for r in request_results if r.get('success', False)]
            
            if successful_results:
                first_token_times = [r['first_token_time'] for r in successful_results]
                last_token_times = [r['last_token_time'] for r in successful_results]
                total_tokens = [r.get('total_tokens', 0) for r in successful_results]
                
                print(f"  首token平均延迟: {sum(first_token_times) / len(first_token_times):.3f} 秒")
                print(f"  首token最小延迟: {min(first_token_times):.3f} 秒")
                print(f"  首token最大延迟: {max(first_token_times):.3f} 秒")
                print(f"  完成平均延迟: {sum(last_token_times) / len(last_token_times):.3f} 秒")
                print(f"  完成最小延迟: {min(last_token_times):.3f} 秒")
                print(f"  完成最大延迟: {max(last_token_times):.3f} 秒")
                print(f"  总生成token数: {sum(total_tokens)}")
                print(f"  平均每个请求token数: {sum(total_tokens) / len(total_tokens):.1f}")
            
            # 打印每个请求的简要信息
            print(f"\n[请求详情]")
            for i, result in enumerate(request_results, 1):
                status = "✓" if result.get('success', False) else "✗"
                first_token = result.get('first_token_time', 0)
                last_token = result.get('last_token_time', 0)
                tokens = result.get('total_tokens', 0)
                print(f"  请求 {i}: {status} | 首token: {first_token:.3f}s | 完成: {last_token:.3f}s | tokens: {tokens}")
        
        # GPU 监控信息
        if gpu_monitor.memory_records:
            print(f"\n[GPU 监控]")
            print(f"  记录点数: {len(gpu_monitor.memory_records)}")
            print(f"  平均GPU内存使用: {sum(gpu_monitor.memory_records) / len(gpu_monitor.memory_records):.2f} MB")
            print(f"  最大GPU内存使用: {max(gpu_monitor.memory_records):.2f} MB")
            print(f"  GPU使用图表已保存到: gpu_usage.png")
        
        print("\n" + "=" * 70)
        print("测试完成！")
        print("=" * 70)
        
        return results
        
    except Exception as e:
        print(f"\n[错误] 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
