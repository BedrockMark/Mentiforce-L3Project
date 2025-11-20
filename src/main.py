import time
from util import *
from src.config import GlobalConfig
from custom_schedulers.naive_scheduler import NaiveScheduler
from vllm import LLM, SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from matrics.matrics import *

########################################
# 1. Custom parameters
########################################
config = GlobalConfig()

########################################
# 2. Benchmark runner
########################################
def run_benchmark():
    print("=" * 70)
    print("开始测试指标记录系统")
    print("=" * 70)
    
    # 配置参数
    vllm_base_url = "http://127.0.0.1:8000"
    lambda_rate = 3  # 每秒 0.5 个请求（每2秒一个请求）
    max_requests = None  # 最多发送5个请求
    duration = 30  # 不限制时长，使用 max_requests
    
    # 测试提示列表
    test_prompts = read_first_column("test/test/astronomy_test.csv")
    
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
            request_sender=sender, # type: ignore
            simulator=simulator, # type: ignore
            max_requests=max_requests, # type: ignore
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
        
        # 新增：根据请求结果生成并保存 TTFT, TPOT, latency, throughput 图
        compute_and_plot_metrics(request_results, gpu_monitor, out_prefix="")

        print("\n" + "=" * 70)
        print("测试完成！")
        print("=" * 70)
        
        return results
        
    except Exception as e:
        print(f"\n[错误] 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise

'''
def run_benchmark(engine: LLM, name: str):
    
    prompts = read_first_column("test/test/astronomy_test.csv")
    for file_path in read_all_files("test/test/"): # TODO: Extend it to fit the config.
        prompts.extend(read_first_column(file_path))

    # sampling = SamplingParams(max_tokens=256)
    # --- TODO: random input ---
    t_start = time.time()
    output = engine.generate(prompts) #, sampling)

    elapsed = time.time() - t_start

    metrics = {
        "name": name,
        "elapsed_time": elapsed,
        "num_requests": len(prompts),
        "avg_latency": elapsed / len(prompts),
    }

    return metrics, output
'''

########################################
# 4. Run both benchmarks
########################################

if __name__ == "__main__":
    run_benchmark()