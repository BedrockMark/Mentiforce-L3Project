import time
from util import *
from src.config import GlobalConfig
from custom_schedulers.naive_scheduler import NaiveScheduler
from vllm import LLM, SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler


########################################
# 1. Custom parameters
########################################
config = GlobalConfig()

########################################
# 2. Benchmark runner
########################################
def run_benchmark(engine: LLM, name: str):
    
    prompts = read_first_column("test/test/astronomy_test.csv")
    # for file_path in read_all_files("test/test/"): # TODO: Extend it to fit the config.
    #     prompts.extend(read_first_column(file_path))

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


########################################
# 4. Run both benchmarks
########################################

if __name__ == "__main__":

    # Default scheduler benchmark
    engine_default = create_engine()
    metrics_default, _ = run_benchmark(engine_default, "default")

    # Custom scheduler benchmark
    engine_custom = create_engine(NaiveScheduler)
    metrics_custom, _ = run_benchmark(engine_custom, "custom")
    
    print("-"*70)
    print("Default scheduler:", metrics_default)
    print("Custom scheduler:", metrics_custom)
    #Default scheduler: {'name': 'default', 'elapsed_time': 13.888915300369263, 'num_requests': 3, 'avg_latency': 4.629638433456421}
