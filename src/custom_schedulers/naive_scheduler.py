import time

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.output import SchedulerOutput, NewRequestData
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.request import RequestStatus

class NaiveScheduler(Scheduler):
    def schedule(self) -> SchedulerOutput:
        # FIFO: 只按请求到达顺序调度，直到资源耗尽
        scheduled_new_reqs = []
        scheduled_running_reqs = []
        scheduled_resumed_reqs = []
        preempted_reqs = []

        req_to_new_blocks = {}
        num_scheduled_tokens = {}
        token_budget = self.max_num_scheduled_tokens

        # 只调度 RUNNING 队列中的请求（保持顺序）
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            num_new_tokens = (request.num_tokens_with_spec +
                              request.num_output_placeholders -
                              request.num_computed_tokens)
            num_new_tokens = min(num_new_tokens, token_budget)
            num_new_tokens = min(num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens)
            if num_new_tokens == 0:
                req_index += 1
                continue
            new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
            if new_blocks is None:
                break
            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

        # FIFO: 只要有资源就从 waiting 队列按顺序调度新请求
        while self.waiting and token_budget > 0 and len(self.running) < self.max_num_running_reqs:
            request = self.waiting.pop_request()
            num_new_tokens = request.num_tokens
            num_new_tokens = min(num_new_tokens, token_budget)
            num_new_tokens = min(num_new_tokens, self.max_model_len - 1)
            if num_new_tokens == 0:
                continue
            new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
            if new_blocks is None:
                break
            self.running.append(request)
            scheduled_new_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = 0

        # 构造输出
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids())
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = CachedRequestData(
            req_ids=[req.request_id for req in scheduled_running_reqs],
            resumed_from_preemption=[False] * len(scheduled_running_reqs),
            new_token_ids=[[] for _ in scheduled_running_reqs],
            new_block_ids=[req_to_new_blocks[req.request_id].get_block_ids(allow_none=True)
                           for req in scheduled_running_reqs],
            num_computed_tokens=[req.num_computed_tokens for req in scheduled_running_reqs],
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * len(self.kv_cache_config.kv_cache_groups),
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )
        self._update_after_schedule(scheduler_output)
        return scheduler_output

if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    print("="*70)
    print("Testing NaiveScheduler with vLLM LLMEngine")
    print("="*70)
    
    # Define model and parameters
    model_name = "Qwen/Qwen3-4B-AWQ"
    try:
        llm = LLM(
            model=model_name,
            tokenizer=model_name,
            quantization="awq",
            max_model_len=2048,
            max_num_seqs=4,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=0.8,
            scheduler_cls=NaiveScheduler,  # Use our custom scheduler
        )
        # Define test prompts
        prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a haiku about artificial intelligence.",
            "What are the main differences between Python and Java?",
        ]
        # Define sampling parameters
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=128,
        )
        print(f"\n📝 Generating responses for {len(prompts)} prompts...")
        print("-"*70)
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\n🔸 Prompt {i+1}:")
            print(f"   {prompt}")
            print(f"\n🔹 Generated:")
            print(f"   {generated_text}")
            print("-"*70)
        total_time = end_time - start_time
        print(f"\n📊 Statistics:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average time per prompt: {total_time/len(prompts):.2f}s")
        print(f"   Prompts processed: {len(outputs)}")
        print("\n✅ Test completed successfully with NaiveScheduler!")
    except Exception as e:
        print(f"\n❌ Error: {e}")