import time
import matplotlib.pyplot as plt
from matrics.senders.vllm_sender import VLLMSender
from matrics.simulators.poisson_simulator import PoissonSimulator
from typing import List, Dict, Any, Optional
import statistics
import numpy as np
from main import config
# vllm serve "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit" --gpu_memory_utilization 0.8 --max_model_len 2048 --host 127.0.0.1 --port 8000

def record_matrics(
		prompt: list,
		request_sender = VLLMSender, # A class or instance of request sender in @senders folder
		simulator = PoissonSimulator, # a class or instance of simulator in @simulators folder
		max_requests: Optional[int] = None, # pyright: ignore[reportArgumentType]
		duration: Optional[float] = None # pyright: ignore[reportArgumentType]
	):
	"""
	记录指标：使用 request_sender 和 simulator 发送请求并记录 GPU 使用情况
	
	Args:
		prompt: 提示文本列表
		request_sender: 请求发送器实例或类（优先接受已实例化的对象，要求有 send_request 方法）
		simulator: 模拟器实例或类（要求有 simulate 方法）
		max_requests: 最大请求数（可选，如果未提供则使用 len(prompt)）
		duration: 模拟总时长（秒，可选）
	
	Returns:
		包含所有请求结果的字典，包括：
		- request_results: 每个请求的结果列表
		- gpu_monitor: GPU 监控器实例（包含 times 和 memory_records）
	"""
	from  matrics.monitors.GPUMonitor import GPUMemoryMonitor
	gpu_monitor = GPUMemoryMonitor()
	
	# 启动 GPU 监控
	gpu_monitor.start()
	
	try:
		# 确定 max_requests 参数
		if max_requests is None and duration is None:
			max_requests = len(prompt)
		
		# 如果传入的是类，进行实例化以获得已绑定方法
		if isinstance(request_sender, type):
			req_sender_instance = request_sender()
		else:
			req_sender_instance = request_sender
		
		if isinstance(simulator, type):
			sim_instance = simulator()
		else:
			sim_instance = simulator
		
		# 确保 send_request 是可调用的已绑定方法
		send_callable = getattr(req_sender_instance, "send_request")
		
		# 使用 simulator 模拟请求发送（传入已绑定的 send_request）
		results = sim_instance.simulate(
			prompt, 
			send_callable, 
			duration=duration, 
			max_requests=max_requests
		) # pyright: ignore[reportCallIssue]
		
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

# 修改：更新 compute_and_plot_metrics，新增 draw_boxplot 函数并调用
def compute_and_plot_metrics(request_results: List[Dict[str, Any]], gpu_monitor: Any, out_prefix: str = ""):
	"""
	从 request_results 和 gpu_monitor 中计算 TTFT, TPOT, latency, throughput，并保存图像。
	新增：调用通用 draw_boxplot 生成不同指标的箱状图（TTFT, TPOT, Throughput）。
	"""
	# 过滤出成功请求并提取时间数据（容错）
	successful = [r for r in request_results if r.get('success', False)]
	first_times_abs = [r.get('first_token_time') for r in successful if r.get('first_token_time') is not None]
	last_times_abs = [r.get('last_token_time') for r in successful if r.get('last_token_time') is not None]
	# latency 优先使用 last-first，否则使用 last
	latencies = []
	for r in successful:
		f = r.get('first_token_time')
		l = r.get('last_token_time')
		if f is not None and l is not None:
			latencies.append(l - f)
		elif l is not None:
			latencies.append(l)
	total_tokens_list = [r.get('total_tokens', 0) for r in successful]
	
	# 统一时间基线：使用 GPU monitor 的 times[0]（假设 GPUMonitor 也使用 perf_counter）
	gp_times = getattr(gpu_monitor, "times", None)
	base_time = gp_times[0] if gp_times and len(gp_times) > 0 else None
	
	# 将绝对 perf_counter 时间转换为相对于 base_time 的相对秒（用于绘图和 throughput）
	if base_time is not None:
		first_times = [ft - base_time for ft in first_times_abs]
		last_times = [lt - base_time for lt in last_times_abs]
	else:
		# 如果没有 gpu monitor 时间，则将最早的事件作为基线（兼容旧逻辑）
		all_abs = [t for t in (first_times_abs + last_times_abs) if t is not None]
		if all_abs:
			anchor = min(all_abs)
			first_times = [ft - anchor for ft in first_times_abs]
			last_times = [lt - anchor for lt in last_times_abs]
		else:
			first_times = []
			last_times = []
	
	tpot = [l - f for f, l in zip(first_times, last_times)] if first_times and last_times else []
	
	# Helper: 简单绘图器（复用已有 plot_graph）
	def _safe_plot(title: str, x_label: str, x_val: list, y_label: str, y_val: list, y_lim: Optional[float], fname: str):
		if not x_val or not y_val:
			print(f"[WARN] 无足够数据绘制 {title}")
			return
		plot_graph(title, x_label, x_val, y_label, y_val, y_lim if y_lim is not None else max(y_val) * 1.1, fname)
	
	# 1) TTFT 时间序列（按序号）
	if first_times:
		xs = list(range(1, len(first_times) + 1))
		_safe_plot("TTFT (Time To First Token)", "请求序号", xs, "秒", first_times, max(first_times) * 1.2, f"{out_prefix}ttft.png")
	
	# 2) TPOT（耗时 = last - first）序列
	if tpot:
		xs = list(range(1, len(tpot) + 1))
		_safe_plot("TPOT (Time Per Output Token approx)", "请求序号", xs, "秒", tpot, max(tpot) * 1.2, f"{out_prefix}tpot.png")
	
	# 3) Latency（完成时间）序列
	if latencies:
		xs = list(range(1, len(latencies) + 1))
		_safe_plot("Completion Latency", "请求序号", xs, "秒", latencies, max(latencies) * 1.2, f"{out_prefix}latency.png")
	
	# 4) Throughput（按 GPU 监控时间点统计每区间完成请求数 / 区间时长）
	throughput_times = []
	throughput_values = []
	if last_times:
		if gp_times and len(gp_times) >= 2:
			# 使用相对 gp_times（以 base_time 为 0）
			rel_gp_times = [t - base_time for t in gp_times]
			for i in range(len(rel_gp_times) - 1):
				t0 = rel_gp_times[i]
				t1 = rel_gp_times[i+1]
				count = sum(1 for lv in last_times if t0 < lv <= t1)
				dt = max(1e-6, t1 - t0)
				throughput_times.append((t0 + t1) / 2.0)
				throughput_values.append(count / dt)
		else:
			max_t = max(last_times)
			step = 0.5
			t = 0.0
			while t < max_t + step:
				t0 = t
				t1 = t + step
				count = sum(1 for lv in last_times if t0 < lv <= t1)
				throughput_times.append((t0 + t1) / 2.0)
				throughput_values.append(count / max(1e-6, (t1 - t0)))
				t += step
	
	if throughput_times and throughput_values:
		_safe_plot("Throughput (requests/sec)", "Time (s)", throughput_times, "Request/second", throughput_values, max(throughput_values) * 1.2, f"{out_prefix}throughput.png")
	
	# 新增：绘制按 token 长度分组的 latency 箱线图（保持兼容性）
	draw_boxplot(latencies, tokens=total_tokens_list, metric_name="latency_by_tokens", out_prefix=f"{out_prefix}latency_boxplot_tokens.png")
	
	# 为 TTFT, TPOT 创建箱状图（优先按照 token 长度分组）
	if first_times:
		draw_boxplot(first_times, tokens=total_tokens_list, metric_name="TTFT", out_prefix=f"{out_prefix}ttft_boxplot.png")
	if tpot:
		draw_boxplot(tpot, tokens=total_tokens_list, metric_name="TPOT", out_prefix=f"{out_prefix}tpot_boxplot.png")
	
	# 为 Throughput 创建箱状图（throughput_values 可能为空）
	if throughput_values:
		draw_boxplot(throughput_values, tokens=None, metric_name="Throughput", out_prefix=f"{out_prefix}throughput_boxplot.png")
	
	print("[✔] 指标图像已生成（若有足够数据）")


# 替换为更通用且健壮的 draw_boxplot
def draw_boxplot(metric_values: List[float],
                 tokens: Optional[List[int]] = None,
                 metric_name: str = "metric",
                 out_prefix: str = None):
    """
    兼容性更强的箱线图函数：
    - metric_values: 数值列表（会自动过滤 NaN/inf）
    - tokens: 可选，同长度时会按 token 分位数分组（short/mid/long）
    - 若 tokens 不可用，则根据 metric_values 自身的 25%/75% 分位数分组三组；当数据量太少时绘制单一箱线图
    - metric_name: 用于文件命名与标题
    - out_prefix: 可选输出文件名（若为 None，则使用 "{metric_name}_boxplot.png"）
    """
    if metric_values is None or len(metric_values) == 0:
        print(f"[WARN] {metric_name}: 无数据可绘制箱线图")
        return

    vals = np.array(metric_values, dtype=float)
    # 过滤非有限值
    finite_mask = np.isfinite(vals)
    if not np.any(finite_mask):
        print(f"[WARN] {metric_name}: 所有值均为非有限（NaN/inf）")
        return
    vals = vals[finite_mask]

    toks = None
    if tokens is not None:
        if len(tokens) != len(metric_values):
            print(f"[WARN] {metric_name}: tokens 数量与 metric_values 长度不匹配，忽略 tokens 分组")
            tokens = None
        else:
            toks = np.array(tokens, dtype=float)[finite_mask]

    groups = {}
    # 当提供 tokens 时，按 tokens 的 25%/75% 分位数分组
    if toks is not None and len(toks) > 0:
        q1 = np.percentile(toks, 25)
        q3 = np.percentile(toks, 75)
        groups[f"short≤{int(q1)}"] = vals[toks <= q1]
        groups[f"mid({int(q1)}-{int(q3)})"] = vals[(toks > q1) & (toks <= q3)]
        groups[f"long>{int(q3)}"] = vals[toks > q3]
    else:
        # 没有 tokens，尝试根据 metric 自身分位数分组（当数据量足够）
        if len(vals) >= 6:
            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)
            groups[f"low≤{round(q1,3)}"] = vals[vals <= q1]
            groups[f"mid({round(q1,3)}-{round(q3,3)})"] = vals[(vals > q1) & (vals <= q3)]
            groups[f"high>{round(q3,3)}"] = vals[vals > q3]
        else:
            # 数据太少，直接作为一个组
            groups["all"] = vals

    # 过滤空组
    labels = []
    data = []
    for k, v in groups.items():
        if v is not None and len(v) > 0:
            labels.append(k)
            data.append(np.asarray(v, dtype=float))

    if not data:
        print(f"[WARN] {metric_name}: 分组后无数据可绘制")
        return

    # 构建输出路径/文件名
    fname = out_prefix if out_prefix else f"{metric_name}_boxplot.png"
    # 确保 out_prefix 若为相对名时符合已有 config 输出路径
    out_path = (config.output_path + fname) if not fname.startswith(config.output_path) else fname

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.title(f"{metric_name} distribution by groups")
    plt.ylabel(metric_name)
    plt.xlabel("Groups")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[✔] {metric_name} boxplot saved to: {out_path}")

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
    plt.savefig(config.output_path + file_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[✔] {title} have been saved to: {file_name}")
