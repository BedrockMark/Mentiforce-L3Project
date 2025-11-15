import time
import threading
from pynvml import *
from matrics import plot_graph


class GPUMemoryMonitor:
    def __init__(self, gpu_id=0, interval=0.5):
        self.gpu_id = gpu_id
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self.times = []
        self.memory_records = []
        print("GPU memory monitor initialized!")

    def _record(self):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(self.gpu_id)
        
        device_name = nvmlDeviceGetName(handle)
        info = nvmlDeviceGetMemoryInfo(handle)
        total_mem = info.used / 1024**2
        print(f"[Monitor Begins] GPU {self.gpu_id}: {device_name}")

        start = time.time()
        while not self._stop_event.is_set():
            info = nvmlDeviceGetMemoryInfo(handle)
            util = nvmlDeviceGetUtilizationRates(handle)
            self.memory_records.append((info.used / 1024**2)*util.memory/100)
            self.times.append(time.time() - start)
            time.sleep(self.interval)
        
        plot_graph(
            f"GPU {self.gpu_id} Memory vs. Time (for {device_name})",
            "Time (s)",
            self.times,
            "vRAM (MB)",
            self.memory_records,
            total_mem,
            "gpu_usage.png"
        )

    def start(self):
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        