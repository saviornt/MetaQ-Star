import psutil
import time
from collections import deque
from typing import Optional
from src.settings import ResourceSettings
from src.utils.logger import setup_logger

try:
    from pynvml import (
        nvmlInit, nvmlDeviceGetHandleByIndex, 
        nvmlDeviceGetUtilizationRates, 
        nvmlDeviceGetCount, nvmlShutdown
    )
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

resource_logger = setup_logger(__name__)

class ResourceManager:
    def __init__(self, config: Optional[ResourceSettings] = None, **kwargs):
        """
        Initialize the resource manager with a ResourceConfig instance or kwargs.
        If config is not provided, create a default ResourceConfig and override fields with kwargs.
        
        If enabled=False is in the final config, the resource manager does nothing (no pruning).
        """
        if config is None:
            # Create default config and override with kwargs
            config_values = {}
            # Instead of __fields__, use model_fields
            for field_name, field_def in ResourceSettings.model_fields.items():
                if field_name in kwargs:
                    config_values[field_name] = kwargs[field_name]
            config = ResourceSettings(**config_values)
        else:
            # If config is provided, but kwargs also provided, override
            # Instead of dict(), use model_dump()
            config_dict = config.model_dump()
            for k, v in kwargs.items():
                if k in config_dict:
                    config_dict[k] = v
            config = ResourceSettings(**config_dict)
        
        self.config = config
        self.enabled = self.config.enabled
        self.check_interval = self.config.check_interval
        self.rolling_window_size = self.config.rolling_window_size
        self.target_utilization = self.config.target_utilization

        self.observation_period = self.config.observation_period
        self.observation_end_time = time.time() + self.observation_period

        # Queues to store recent samples
        self.cpu_samples = deque(maxlen=self.rolling_window_size)
        self.mem_samples = deque(maxlen=self.rolling_window_size)
        self.gpu_samples = deque(maxlen=self.rolling_window_size)
        self.net_samples = deque(maxlen=self.rolling_window_size)

        self.gpu_handles = []
        if NVML_AVAILABLE:
            nvmlInit()
            device_count = nvmlDeviceGetCount()
            for i in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                self.gpu_handles.append(handle)

        self.last_net_io = psutil.net_io_counters()
        self.initialized_net = False

        self.cpu_threshold = None
        self.mem_threshold = None
        self.gpu_threshold = None
        self.net_threshold = None

    def close(self):
        if NVML_AVAILABLE:
            nvmlShutdown()

    def _get_gpu_usage(self):
        if not self.gpu_handles:
            return 0.0
        usages = []
        for handle in self.gpu_handles:
            util = nvmlDeviceGetUtilizationRates(handle)
            usages.append(util.gpu)  # GPU usage in %
        return float(max(usages)) if usages else 0.0

    def _get_network_usage(self):
        current_net_io = psutil.net_io_counters()
        if not self.initialized_net:
            self.last_net_io = current_net_io
            self.initialized_net = True
            return 0.0

        bytes_sent = current_net_io.bytes_sent - self.last_net_io.bytes_sent
        bytes_recv = current_net_io.bytes_recv - self.last_net_io.bytes_recv
        self.last_net_io = current_net_io
        total_bytes = bytes_sent + bytes_recv
        return float(total_bytes)

    def _update_samples(self):
        process = psutil.Process()
        mem_info = process.memory_info()
        total_mem = psutil.virtual_memory().total
        mem_usage_fraction = mem_info.rss / total_mem

        cpu_usage = psutil.cpu_percent(interval=None)  # % CPU usage
        gpu_usage = self._get_gpu_usage() if NVML_AVAILABLE else 0.0
        net_usage = self._get_network_usage()

        self.cpu_samples.append(cpu_usage / 100.0)
        self.mem_samples.append(mem_usage_fraction)
        self.gpu_samples.append(gpu_usage / 100.0)
        self.net_samples.append(net_usage)

    def _compute_rolling_average(self, samples):
        if samples:
            return sum(samples) / len(samples)
        return 0.0

    def _adjust_thresholds(self):
        avg_cpu = self._compute_rolling_average(self.cpu_samples)
        avg_mem = self._compute_rolling_average(self.mem_samples)
        avg_gpu = self._compute_rolling_average(self.gpu_samples)
        avg_net = self._compute_rolling_average(self.net_samples)

        def safe_threshold(avg):
            # If no meaningful baseline, allow full utilization
            if avg < 0.05:
                return 1.0
            return self.target_utilization * avg

        self.cpu_threshold = safe_threshold(avg_cpu)
        self.mem_threshold = safe_threshold(avg_mem)
        self.gpu_threshold = safe_threshold(avg_gpu)
        self.net_threshold = safe_threshold(avg_net)

        resource_logger.info(
            f"Adjusted thresholds: CPU={self.cpu_threshold*100:.2f}%, "
            f"MEM={self.mem_threshold*100:.2f}%, "
            f"GPU={self.gpu_threshold*100:.2f}%, "
            f"NET≈{self.net_threshold:.2f} bytes"
        )

    def should_prune(self) -> bool:
        if not self.enabled:
            # If resource manager is disabled, never prune.
            return False

        current_time = time.time()
        self._update_samples()

        if current_time < self.observation_end_time:
            return False

        if self.cpu_threshold is None:
            self._adjust_thresholds()
            return False

        self._adjust_thresholds()

        current_cpu = self.cpu_samples[-1]
        current_mem = self.mem_samples[-1]
        current_gpu = self.gpu_samples[-1]
        current_net = self.net_samples[-1]

        resource_logger.debug(
            f"Check prune: CPU={current_cpu*100:.2f}%/thr={self.cpu_threshold*100:.2f}% "
            f"MEM={current_mem*100:.2f}%/thr={self.mem_threshold*100:.2f}% "
            f"GPU={current_gpu*100:.2f}%/thr={self.gpu_threshold*100:.2f}% "
            f"NET={current_net:.2f}B/thr={self.net_threshold:.2f}B"
        )

        prune = (
            (current_cpu > self.cpu_threshold) or
            (current_mem > self.mem_threshold) or
            (current_gpu > self.gpu_threshold) or
            (current_net > self.net_threshold)
        )

        return prune
