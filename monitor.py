from pynvml import *
from threading import Thread
import time
import pynvml

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        nvmlInit()
        self.stopped = False
        self.delay = delay / 1000000.0
        self.handle = nvmlDeviceGetHandleByIndex(0)

        self.peak_gpu_util = 0
        self.peak_mem_util = 0

        self.start()
        self.stime = time.process_time()

    def run(self):
        while not self.stopped:
            utilization = nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util = utilization.gpu
            mem_util = utilization.memory

            if gpu_util > self.peak_gpu_util:
                self.peak_gpu_util = gpu_util
            if mem_util > self.peak_mem_util:
                self.peak_mem_util = mem_util

            time.sleep(self.delay)
    
    def stop(self):
        self.stopped = True
        etime = time.process_time()
        latency = etime - self.stime
        platform = nvmlDeviceGetName(self.handle).decode()
        return (platform, latency, self.peak_gpu_util, self.peak_mem_util)


if __name__ == "__main__":
    nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpuinfo = nvmlDeviceGetUtilizationRates(handle)

    platform = nvmlDeviceGetName(handle).decode()

    print('platform: ', platform, ", ", type(platform))
    print("meminfo: ", meminfo)
    print("gpuinfo: ", gpuinfo)
    print(gpuinfo.gpu)
    help(platform)