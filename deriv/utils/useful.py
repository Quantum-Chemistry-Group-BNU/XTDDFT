import time
from threading import Thread

from psutil import Process


class MemoryMonitor(Thread):
    """Monitor the memory usage in MB in a separate thread.

    Note that this class is good enough to highlight the memory profile of
    Parallel in this example, but is not a general purpose profiler fit for
    all cases.

    Usage:
        monitor = MemoryMonitor()

        ......  # some code

        # Report memory usage
        monitor.join()
        peak = max(monitor.memory_buffer) / 1e6
    """

    def __init__(self):
        super().__init__()
        self.stop = False
        self.memory_buffer = []
        self.start()

    def get_memory(self):
        "Get memory of a process and its children."
        p = Process()
        memory = p.memory_info().rss
        for c in p.children():
            memory += c.memory_info().rss
        return memory

    def run(self):
        memory_start = self.get_memory()
        while not self.stop:
            self.memory_buffer.append(self.get_memory() - memory_start)
            time.sleep(0.01)

    def join(self):
        self.stop = True
        super().join()