import threading
from typing import Optional, Tuple


class AudioRingBuffer:
    def __init__(self, size_bytes: int = 300_000) -> None:
        self._size = size_bytes
        self._buffer = bytearray(size_bytes)
        self._write_pos = 0
        self._total_written = 0
        self._lock = threading.Lock()
        self._data_available = threading.Condition(self._lock)

    def write(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            chunk_size = len(data)
            end_pos = self._write_pos + chunk_size
            if end_pos > self._size:
                bytes_until_end = self._size - self._write_pos
                self._buffer[self._write_pos : self._size] = data[:bytes_until_end]
                bytes_remaining = chunk_size - bytes_until_end
                self._buffer[0:bytes_remaining] = data[bytes_until_end:]
                self._write_pos = bytes_remaining
            else:
                self._buffer[self._write_pos:end_pos] = data
                self._write_pos = end_pos % self._size
            self._total_written += chunk_size
            self._data_available.notify_all()

    def wait_for_data(self, timeout: Optional[float] = None) -> bool:
        with self._lock:
            if self._total_written > 0:
                return True
            return self._data_available.wait(timeout=timeout)

    def read_available_since(self, last_total_bytes: int, wait_timeout: Optional[float] = None) -> Tuple[bytes, int]:
        with self._lock:
            if self._total_written == last_total_bytes:
                if wait_timeout is not None:
                    self._data_available.wait(timeout=wait_timeout)
                if self._total_written == last_total_bytes:
                    return b"", last_total_bytes
            available = self._total_written - last_total_bytes
            if available <= 0:
                return b"", last_total_bytes
            start_index = last_total_bytes % self._size
            end_index = (last_total_bytes + available) % self._size
            if end_index > start_index:
                data = bytes(self._buffer[start_index:end_index])
            else:
                data = bytes(self._buffer[start_index:] + self._buffer[:end_index])
            return data, last_total_bytes + available

    def snapshot_total_written(self) -> int:
        with self._lock:
            return self._total_written
