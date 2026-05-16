import time
from typing import Dict, List, Any
import threading

class TelemetryManager:
    """
    Tracks performance metrics and LLM token estimates across the memory system.
    Thread-safe to support the async background workers.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.metrics = {
            "sync_read_latencies_ms": [],
            "async_write_latencies_ms": [],
            "total_llm_calls": 0,
            "estimated_tokens_used": 0,
        }

    def record_read_latency(self, latency_ms: float):
        with self.lock:
            self.metrics["sync_read_latencies_ms"].append(latency_ms)

    def record_write_latency(self, latency_ms: float):
        with self.lock:
            self.metrics["async_write_latencies_ms"].append(latency_ms)

    def record_llm_call(self, input_text: str, output_text: str):
        """Rough heuristic: 1 token = ~4 characters"""
        estimated_input = len(input_text) // 4
        estimated_output = len(output_text) // 4
        with self.lock:
            self.metrics["total_llm_calls"] += 1
            self.metrics["estimated_tokens_used"] += (estimated_input + estimated_output)

    def get_report(self) -> Dict[str, Any]:
        with self.lock:
            reads = self.metrics["sync_read_latencies_ms"]
            writes = self.metrics["async_write_latencies_ms"]
            return {
                "avg_read_latency_ms": round(sum(reads)/len(reads), 2) if reads else 0,
                "avg_write_latency_ms": round(sum(writes)/len(writes), 2) if writes else 0,
                "total_reads": len(reads),
                "total_writes": len(writes),
                "total_llm_calls": self.metrics["total_llm_calls"],
                "estimated_tokens_used": self.metrics["estimated_tokens_used"]
            }
