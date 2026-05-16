import time
from typing import Callable, Any, Dict, List, Optional
from pydantic import BaseModel

from .db.base import MemoryDatabase
from .db.local_manager import LocalMemoryManager
from .core.buffer import BufferManager
from .core.constructor import GraphConstructor
from .core.router import SemanticRouter
from .schemas import Message
from .utils.telemetry import TelemetryManager

class MemoryClient:
    """
    The main Facade for the Graph-RAG Memory Library.
    Provides a simple API for adding messages and retrieving context,
    while orchestrating the complex L1/L2 caching and background graph construction.
    """
    def __init__(
        self,
        llm_generate: Callable[[str, str, Optional[BaseModel]], Any],
        embed_text: Callable[[str], List[float]],
        db: Optional[MemoryDatabase] = None,
        use_query_expansion: bool = True,
        buffer_size: int = 20,
        batch_size: int = 10
    ):
        self.db = db or LocalMemoryManager()
        self.telemetry = TelemetryManager()
        
        # Telemetry Wrappers
        def tracked_llm(sys_prompt, user_prompt, schema=None):
            res = llm_generate(sys_prompt, user_prompt, schema)
            out_str = str(res.model_dump_json()) if hasattr(res, "model_dump_json") else str(res)
            self.telemetry.record_llm_call(sys_prompt + user_prompt, out_str)
            return res
            
        def tracked_batch_callback(uid, batch):
            t0 = time.time()
            self.constructor.process_batch(uid, batch)
            self.telemetry.record_write_latency((time.time() - t0) * 1000)
        
        self.constructor = GraphConstructor(self.db, tracked_llm, embed_text)
        self.router = SemanticRouter(self.db, tracked_llm, embed_text, use_query_expansion)
        
        self.buffer = BufferManager(
            db=self.db, 
            buffer_size=buffer_size, 
            batch_size=batch_size,
            batch_callback=tracked_batch_callback
        )

    def add_message(self, user_id: str, role: str, content: str) -> None:
        self.buffer.add_message(user_id, role, content)

    def retrieve_context(self, user_id: str, query: str) -> Dict[str, Any]:
        t0 = time.time()
        res = self.router.retrieve_context(user_id, query)
        self.telemetry.record_read_latency((time.time() - t0) * 1000)
        return res

    def generate_report(self, user_id: str, filepath: str = "METRICS_REPORT.md") -> None:
        """Generates a detailed markdown report of the system's performance and memory state."""
        stats = self.telemetry.get_report()
        recent = len(self.db.get_recent_messages(user_id, limit=100))
        nodes = self.db.get_all_nodes(user_id)
        state = self.db.get_quantitative_state(user_id)
        
        report = f"# Graph-Memory: Explorer & Telemetry Report\n\n"
        
        report += f"## ⚡ Performance Metrics\n"
        report += f"- **Avg Sync Retrieval Latency**: {stats['avg_read_latency_ms']} ms\n"
        report += f"- **Avg Async Graph Construction**: {stats['avg_write_latency_ms']} ms\n"
        report += f"- **Total LLM Calls**: {stats['total_llm_calls']}\n"
        report += f"- **Estimated Tokens Used**: {stats['estimated_tokens_used']} tokens\n\n"
        
        report += f"## 🗄️ Storage Metrics\n"
        report += f"- **L1 Buffer (Recent Messages)**: {recent}\n"
        report += f"- **L2 Graph Topics**: {len(nodes)}\n"
        report += f"- **Quantitative State Keys**: {len(state.state_data.keys())}\n\n"
        
        report += f"## 🌳 Memory Graph Visualization\n"
        report += f"```text\n"
        if not nodes:
            report += "(Graph is empty)\n"
        else:
            for n in nodes:
                report += f"[{n.node_type.upper()}] {n.topic_label} (ID: {str(n.node_id)[:8]})\n"
                report += f"    Desc: {n.topic_description}\n"
                if n.parent_ids:
                    report += f"    Parents: {[str(pid)[:8] for pid in n.parent_ids]}\n"
        report += f"```\n\n"
        
        report += f"## 📊 Quantitative JSON State\n"
        report += f"```json\n{state.state_data}\n```\n"
        
        with open(filepath, "w") as f:
            f.write(report)
        print(f"Report generated at {filepath}")
