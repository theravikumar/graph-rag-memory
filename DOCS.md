# Graph-Memory API Reference

Welcome to the `graph_memory` documentation. This library is designed to be highly modular and framework-agnostic.

---

## 1. `MemoryClient`
The primary interface for interacting with the Graph-Memory system. It wraps the underlying database, buffer manager, and semantic router.

### `__init__`
```python
def __init__(
    self,
    llm_generate: Callable[[str, str, Optional[BaseModel]], Any],
    embed_text: Callable[[str], List[float]],
    db: Optional[MemoryDatabase] = None,
    use_query_expansion: bool = True,
    buffer_size: int = 20,
    batch_size: int = 10
)
```
**Parameters:**
- `llm_generate` *(Callable)*: Your custom function to generate LLM responses. It must accept a system prompt, user prompt, and an optional Pydantic schema for structured JSON output.
- `embed_text` *(Callable)*: Your custom function to generate vector embeddings. Must return a flat `List[float]`.
- `db` *(MemoryDatabase, optional)*: A custom database instance. Defaults to `LocalMemoryManager` (SQLite + FAISS).
- `use_query_expansion` *(bool)*: If `True`, uses the LLM to expand queries during retrieval for higher accuracy. Defaults to `True`.
- `buffer_size` *(int)*: Maximum number of messages kept in the L1 instant-recall buffer before triggering a background graph construction. Defaults to `20`.
- `batch_size` *(int)*: Number of oldest messages to slice and send to the background graph constructor when the buffer overflows. Defaults to `10`.

### `add_message`
```python
def add_message(self, user_id: str, role: str, content: str) -> None
```
Adds a new chat message to the L1 short-term memory buffer. If the buffer hits `buffer_size`, it automatically spins up a non-blocking background thread to construct the L2 Graph.
- `user_id`: Unique identifier for the user.
- `role`: Usually `"user"`, `"assistant"`, or `"system"`.
- `content`: The text content of the message.

### `retrieve_context`
```python
def retrieve_context(self, user_id: str, query: str) -> Dict[str, Any]
```
Performs a Dual-Semantic Vector Search across the L2 Graph and returns all relevant context in a structured dictionary.
**Returns:**
```json
{
  "short_term_history": "USER: Hi...\nASSISTANT: Hello...",
  "global_state": {"user_age": 25, "favorite_color": "blue"},
  "long_term_graph_context": "--- Topic: Programming ---\n...",
  "expanded_intent": "The user is asking about..." 
}
```

### `generate_report`
```python
def generate_report(self, user_id: str, filepath: str = "METRICS_REPORT.md") -> None
```
Analyzes the database and built-in telemetry to generate a Markdown file showing latency, tokens used, and a visual text-tree of the Graph Memory structure.

---

## 2. Core Schemas

You can import the underlying data models directly if you are building custom database adapters.

```python
from graph_memory import Message, QuantitativeState, MemoryNode
```

### `Message`
Represents a single chat turn.
- `id`: UUID
- `user_id`: str
- `role`: str
- `content`: str
- `timestamp`: datetime

### `QuantitativeState`
Represents extracted JSON facts.
- `user_id`: str
- `state_data`: Dict[str, Any]

### `MemoryNode`
Represents a Topic in the L2 Hierarchical Graph.
- `node_id`: UUID
- `user_id`: str
- `parent_ids`: List[UUID]
- `related_node_ids`: List[UUID]
- `topic_label`: str *(Short title)*
- `topic_description`: str *(Detailed summary)*
- `node_type`: str *('root', 'branch', or 'leaf')*
- `summary`: str *(Factual bullet points)*
- `node_state`: Dict[str, Any] *(Node-specific JSON facts)*
