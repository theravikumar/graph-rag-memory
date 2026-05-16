# Graph-Memory

A blazingly fast, agentic memory library for Chatbots. 
Graph-Memory combines an L1 Short-Term Buffer with a background L2 Hierarchical Topic Graph and Quantitative JSON State.

## Features
- **< 200ms Retrieval Latency**: Never wait for an LLM to read chat history.
- **Async Graph Construction**: Builds a conceptual memory graph in the background without blocking the user.
- **Dual Semantic Search**: Prevents "Catastrophic Misrouting" by searching against both Topic Labels and Expanded Topic Descriptions.
- **Bring Your Own Key (BYOK)**: Use your own free local embeddings (e.g., HuggingFace) and your own LLM provider (Groq, OpenAI, Gemini) to avoid vendor lock-in and high SaaS fees.

## Installation
```bash
pip install graph-rag-memory
```

## Quick Start (BYOK)

Because this library requires no paid SaaS subscription, you must inject your own LLM and Embedding functions.

```python
import os
from pydantic import BaseModel
from typing import Optional, Any
from groq import Groq
from sentence_transformers import SentenceTransformer
from graph_memory import MemoryClient

# 1. Define your free local embedder
embedder = SentenceTransformer('all-MiniLM-L6-v2')
def embed_func(text: str) -> list[float]:
    return embedder.encode(text).tolist()

# 2. Define your LLM generator (e.g., using Groq for blazing speed)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def llm_func(sys_prompt: str, user_prompt: str, schema: Optional[BaseModel] = None) -> Any:
    # If a Pydantic schema is provided, force JSON output
    # (Implementation omitted for brevity, use instructor or standard JSON mode)
    pass

# 3. Initialize the Memory Client
memory = MemoryClient(
    llm_generate=llm_func,
    embed_text=embed_func,
    use_query_expansion=True # Turn off for absolute maximum speed
)

# 4. Use it in your chatbot loop
user_id = "user_123"

# Add messages (silently batches to the Graph in the background)
memory.add_message(user_id, "user", "I bought a new Tesla Model 3 today!")

# Retrieve instant, highly accurate context for your chatbot's prompt
context = memory.retrieve_context(user_id, "What car do I drive?")

print(context['short_term_history'])
print(context['long_term_graph_context'])
print(context['global_state'])
```
