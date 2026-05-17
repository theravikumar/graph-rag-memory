<div align="center">
  <h1>Graphmemo</h1>
  <p><strong>A blazingly fast, agentic BYOK memory library for chatbots.</strong></p>

  <p>
    <a href="https://pypi.org/project/graphmemo/"><img src="https://img.shields.io/pypi/v/graphmemo.svg" alt="PyPI Version"></a>
    <a href="https://pypi.org/project/graphmemo/"><img src="https://img.shields.io/pypi/pyversions/graphmemo.svg" alt="Python Versions"></a>
    <a href="https://github.com/yourusername/graphmemo/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
    <a href="https://pepy.tech/project/graphmemo"><img src="https://static.pepy.tech/badge/graphmemo" alt="Downloads"></a>
  </p>
</div>

---

**Graphmemo** provides a sub-400ms dual-layered memory architecture (L1 Short-Term Buffer + L2 Hierarchical Graph) designed for LLM agents. It completely eliminates the latency of traditional RAG by executing graph construction *asynchronously* while providing instant routing.

## 🚀 Key Features

* **< 400ms Retrieval Latency**: Never wait for an LLM to read chat history. The query context is returned instantly.
* **Async Graph Construction**: Builds a conceptual memory graph in the background without blocking the user conversation.
* **Dual Semantic Search**: Prevents "Catastrophic Misrouting" by searching against both exact Topic Labels and Expanded Topic Descriptions.
* **Bring Your Own Key (BYOK)**: Use your own free local embeddings (e.g., HuggingFace) and your own LLM provider (Groq, OpenAI, Gemini) to avoid vendor lock-in and high SaaS fees.
* **100% Local DB**: Stores state cleanly in a local FAISS index + SQLite/SQLAlchemy database, meaning your data never leaves your environment.

## 📦 Installation

Graphmemo is lightweight and highly modular.

**For a lightning-fast install** (if using external API embeddings like OpenAI, Google, Anthropic):
```bash
pip install graphmemo
```

**To include local embeddings** (installs `sentence-transformers` and `PyTorch`, which may take a few minutes):
```bash
pip install "graphmemo[local]"
```

## ⚡ Quick Start

Because this library requires no paid SaaS subscription, you must inject your own LLM and Embedding functions.

```python
import os
from pydantic import BaseModel
from typing import Optional, Any
from groq import Groq
from sentence_transformers import SentenceTransformer
from graphmemo import MemoryClient

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

## 📖 Architecture Overview

Graphmemo relies on a dual-engine system:
1. **The Fast Router**: When a user queries your agent, Graphmemo bypasses the LLM entirely. It embeds the query and runs a highly optimized multi-dimensional FAISS search over your historical topic graph. Latency: `~300-400ms`.
2. **The Background Constructor**: As the user speaks, Graphmemo silently pushes conversations into an Async Queue. An LLM agent processes batches in the background, updating topics and maintaining global state quantitative values without blocking the chat UI.

## 🤝 Contributing

Contributions are welcome! If you find a bug, or have a feature request, please open an issue.

## 📄 License

Graphmemo is MIT Licensed.
