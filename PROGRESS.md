# Graph-RAG Memory Library: Progress Tracker

> **NOTE TO FUTURE AI AGENTS**: If you are reading this in a new conversation, this is the master tracker for the `graph-rag` memory library project. Read this file to understand the current state before writing any code. The full architectural blueprint is in the `implementation_plan.md` artifact from conversation `78b2206d`.

## Architecture Summary
A "Bring Your Own Key" (BYOK) Python library providing low-latency, agentic memory.
*   **L1 Cache**: Rolling window buffer (raw text).
*   **L2 Cache (Async)**: Background worker batches older messages, extracts quantitative JSON state, and builds a Hierarchical Topic Graph.
*   **Retrieval**: Dual Semantic Search (Label + Description Vectors) + Query Expansion (Instruction generation) for < 500ms latency.

---

## 🚀 Implementation Phases

### Phase 1: Foundation & Schemas
- [ ] Initialize Python project structure (`setup.py`, `requirements.txt`).
- [ ] Define core Pydantic models (`MemoryNode`, `QuantitativeState`, `Message`).
- [ ] Set up the database interface (Abstract Base Class).
- [ ] Implement the SQLite (local) + FAISS (vector) backend adapter.
- [ ] Implement the PostgreSQL (pgvector) backend adapter (optional/enterprise).

### Phase 2: The Short-Term Buffer
- [ ] Create the `BufferManager` class.
- [ ] Implement `add_message()` (append to rolling window).
- [ ] Implement `get_context()` (retrieve the raw L1 buffer).
- [ ] Implement the batch trigger logic (check if buffer > `M` messages).

### Phase 3: The Write Path (Background Batching)
- [ ] Create the `GraphConstructor` class.
- [ ] Implement LLM Prompt: **Quantitative JSON Extraction**.
- [ ] Implement **Retrieve-Then-Decide** logic (Find top 3 existing nodes).
- [ ] Implement LLM Prompt: **Graph Routing/Merging** (Action: Append, Child, Branch).
- [ ] Write logic to update the database and vectors safely.

### Phase 4: The Read Path (Retrieval Engine)
- [ ] Create the `SemanticRouter` class.
- [ ] Implement **Query Expansion** LLM prompt (generate query instructions).
- [ ] Implement **Dual Semantic Traversal** (Search against Label and Description vectors).
- [ ] Implement fast keyword fallback (BM25 or simple SQL ILIKE).
- [ ] Combine Buffer + Graph Node + JSON State into the final retrieved context.

### Phase 5: The Client API & Testing
- [ ] Create the main `MemoryClient` facade class.
- [ ] Write integration tests for the full Sync Read + Async Write pipeline.
- [ ] Write `README.md` with "Bring Your Own Key" instructions.

---
**Current Status**: Planning Complete. Ready to start Phase 1.
