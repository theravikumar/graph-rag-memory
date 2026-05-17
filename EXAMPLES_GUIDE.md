# Enterprise Integrations & MCP Examples Guide

This folder (`examples/`) contains concrete implementations demonstrating how to deploy the `graphmemo` library into production enterprise architectures.

Specifically, these examples showcase two massive concepts:
1. **Model Context Protocol (MCP)** integration with complex multi-tool architectures.
2. **Abstract Subclassing** to run the library on managed cloud databases (Azure, AWS, GCP).

---

## 1. The Core Architecture: "Parallel MCP & Temporal Override"

When building AI agents that use both **Tools** (like fetching live Jira tickets) and **Agentic Memory** (like remembering a user's preferences), developers face a massive dilemma: *How do you route the LLM and combine the data without causing latency spikes or hallucinations?*

All 4 scripts in this folder implement the ultimate solution to this problem:

1. **Instant L1 Routing:** When a user asks a question, the scripts instantly pull the last 5 messages from the `short_term_history` buffer. This 0ms lookup allows the LLM Router to understand if a user saying "Yes" means "Yes, create a ticket."
2. **Parallel `asyncio` Execution:** If a tool is selected, the scripts use `asyncio.gather()` to execute the API Tool AND fetch the deep Graph Memory **simultaneously**. This guarantees the perceived latency stays under 1 second.
3. **The Temporal Override Prompt:** To prevent the LLM from hallucinating by using outdated memory data, the final synthesis prompt strictly enforces a hierarchy: *Real-Time Tool Data is the absolute ground truth. Historical memory must only be used to fill in conversational gaps.*

---

## 2. Script Walkthroughs

### A. `mcp_local_sqlite.py`
**Intention:** This is the baseline example. It uses the default SQLite + FAISS implementation, making it perfect for free open-source users.
**How to run:**
```bash
# Set your Groq API Key
export GROQ_API_KEY="your_key"
# Run the script
python examples/mcp_local_sqlite.py
```
**What it does:** It runs a mock scenario where a user asks for a complex action. The guardrails intercept it, ask to raise a ticket, and when the user says "Yes," it successfully triggers the simulated `create_jira_ticket` tool using the Graph Memory context.

### B. `mcp_azure_cosmos.py`
**Intention:** Demonstrates how Microsoft Azure teams can plug the library into Cosmos DB or Azure SQL.
**How it works:** Look at the top of the file. You will see a custom class `AzureCosmosDB(MemoryDatabase)`. By subclassing `MemoryDatabase`, you can inject standard `azure-cosmos` Python SDK commands into the `add_message()` or `search_nodes()` functions. 
**How to run:**
```bash
pip install azure-cosmos
python examples/mcp_azure_cosmos.py
```

### C. `mcp_aws_dynamo.py`
**Intention:** Demonstrates how AWS teams can plug the library into DynamoDB (for relational storage) and Amazon OpenSearch (for vector embeddings).
**How it works:** It uses the exact same Parallel MCP logic as the Azure example, but demonstrates how to pass a custom `AWSDynamoDB()` class into `MemoryClient(db=aws_db)`.
**How to run:**
```bash
pip install boto3 opensearch-py
python examples/mcp_aws_dynamo.py
```

### D. `mcp_gcp_alloydb.py`
**Intention:** Demonstrates how Google Cloud teams can run the entire memory architecture on GCP AlloyDB (PostgreSQL) using the `pgvector` extension.
**How it works:** It provides the boilerplate for connecting `psycopg2` to GCP AlloyDB, replacing the default SQLite local storage with a massive managed cloud SQL database.
**How to run:**
```bash
pip install psycopg2-binary
python examples/mcp_gcp_alloydb.py
```
