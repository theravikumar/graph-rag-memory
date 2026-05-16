import os
import time
import asyncio
from typing import Optional, Any, List, Dict
from uuid import UUID
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq
import instructor

from graph_memory import MemoryClient
from graph_memory.db.base import MemoryDatabase
from graph_memory.schemas import Message, QuantitativeState, MemoryNode

# ---------------------------------------------------------
# 1. Azure Enterprise Subclassing
# ---------------------------------------------------------
# In a real scenario, you would import:
# from azure.cosmos import CosmosClient

class AzureCosmosDB(MemoryDatabase):
    """
    Example implementation of an Azure Cosmos DB adapter for the graph-memory library.
    This seamlessly plugs into MemoryClient to back all graph logic with Azure.
    """
    def __init__(self, connection_string: str):
        # self.client = CosmosClient(connection_string)
        # self.db = self.client.get_database_client("AgenticMemory")
        # self.container = self.db.get_container_client("Nodes")
        print(f"Connected to Azure Cosmos DB at {connection_string}")
        
    def add_message(self, message: Message) -> None:
        # self.container.upsert_item(message.dict())
        pass

    def get_recent_messages(self, user_id: str, limit: int = 20) -> List[Message]:
        # query = f"SELECT * FROM c WHERE c.user_id = '{user_id}' ORDER BY c.timestamp DESC OFFSET 0 LIMIT {limit}"
        # return list(self.container.query_items(query=query))
        return []

    def delete_messages(self, message_ids: List[UUID]) -> None:
        pass

    def get_quantitative_state(self, user_id: str) -> QuantitativeState:
        return QuantitativeState(user_id=user_id)

    def update_quantitative_state(self, user_id: str, patches: Dict[str, Any]) -> None:
        pass

    def get_node(self, node_id: UUID) -> Optional[MemoryNode]:
        return None

    def add_node(self, node: MemoryNode) -> None:
        pass

    def update_node(self, node: MemoryNode) -> None:
        pass

    def add_relationship(self, from_node_id: UUID, to_node_id: UUID, rel_type: str = "parent") -> None:
        pass

    def get_all_nodes(self, user_id: str) -> List[MemoryNode]:
        return []

    def search_nodes(self, user_id: str, query_embedding: List[float], top_k: int = 5, distance_threshold: float = 1.0) -> List[MemoryNode]:
        # Azure Cosmos DB supports vector search via Vector Indexing Policies
        # return self.container.query_items(query="SELECT TOP @k * FROM c ORDER BY VectorDistance(c.embedding, @query_embedding)", ...)
        return []

# ---------------------------------------------------------
# 2. Boilerplate Setup (LLM & Embedder)
# ---------------------------------------------------------
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "your_groq_api_key"

raw_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
groq_client = instructor.from_groq(raw_groq_client)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def embed_func(text: str) -> list[float]:
    return embedder.encode(text).tolist()

def llm_func(sys_prompt: str, user_prompt: str, schema: Optional[BaseModel] = None) -> Any:
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
    if schema:
        return groq_client.chat.completions.create(model="llama-3.1-8b-instant", response_model=schema, messages=messages)
    return raw_groq_client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages).choices[0].message.content

# INITIALIZE WITH AZURE
azure_db = AzureCosmosDB("AccountEndpoint=https://myaccount.documents.azure.com:443/;AccountKey=...")
memory = MemoryClient(llm_generate=llm_func, embed_text=embed_func, db=azure_db)

# ---------------------------------------------------------
# 3. Mock MCP Tools & Parallel Architecture
# ---------------------------------------------------------
class ToolRouter(BaseModel):
    selected_tool: str = "none" # 'get_jira_tickets', 'create_jira_ticket', 'get_weather', 'none'
    reasoning: str

async def execute_tool(tool_name: str, query: str, context: str) -> str:
    await asyncio.sleep(0.2)
    if tool_name == "create_jira_ticket":
        return f"Tool Output (JIRA): Successfully created a new ticket based on context: '{context}'"
    return "Tool Output: No tool was called."

async def process_user_query(user_id: str, query: str):
    print(f"\n[USER]: {query}")
    start_time = time.time()
    
    recent_msgs = memory.db.get_recent_messages(user_id, limit=5)
    history_str = "\n".join([f"{m.role}: {m.content}" for m in recent_msgs])
    memory.add_message(user_id, "user", query)
    
    router_sys = "You are a tool router. Select 'create_jira_ticket' if the user confirms a ticket, else 'none'."
    router_user = f"History:\n{history_str}\n\nCurrent Query: {query}"
    route = llm_func(router_sys, router_user, schema=ToolRouter)
    
    if route.selected_tool == "none" and "complex" in query.lower():
        bot_reply = "I don't have a tool to perform that complex action. Should I raise an IT support ticket for you?"
        print(f"[AGENT]: {bot_reply}")
        memory.add_message(user_id, "assistant", bot_reply)
        return

    print(f"  -> Selected Tool: {route.selected_tool}")

    # PARALLEL EXECUTION
    async def fetch_memory(): return memory.retrieve_context(user_id, query)
    async def fetch_tool(): return await execute_tool(route.selected_tool, query, history_str)

    memory_result, tool_result = await asyncio.gather(fetch_memory(), fetch_tool())

    # SYNTHESIS (TEMPORAL OVERRIDE)
    synthesis_sys = f"""
    1. Real-Time Tool Data: {tool_result}
    2. Historical Graph Memory: {memory_result['long_term_graph_context']}
    CRITICAL INSTRUCTION (TEMPORAL OVERRIDE): Real-Time Tool Data is the absolute ground truth. Prioritize it over memory if there is a conflict.
    """
    
    final_response = llm_func(synthesis_sys, query)
    latency = (time.time() - start_time) * 1000
    
    memory.add_message(user_id, "assistant", final_response)
    print(f"[AGENT] ({latency:.0f}ms): {final_response}")

if __name__ == "__main__":
    async def run_scenario():
        user = "azure_worker_1"
        await process_user_query(user, "Can you deploy a complex Kubernetes cluster for me?")
        await process_user_query(user, "Yes, please go ahead and raise it.")
        
    asyncio.run(run_scenario())
