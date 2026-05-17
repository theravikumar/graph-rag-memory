import os
import time
import asyncio
from typing import Optional, Any
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq
import instructor

from graphmemo import MemoryClient

# ---------------------------------------------------------
# 1. Boilerplate Setup (LLM & Embedder)
# ---------------------------------------------------------
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "your_groq_api_key"

raw_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
groq_client = instructor.from_groq(raw_groq_client)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def embed_func(text: str) -> list[float]:
    return embedder.encode(text).tolist()

def llm_func(sys_prompt: str, user_prompt: str, schema: Optional[BaseModel] = None) -> Any:
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    if schema:
        return groq_client.chat.completions.create(model="llama-3.1-8b-instant", response_model=schema, messages=messages)
    response = raw_groq_client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages)
    return response.choices[0].message.content

memory = MemoryClient(llm_generate=llm_func, embed_text=embed_func)

# ---------------------------------------------------------
# 2. Mock MCP Tools
# ---------------------------------------------------------
class ToolRouter(BaseModel):
    selected_tool: str = "none" # 'get_jira_tickets', 'create_jira_ticket', 'get_weather', 'none'
    reasoning: str

async def execute_tool(tool_name: str, query: str, context: str) -> str:
    """Mock execution of MCP Tools."""
    await asyncio.sleep(0.2) # Simulate network latency
    if tool_name == "get_jira_tickets":
        return "Tool Output (JIRA): User has 3 open tickets. Latest ticket is #1042 'Fix login bug'."
    elif tool_name == "create_jira_ticket":
        return f"Tool Output (JIRA): Successfully created a new ticket based on context: '{context}'"
    elif tool_name == "get_weather":
        return "Tool Output (WEATHER): It is currently 72F and sunny."
    return "Tool Output: No tool was called."

# ---------------------------------------------------------
# 3. Parallel Execution Agent
# ---------------------------------------------------------
async def process_user_query(user_id: str, query: str):
    print(f"\n[USER]: {query}")
    start_time = time.time()
    
    # A. Fast Guardrails (0ms)
    if "ignore all previous instructions" in query.lower():
        print("[AGENT]: Blocked by guardrails.")
        return

    # B. L1 History Retrieval (Instant)
    # Get the last 5 messages so the LLM knows what we are talking about (e.g., if user says "yes")
    recent_msgs = memory.db.get_recent_messages(user_id, limit=5)
    history_str = "\n".join([f"{m.role}: {m.content}" for m in recent_msgs])
    
    # Log the user's query into memory immediately
    memory.add_message(user_id, "user", query)
    
    # C. Tool Routing
    router_sys = "You are a tool router. You have tools: 'get_jira_tickets', 'create_jira_ticket', 'get_weather'. Based on the query and recent history, select a tool. If the user is confirming a ticket creation, select 'create_jira_ticket'. Otherwise select 'none'."
    router_user = f"History:\n{history_str}\n\nCurrent Query: {query}"
    route = llm_func(router_sys, router_user, schema=ToolRouter)
    
    # Intercept missing tools logic (Guardrail for missing capability)
    if route.selected_tool == "none" and "complex" in query.lower():
        bot_reply = "I don't have a tool to perform that complex action. Should I raise an IT support ticket for you?"
        print(f"[AGENT]: {bot_reply}")
        memory.add_message(user_id, "assistant", bot_reply)
        return

    print(f"  -> Selected Tool: {route.selected_tool}")

    # D. PARALLEL EXECUTION (Crucial for <1s latency)
    async def fetch_memory():
        return memory.retrieve_context(user_id, query)
        
    async def fetch_tool():
        return await execute_tool(route.selected_tool, query, history_str)

    # Run both simultaneously!
    memory_result, tool_result = await asyncio.gather(fetch_memory(), fetch_tool())

    # E. Final Synthesis (Temporal Override Prompt)
    synthesis_sys = f"""
    You are an AI assistant. You have access to:
    1. Real-Time Tool Data: {tool_result}
    2. Historical Graph Memory: {memory_result['long_term_graph_context']}
    
    CRITICAL INSTRUCTION (TEMPORAL OVERRIDE):
    Real-Time Tool Data is the absolute ground truth. If the Historical Memory contradicts the Tool Data, 
    you MUST prioritize the Tool Data, as memory may be outdated. Use Memory Data only to fill in conversational gaps.
    """
    
    final_response = llm_func(synthesis_sys, query)
    latency = (time.time() - start_time) * 1000
    
    memory.add_message(user_id, "assistant", final_response)
    print(f"[AGENT] ({latency:.0f}ms): {final_response}")

# ---------------------------------------------------------
# Run Scenario
# ---------------------------------------------------------
if __name__ == "__main__":
    async def run_scenario():
        user = "office_worker_1"
        
        # 1. Ask a question we don't have a tool for
        await process_user_query(user, "Can you deploy a complex Kubernetes cluster for me?")
        
        # 2. Confirm the ticket creation
        await process_user_query(user, "Yes, please go ahead and raise it.")
        
        # 3. Check tickets (Testing Temporal Override)
        await process_user_query(user, "What are my current open tickets?")
        
    asyncio.run(run_scenario())
