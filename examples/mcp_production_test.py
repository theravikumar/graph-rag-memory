import os
import time
import asyncio
import json
import uuid
import numpy as np
from typing import Optional, Any, Dict, Tuple, List
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from groq import Groq
import instructor

from graphmemo import MemoryClient
from graphmemo.schemas import MemoryNode

# ============================================================================
# 1. Boilerplate & Initialization (LLM, Embedder, Telemetry)
# ============================================================================

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"

raw_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
groq_client = instructor.from_groq(raw_groq_client)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def llm_func(sys_prompt: str, user_prompt: str, schema: Optional[BaseModel] = None) -> Any:
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
    if schema:
        return groq_client.chat.completions.create(model="llama-3.1-8b-instant", response_model=schema, messages=messages)
    return raw_groq_client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages).choices[0].message.content

def graphmemo_embed_func(text: str) -> list[float]:
    return embedder.encode(text).tolist()

memory = MemoryClient(llm_generate=llm_func, embed_text=graphmemo_embed_func, use_query_expansion=False)

class Telemetry:
    """A simple class to track latency and estimated token usage across our parallel pipelines."""
    total_latency_ms: float = 0
    total_tokens: int = 0
    
    @classmethod
    def log(cls, step: str, latency: float, tokens: int = 0):
        cls.total_latency_ms += latency
        cls.total_tokens += tokens
        print(f"  [LATENCY] {step}: {latency:.1f}ms | Tokens: ~{tokens}")

# ============================================================================
# 2. Native Graphmemo Seeding (Using SYSTEM namespaces)
# ============================================================================

def seed_native_graphmemo():
    """
    Instead of separate databases, we seed Tools and Guardrails natively into 
    Graphmemo using strict namespaces (user_id = 'SYSTEM_TOOLS' / 'SYSTEM_RULES').
    """
    # Check if already seeded to prevent duplication
    existing_tools = memory.db.get_all_nodes("SYSTEM_TOOLS")
    if len(existing_tools) > 0:
        return

    print("[SYSTEM] Seeding native Graphmemo with Tools & Guardrails...")

    # 1. Seed Tools into 'SYSTEM_TOOLS' namespace
    tools = [
        {
            "intent": "fuel_maintenance",
            "desc": "Use this tool when you need information about repairing, fixing, or maintaining fuel tanks.",
            "state": {
                "api_request_body": {"action": "repair", "target": "{user_query_target}"},
                "api_response_body_keys": ["status", "mechanic_assigned", "eta"]
            }
        },
        {
            "intent": "fuel_reading",
            "desc": "Use this tool to get real-time fuel levels, readings, and gallons remaining.",
            "state": {
                "api_request_body": {"action": "get_level", "sensor": "{session_sensor_id}"},
                "api_response_body_keys": ["gallons_remaining", "percentage"]
            }
        }
    ]
    
    for t in tools:
        node = MemoryNode(
            user_id="SYSTEM_TOOLS",
            topic_label=t["intent"],              # Mapped to Intent Vector
            topic_description=t["desc"],          # Mapped to Description Vector
            node_type="tool",                     # Identifier
            summary="",                           # Unused for tools
            node_state=t["state"],                # Contains our robust execution schemas!
            label_embedding=graphmemo_embed_func(t["intent"]),
            description_embedding=graphmemo_embed_func(t["desc"])
        )
        memory.db.add_node(node)

    # 2. Seed Guardrails into 'SYSTEM_RULES' namespace
    rules = [
        {
            "desc": "Do not allow users to perform fuel maintenance actions. They can only read fuel levels.",
            "state": {"role_type": "normal_user"}
        },
        {
            "desc": "Do not answer questions about unrelated topics like cooking or politics.",
            "state": {"role_type": "all"}
        }
    ]
    
    for r in rules:
        node = MemoryNode(
            user_id="SYSTEM_RULES",
            topic_label="guardrail_rule",
            topic_description=r["desc"],
            node_type="guardrail",
            summary="",
            node_state=r["state"],
            label_embedding=graphmemo_embed_func("guardrail_rule"),
            description_embedding=graphmemo_embed_func(r["desc"])
        )
        memory.db.add_node(node)

seed_native_graphmemo()

# ============================================================================
# 3. Parallel Execution Logic (The Native Core Pipeline)
# ============================================================================

async def expand_query_async(query: str) -> str:
    """Uses the LLM to expand a short query into a highly descriptive semantic paragraph."""
    start_time = time.time()
    sys_prompt = "You are a query expansion engine. Take the user's short query and expand it into a detailed, descriptive paragraph clarifying intent, synonyms, and context. Output ONLY the expanded query, no conversational filler."
    
    expanded_text = await asyncio.to_thread(
        lambda: raw_groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": query}]
        ).choices[0].message.content
    )
    
    latency = (time.time() - start_time) * 1000
    estimated_tokens = (len(sys_prompt) + len(query) + len(expanded_text)) // 4
    Telemetry.log("Query Expansion (LLM)", latency, tokens=estimated_tokens)
    
    return expanded_text

async def run_input_guardrails(raw_query: str, expanded_query: str, user_role: str) -> Tuple[bool, str]:
    """
    Runs an ultra-fast semantic check natively against Graphmemo.
    """
    start_time = time.time()
    
    # 1. Embed the EXPANDED query for Guardrail Rule matching
    q_vec = graphmemo_embed_func(expanded_query)
    # We search the description vector for rule matching
    matched_rules = memory.db.search_nodes(user_id="SYSTEM_RULES", desc_vector=q_vec, top_k=1)
    
    latency = (time.time() - start_time) * 1000
    Telemetry.log("Input Guardrail (Native Graphmemo)", latency, tokens=0)
    
    if matched_rules:
        rule_node, score = matched_rules[0]
        # High semantic match threshold
        if score > 0.7:
            role_required = rule_node.node_state.get("role_type", "all")
            if role_required in ["all", user_role]:
                return True, f"Blocked by Input Guardrail: {rule_node.topic_description}"
    
    return False, "Safe"

async def run_tool_selection(raw_query: str, expanded_query: str) -> Optional[MemoryNode]:
    """
    Native Dual-Vector retrieval with Split-Query logic!
    Solves the 'Wife driving' semantic leap problem automatically.
    """
    start_time = time.time()
    
    # We embed BOTH queries
    raw_q_vec = graphmemo_embed_func(raw_query)
    expanded_q_vec = graphmemo_embed_func(expanded_query)
    
    # Split-Vector Search:
    # Description index loves the detailed expanded query
    desc_matches = memory.db.search_nodes(user_id="SYSTEM_TOOLS", desc_vector=expanded_q_vec, top_k=2)
    # Intent index loves the short, raw query
    intent_matches = memory.db.search_nodes(user_id="SYSTEM_TOOLS", label_vector=raw_q_vec, top_k=2)
    
    latency = (time.time() - start_time) * 1000
    Telemetry.log("Tool Retrieval (Native Dual-Vector)", latency, tokens=0)
    
    # Consolidate and tie-break scores
    scores = {}
    nodes_map = {}
    
    for node, score in desc_matches:
        scores[node.node_id] = scores.get(node.node_id, 0) + score
        nodes_map[node.node_id] = node
        
    for node, score in intent_matches:
        # Intent carries a 1.5x weight multiplier for tie-breaking
        scores[node.node_id] = scores.get(node.node_id, 0) + (score * 1.5)
        nodes_map[node.node_id] = node
        
    if not scores:
        return None
        
    # Get the highest scoring tool node
    best_node_id = max(scores.items(), key=lambda x: x[1])[0]
    best_score = scores[best_node_id]
    
    if best_score > 1.0:
        return nodes_map[best_node_id]
    return None

async def execute_tool_payload(tool_node: MemoryNode, query: str) -> Dict[str, Any]:
    """
    Extracts dynamic schemas directly from the native `node_state` JSON field!
    """
    start_time = time.time()
    state = tool_node.node_state
    
    # 1. Format API Request Body (Dynamic Injection)
    request_payload = json.dumps(state["api_request_body"])
    request_payload = request_payload.replace("{user_query_target}", query.split()[-1]) # Mock
    request_payload = request_payload.replace("{session_sensor_id}", "SENSOR-99X")
    
    # 2. Execute Mock API Call
    await asyncio.sleep(0.3) 
    
    # 3. Mock API Response
    if tool_node.topic_label == "fuel_reading":
        api_response = {"gallons_remaining": 450, "percentage": "85%"}
    else:
        api_response = {"status": "scheduled", "mechanic_assigned": "Bob", "eta": "2 hours"}
        
    # 4. Validate API Response against schema stored in Graphmemo!
    missing_keys = [k for k in state["api_response_body_keys"] if k not in api_response]
    if missing_keys:
        raise ValueError(f"Tool execution failed. Missing required keys: {missing_keys}")
        
    latency = (time.time() - start_time) * 1000
    Telemetry.log(f"API Execution ({tool_node.topic_label})", latency, tokens=0)
    
    return api_response

async def run_output_guardrails_and_synthesis(query: str, tool_context: str, graph_context: Dict[str, Any]) -> str:
    start_time = time.time()
    
    sys_prompt = f"You are a helpful industrial agent. Synthesize the tool data into a friendly response.\n\nLong-Term Memory Facts:\n{graph_context['long_term_graph_context']}\n\nRecent Chat History:\n{graph_context['short_term_history']}"
    user_prompt = f"Tool Output Data: {tool_context}\nUser Query: {query}"
    
    async def synthesize_llm():
        return await asyncio.to_thread(
            lambda: raw_groq_client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
            ).choices[0].message.content
        )
        
    async def check_output_guardrail():
        await asyncio.sleep(0.1) 
        if "CONFIDENTIAL" in tool_context:
            return True 
        return False 
        
    synthesis_task = asyncio.create_task(synthesize_llm())
    guardrail_task = asyncio.create_task(check_output_guardrail())
    
    is_blocked = await guardrail_task
    if is_blocked:
        synthesis_task.cancel() 
        latency = (time.time() - start_time) * 1000
        Telemetry.log("Parallel Output Guardrail", latency, tokens=0)
        return "I'm sorry, I cannot output that information due to security policies."
        
    final_text = await synthesis_task
    
    latency = (time.time() - start_time) * 1000
    estimated_tokens = (len(sys_prompt) + len(user_prompt) + len(final_text)) // 4
    Telemetry.log("Parallel Synthesis & Output Guardrail", latency, tokens=estimated_tokens)
    
    return final_text

# ============================================================================
# 4. The Orchestrator
# ============================================================================

async def process_user_query(query: str, user_role: str = "normal_user"):
    user_id = f"mock_{user_role}_id"
    
    print(f"\n==================================================")
    print(f"[USER]: {query}")
    print(f"==================================================")
    # Log the user message to conversational memory
    memory.add_message(user_id, "user", query)
    
    # Phase 0: Query Expansion (Sequential Bottleneck for accuracy)
    expanded_query = await expand_query_async(query)
    print(f"  [EXPANDED QUERY]: {expanded_query.strip()}")
    
    # Phase 1: Parallel Input Guardrails & Tool Selection natively!
    guardrail_task = asyncio.create_task(run_input_guardrails(query, expanded_query, user_role))
    tool_task = asyncio.create_task(run_tool_selection(query, expanded_query))
    
    is_blocked, block_reason = await guardrail_task
    if is_blocked:
        print(f" [AGENT]: {block_reason}")
        return
        
    selected_tool_node = await tool_task
    
    # Phase 2: Tool Execution
    tool_context = "No tool was used."
    if selected_tool_node:
        print(f"  [TOOL SELECTED] {selected_tool_node.topic_label} (via Native Dual-Vector)")
        try:
            api_result = await execute_tool_payload(selected_tool_node, query)
            tool_context = json.dumps(api_result)
        except Exception as e:
            print(f" [TOOL ERROR]: {e}")
            tool_context = "Tool execution failed."
    else:
        print(f"  [TOOL SELECTED] None (Standard Chat Fallback)")

    # Phase 3: Fetch Conversational Memory
    t_ret = time.time()
    graph_context = memory.retrieve_context(user_id, query)
    Telemetry.log("Graphmemo Context Retrieval", (time.time() - t_ret) * 1000, tokens=0)

    # Phase 4: Parallel Synthesis
    final_response = await run_output_guardrails_and_synthesis(query, tool_context, graph_context)
    
    memory.add_message(user_id, "assistant", final_response)
    
    print(f" [AGENT]: {final_response}")
    print(f"\n Total Latency: {Telemetry.total_latency_ms:.1f}ms | Total Estimated Tokens: {Telemetry.total_tokens}")
    
    Telemetry.total_latency_ms = 0
    Telemetry.total_tokens = 0

# ============================================================================
# Run Scenarios
# ============================================================================
if __name__ == "__main__":
    async def run():
        print("\n--- TEST 1: The 'Fuel Maintenance vs Reading' Overlap ---")
        await process_user_query("Can you check the current fuel reading on the tank?", user_role="normal_user")
        
        print("\n--- TEST 2: Input Guardrail Interception ---")
        await process_user_query("Schedule fuel maintenance for the generator.", user_role="normal_user")
        
        print("\n--- TEST 3: App/System Guardrail ---")
        await process_user_query("Who is winning the election?", user_role="all")
        
        print("\n--- TEST 4: Fallback to Normal Chat ---")
        await process_user_query("Hello, how are you today?", user_role="normal_user")
        
    asyncio.run(run())
