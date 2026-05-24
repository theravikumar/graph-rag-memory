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
import faiss

# ============================================================================
# 1. Boilerplate & Initialization (LLM, Embedder, Telemetry)
# ============================================================================

# Ensure Groq API Key is available
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"

# Initialize Instructor-patched Groq for structured outputs
raw_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
groq_client = instructor.from_groq(raw_groq_client)

# Initialize lightweight local embedder for fast FAISS lookups
embedder = SentenceTransformer('all-MiniLM-L6-v2')
VECTOR_DIM = 384

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
# 2. Local Extension Tables (Tools & Guardrails)
# ============================================================================
# Instead of polluting the core graphmemo conversational memory, we build 
# dedicated enterprise FAISS indices for tools and rules here.

# FAISS Indices for Dual-Vector Tool Retrieval
tool_intent_index = faiss.IndexIDMap(faiss.IndexFlatIP(VECTOR_DIM))      # Vector 1: topic_label (Intent)
tool_desc_index = faiss.IndexIDMap(faiss.IndexFlatIP(VECTOR_DIM))        # Vector 2: topic_description

# FAISS Index for Guardrails/Rules
rule_index = faiss.IndexIDMap(faiss.IndexFlatIP(VECTOR_DIM))

# In-memory mock databases for our tables
mcp_tools_table: Dict[int, Dict[str, Any]] = {}
mcp_rules_table: Dict[int, Dict[str, Any]] = {}

def embed_func(text: str) -> np.ndarray:
    """Helper to embed text into a FAISS-compatible numpy array."""
    return np.array([embedder.encode(text).tolist()], dtype=np.float32)

def seed_enterprise_db():
    """
    Pre-loads our Tools and Guardrails into the standalone DBs.
    Notice the Dual-Vector setup for Fuel to solve the semantic overlap!
    """
    # ---------------------------------------------------------
    # Seed MCP Tools
    # ---------------------------------------------------------
    tools = [
        {
            "id": 1,
            "topic_label": "fuel_maintenance", # Intent (Vector 1)
            "topic_description": "Use this tool when you need information about repairing, fixing, or maintaining fuel tanks.", # Description (Vector 2)
            "api_request_body": {"action": "repair", "target": "{user_query_target}"}, # Dynamic payload template
            "api_response_body_keys": ["status", "mechanic_assigned", "eta"] # For validation checks
        },
        {
            "id": 2,
            "topic_label": "fuel_reading", # Intent (Vector 1)
            "topic_description": "Use this tool to get real-time fuel levels, readings, and gallons remaining.", # Description (Vector 2)
            "api_request_body": {"action": "get_level", "sensor": "{session_sensor_id}"},
            "api_response_body_keys": ["gallons_remaining", "percentage"]
        }
    ]
    
    for t in tools:
        mcp_tools_table[t["id"]] = t
        # We index BOTH the Intent and the Description for the Tie-Breaker logic!
        tool_intent_index.add_with_ids(embed_func(t["topic_label"]), np.array([t["id"]]))
        tool_desc_index.add_with_ids(embed_func(t["topic_description"]), np.array([t["id"]]))

    # ---------------------------------------------------------
    # Seed MCP Guardrails / Rules
    # ---------------------------------------------------------
    rules = [
        {
            "id": 1,
            "role_type": "normal_user",
            "rule_description": "Do not allow users to perform fuel maintenance actions. They can only read fuel levels."
        },
        {
            "id": 2,
            "role_type": "all",
            "rule_description": "Do not answer questions about unrelated topics like cooking or politics."
        }
    ]
    
    for r in rules:
        mcp_rules_table[r["id"]] = r
        rule_index.add_with_ids(embed_func(r["rule_description"]), np.array([r["id"]]))

seed_enterprise_db()

# ============================================================================
# 3. Parallel Execution Logic (The Core Pipeline)
# ============================================================================

async def run_input_guardrails(query: str, user_role: str) -> Tuple[bool, str]:
    """
    Runs an ultra-fast semantic check against the Guardrails table.
    If a rule matches heavily, we block the request. 0 LLM calls!
    """
    start_time = time.time()
    
    # 1. Embed the user query
    q_vec = embed_func(query)
    
    # 2. Search the Rule Index
    distances, rule_ids = rule_index.search(q_vec, 1)
    
    latency = (time.time() - start_time) * 1000
    Telemetry.log("Input Guardrail (FAISS)", latency, tokens=0)
    
    if len(rule_ids[0]) > 0 and rule_ids[0][0] != -1:
        score = distances[0][0]
        # If the semantic match is very high (> 0.7 IP)
        if score > 0.7:
            rule = mcp_rules_table[int(rule_ids[0][0])]
            # Check Role constraint
            if rule["role_type"] in ["all", user_role]:
                return True, f"Blocked by Input Guardrail: {rule['rule_description']}"
    
    return False, "Safe"

async def run_tool_selection(query: str) -> Optional[Dict[str, Any]]:
    """
    Dual-Vector retrieval. Solves the 'Fuel Reading vs Fuel Maintenance' overlap.
    """
    start_time = time.time()
    q_vec = embed_func(query)
    
    # We search both the description index AND the intent index
    desc_dist, desc_ids = tool_desc_index.search(q_vec, 2)
    int_dist, int_ids = tool_intent_index.search(q_vec, 2)
    
    latency = (time.time() - start_time) * 1000
    Telemetry.log("Tool Retrieval (Dual FAISS)", latency, tokens=0)
    
    # Simplistic tie-breaker: We sum the scores if an ID appears in both, 
    # prioritizing Intent matches over Description matches.
    scores = {}
    for i, t_id in enumerate(desc_ids[0]):
        if t_id != -1: scores[t_id] = scores.get(t_id, 0) + float(desc_dist[0][i])
    for i, t_id in enumerate(int_ids[0]):
        if t_id != -1: scores[t_id] = scores.get(t_id, 0) + float(int_dist[0][i]) * 1.5 # Intent weighting
        
    if not scores:
        return None
        
    # Get the highest scoring tool
    best_tool_id = max(scores.items(), key=lambda x: x[1])[0]
    best_score = scores[best_tool_id]
    
    # Threshold check: Only return a tool if we are confident (score > 1.0)
    if best_score > 1.0:
        return mcp_tools_table[best_tool_id]
    return None

async def execute_tool_payload(tool: Dict[str, Any], query: str) -> Dict[str, Any]:
    """
    Mocks the API execution. 
    Injects dynamic variables into the Request Body and validates the Response Body keys.
    """
    start_time = time.time()
    
    # 1. Format API Request Body (Dynamic Injection)
    # We replace the template strings with actual context
    request_payload = json.dumps(tool["api_request_body"])
    request_payload = request_payload.replace("{user_query_target}", query.split()[-1]) # Mock extraction
    request_payload = request_payload.replace("{session_sensor_id}", "SENSOR-99X")
    
    # 2. Execute Mock API Call
    await asyncio.sleep(0.3) # Simulate network latency
    
    # 3. Mock API Response
    if tool["topic_label"] == "fuel_reading":
        api_response = {"gallons_remaining": 450, "percentage": "85%"}
    else:
        api_response = {"status": "scheduled", "mechanic_assigned": "Bob", "eta": "2 hours"}
        
    # 4. Validate API Response against our schema
    missing_keys = [k for k in tool["api_response_body_keys"] if k not in api_response]
    if missing_keys:
        raise ValueError(f"Tool execution failed. Missing required keys: {missing_keys}")
        
    latency = (time.time() - start_time) * 1000
    Telemetry.log(f"API Execution ({tool['topic_label']})", latency, tokens=0)
    
    return api_response

async def run_output_guardrails_and_synthesis(query: str, tool_context: str) -> str:
    """
    Runs the LLM Synthesis to format the final answer to the user,
    WHILE running Output Guardrails in parallel to ensure the LLM isn't leaking secrets.
    """
    start_time = time.time()
    
    # 1. The Synthesis Prompt (combining tools and query)
    sys_prompt = "You are a helpful industrial agent. Synthesize the tool data into a friendly response."
    user_prompt = f"Tool Output Data: {tool_context}\nUser Query: {query}"
    
    # 2. Define the tasks we will run in parallel
    async def synthesize_llm():
        # A blocking LLM call wrapped in a thread so it doesn't block asyncio
        return await asyncio.to_thread(
            lambda: raw_groq_client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
            ).choices[0].message.content
        )
        
    async def check_output_guardrail():
        # Imagine this checks for sensitive data (like SSNs or passwords) in the tool output
        # before we even let the LLM see it, or checking the LLM output as it streams.
        # For this demo, we run a fast regex/semantic check on the raw tool_context.
        await asyncio.sleep(0.1) 
        if "CONFIDENTIAL" in tool_context:
            return True # Blocked
        return False # Safe
        
    # 3. Run BOTH in parallel
    synthesis_task = asyncio.create_task(synthesize_llm())
    guardrail_task = asyncio.create_task(check_output_guardrail())
    
    is_blocked = await guardrail_task
    if is_blocked:
        synthesis_task.cancel() # Stop the LLM mid-generation if guardrail fails!
        latency = (time.time() - start_time) * 1000
        Telemetry.log("Parallel Output Guardrail", latency, tokens=0)
        return "I'm sorry, I cannot output that information due to security policies."
        
    final_text = await synthesis_task
    
    latency = (time.time() - start_time) * 1000
    # Estimating tokens: 1 token ~= 4 chars
    estimated_tokens = (len(sys_prompt) + len(user_prompt) + len(final_text)) // 4
    Telemetry.log("Parallel Synthesis & Output Guardrail", latency, tokens=estimated_tokens)
    
    return final_text

# ============================================================================
# 4. The Orchestrator
# ============================================================================

async def process_user_query(query: str, user_role: str = "normal_user"):
    print(f"\n==================================================")
    print(f"[USER]: {query}")
    print(f"==================================================")
    
    # Phase 1: Parallel Input Guardrails & Tool Selection
    # By running these together, we eliminate sequential latency.
    guardrail_task = asyncio.create_task(run_input_guardrails(query, user_role))
    tool_task = asyncio.create_task(run_tool_selection(query))
    
    is_blocked, block_reason = await guardrail_task
    if is_blocked:
        print(f" [AGENT]: {block_reason}")
        return
        
    selected_tool = await tool_task
    
    # Phase 2: Tool Execution (or Fallback)
    tool_context = "No tool was used."
    if selected_tool:
        print(f"  [TOOL SELECTED] {selected_tool['topic_label']} (via Dual-Vector Tie-Breaker)")
        try:
            api_result = await execute_tool_payload(selected_tool, query)
            tool_context = json.dumps(api_result)
        except Exception as e:
            print(f" [TOOL ERROR]: {e}")
            tool_context = "Tool execution failed."
    else:
        print(f"  [TOOL SELECTED] None (Standard Chat Fallback)")

    # Phase 3: Parallel Output Guardrails & LLM Synthesis
    final_response = await run_output_guardrails_and_synthesis(query, tool_context)
    
    print(f" [AGENT]: {final_response}")
    print(f"\n Total Latency: {Telemetry.total_latency_ms:.1f}ms | Total Estimated Tokens: {Telemetry.total_tokens}")
    
    # Reset telemetry for next query
    Telemetry.total_latency_ms = 0
    Telemetry.total_tokens = 0

# ============================================================================
# Run Scenarios
# ============================================================================
if __name__ == "__main__":
    async def run():
        print("\n--- TEST 1: The 'Fuel Maintenance vs Reading' Overlap ---")
        # Notice how semantic description overlap won't fool it! It checks intent.
        await process_user_query("Can you check the current fuel reading on the tank?", user_role="normal_user")
        
        print("\n--- TEST 2: Input Guardrail Interception ---")
        # Normal users are NOT allowed to perform maintenance. This should block instantly (0 LLM calls).
        await process_user_query("Schedule fuel maintenance for the generator.", user_role="normal_user")
        
        print("\n--- TEST 3: App/System Guardrail ---")
        # Blocks unrelated talk instantly (0 LLM calls)
        await process_user_query("Who is winning the election?", user_role="all")
        
        print("\n--- TEST 4: Fallback to Normal Chat ---")
        await process_user_query("Hello, how are you today?", user_role="normal_user")
        
    asyncio.run(run())
