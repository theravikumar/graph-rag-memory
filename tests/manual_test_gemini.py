import os
import time
from typing import Optional, Any
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

from graphmemo import MemoryClient

# ---------------------------------------------------------
# 1. Setup Gemini SDK
# ---------------------------------------------------------
if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = ""

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ---------------------------------------------------------
# 2. Boilerplate Setup (LLM & Embedder)
# ---------------------------------------------------------
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def embed_func(text: str) -> list[float]:
    return embedder.encode(text).tolist()

def llm_func(sys_prompt: str, user_prompt: str, schema: Optional[BaseModel] = None) -> Any:
    # Google GenAI uses System Instructions differently than Groq
    config = types.GenerateContentConfig(
        system_instruction=sys_prompt,
        response_mime_type="application/json" if schema else "text/plain",
    )
    
    # If a pydantic schema is passed, we can instruct the LLM to output JSON matching the schema
    # (Note: Google's structured output works best when the schema is described in the system prompt)
    if schema:
        import json
        config.system_instruction += f"\nOUTPUT STRICT JSON MATCHING THIS SCHEMA:\n{json.dumps(schema.model_json_schema())}"
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=user_prompt,
        config=config,
    )
    
    if schema:
        return schema.model_validate_json(response.text)
    return response.text

# ---------------------------------------------------------
# 3. Initialize Graph Memory
# ---------------------------------------------------------
memory = MemoryClient(llm_generate=llm_func, embed_text=embed_func)
USER_ID = "gemini_tester_1"

# ---------------------------------------------------------
# 4. Interactive CLI
# ---------------------------------------------------------
def main():
    print("="*60)
    print("🧠 Gemini-Powered Graph Memory Tester")
    print("="*60)
    print("Type anything to chat. The agent will automatically build a knowledge graph.")
    print("Type 'exit' or 'quit' to close.\n")
    
    while True:
        try:
            user_input = input("\n[You]: ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            if not user_input.strip():
                continue
                
            start_time = time.time()
            
            # 1. Add User message to memory
            memory.add_message(USER_ID, "user", user_input)
            
            # 2. Check if we need deep context (simple heuristic: if it's a question)
            is_question = "?" in user_input
            
            if is_question:
                print("  [System]: Retrieving long-term Graph Context...")
                t_ret_start = time.time()
                context = memory.retrieve_context(USER_ID, user_input)
                ret_latency = (time.time() - t_ret_start) * 1000
                print(f"  [System]: Retrieval completed in {ret_latency:.0f}ms")
                
                sys_prompt = f"""You are a helpful AI assistant with perfect long-term memory.
                
                L1 Short-Term History:
                {context['short_term_history']}
                
                L2 Long-Term Graph Memory:
                {context['long_term_graph_context']}
                
                Answer the user's question accurately using your memory.
                """
            else:
                # Just use short-term history for normal conversation to save tokens
                recent_msgs = memory.db.get_recent_messages(USER_ID, limit=5)
                history_str = "\n".join([f"{m.role}: {m.content}" for m in recent_msgs])
                
                sys_prompt = f"You are a helpful AI assistant. Here is the recent conversation context:\n{history_str}"
            
            # 3. Generate Gemini Answer
            t_gen_start = time.time()
            response_text = llm_func(sys_prompt, user_input)
            gen_latency = (time.time() - t_gen_start) * 1000
            total_latency = (time.time() - start_time) * 1000
            
            # 4. Save Bot answer to memory
            memory.add_message(USER_ID, "assistant", response_text)
            
            print(f"\n[Gemini] (Total: {total_latency:.0f}ms | Generation: {gen_latency:.0f}ms): {response_text}")
            
            print("\n  [System]: Waiting 12 seconds to respect Gemini Free Tier 5 Requests/Min rate limit...")
            time.sleep(12)
            
        except Exception as e:
            print(f"\n[Error]: {e}")

if __name__ == "__main__":
    main()
