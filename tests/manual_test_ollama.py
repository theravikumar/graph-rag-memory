import os
import time
from typing import Optional, Any
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import instructor

from graphmemo import MemoryClient

# ---------------------------------------------------------
# 1. Setup Ollama (Local LLM)
# ---------------------------------------------------------
# Ollama hosts an OpenAI-compatible endpoint at port 11434
# We use instructor to easily force JSON schema outputs if needed
raw_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama", # API key is required by the SDK but ignored by Ollama
)
client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)

MODEL_NAME = "gemma:2b"

# ---------------------------------------------------------
# 2. Boilerplate Setup (Embedder & LLM func)
# ---------------------------------------------------------
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def embed_func(text: str) -> list[float]:
    return embedder.encode(text).tolist()

def llm_func(sys_prompt: str, user_prompt: str, schema: Optional[BaseModel] = None) -> Any:
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    if schema:
        # Instructor extracts the strict JSON automatically
        return client.chat.completions.create(
            model=MODEL_NAME,
            response_model=schema,
            messages=messages
        )
    
    # Standard text generation
    response = raw_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )
    return response.choices[0].message.content

# ---------------------------------------------------------
# 3. Initialize Graph Memory
# ---------------------------------------------------------
memory = MemoryClient(llm_generate=llm_func, embed_text=embed_func, use_query_expansion=False)
USER_ID = "mass_test_user"

# ---------------------------------------------------------
# 4. Interactive CLI
# ---------------------------------------------------------
def main():
    print("="*60)
    print(f"🦙 Ollama-Powered Graph Memory Tester ({MODEL_NAME})")
    print("="*60)
    print("Running 100% locally. NO RATE LIMITS. NO API COSTS.")
    print("Type anything to chat. The agent will automatically build a knowledge graph.")
    print("Type 'exit' or 'quit' to close.\n")
    
    while True:
        try:
            user_input = input("\n[You]: ")
            if user_input.lower() in ["exit", "quit"]:
                print("  [System]: Shutting down. Flushing L1 buffer to Long-Term Graph Memory...")
                memory.buffer.flush(USER_ID)
                break
                
            if not user_input.strip():
                continue
                
            start_time = time.time()
            
            # 1. Add User message to memory
            memory.add_message(USER_ID, "user", user_input)
            
            # 2. Always Retrieve Graph Context (Now ultra-fast without query expansion)
            print("  [System]: Retrieving long-term Graph Context...")
            t_ret_start = time.time()
            context = memory.retrieve_context(USER_ID, user_input)
            ret_latency = (time.time() - t_ret_start) * 1000
            print(f"  [System]: Retrieval completed in {ret_latency:.0f}ms")
            
            sys_prompt = f"""You are a helpful AI assistant.
            
            Recent Conversation:
            {context['short_term_history']}
            
            Information you remember about the user and their life:
            {context['long_term_graph_context']}
            
            Answer the user's input naturally. DO NOT mention your "memory", "graph", "database", or the fact that you are retrieving context. Just weave the facts naturally into your conversation as if you have always known them.
            """
            
            # 3. Generate Ollama Answer
            t_gen_start = time.time()
            response_text = llm_func(sys_prompt, user_input)
            gen_latency = (time.time() - t_gen_start) * 1000
            total_latency = (time.time() - start_time) * 1000
            
            # 4. Save Bot answer to memory
            memory.add_message(USER_ID, "assistant", response_text)
            
            print(f"\n[Ollama] (Total: {total_latency:.0f}ms | Generation: {gen_latency:.0f}ms): {response_text}")
            
        except Exception as e:
            print(f"\n[Error]: {e}")

if __name__ == "__main__":
    main()
