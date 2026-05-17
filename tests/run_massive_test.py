import os
import time
import json
from typing import Optional, Any
from pydantic import BaseModel
import instructor
from groq import Groq
from sentence_transformers import SentenceTransformer

from graphmemo import MemoryClient

print("--- MASSIVE 5000 QUERY SIMULATOR ---")

# Ensure API Key is set
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"

# 1. Init BYOK Architecture
print("Loading local embedder...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
def embed_func(text: str) -> list[float]:
    return embedder.encode(text).tolist()

print("Initializing Groq LLM...")
raw_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
groq_client = instructor.from_groq(raw_groq_client)

def llm_func(sys_prompt: str, user_prompt: str, schema: Optional[BaseModel] = None) -> Any:
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    if schema:
        return groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            response_model=schema,
            messages=messages,
            max_tokens=1000
        )
    else:
        response = raw_groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=1000
        )
        return response.choices[0].message.content

print("Starting MemoryClient...")
# We use standard buffer_size (20) and batch_size (10) for production scaling
memory = MemoryClient(
    llm_generate=llm_func,
    embed_text=embed_func,
    use_query_expansion=True,
    buffer_size=20,
    batch_size=10
)

# 2. Load Dataset
if not os.path.exists("dataset_organic_2000.json"):
    print("Error: dataset_organic_2000.json not found.")
    exit(1)

with open("dataset_organic_2000.json", "r") as f:
    dataset = json.load(f)

USER_ID = "mass_test_user"

# Initialize LOG file
LOG_FILE = "MASSIVE_TEST_LOG.md"
with open(LOG_FILE, "w") as f:
    f.write("# Organic Conversation Test Log\n\n")

JSONL_FILE = "TEST_RESULTS.jsonl"
# Clear previous JSONL file
if os.path.exists(JSONL_FILE):
    os.remove(JSONL_FILE)

print(f"Loaded {len(dataset)} interactions. Beginning Simulation...")
print("NOTE: This script enforces strict sleep timers to respect the 2000 TPM Groq Rate Limit.")
print("Logs are written continuously to MASSIVE_TEST_LOG.md\n")

statements_added = 0
queries_made = 0

for interaction in dataset:
    i_id = interaction['id']
    i_type = interaction['type']
    content = interaction['content']
    
    if i_type == "statement":
        # Add to L1 buffer silently
        memory.add_message(USER_ID, "user", content)
        statements_added += 1
        print(f"[{i_id}/5000] Statement added to L1 Buffer.")
        
        # RATE LIMIT PACING
        # Every time we hit the batch size (10), the L2 constructor runs in the background.
        # It uses ~1000 tokens per batch. Groq limits us to 6000 TPM.
        # We must pause for 15 seconds every 10 messages to avoid 429 Errors.
        if statements_added % 10 == 0:
            print(">> Batch triggered! Sleeping 20 seconds for background LLM and rate limits...")
            time.sleep(20)
            
    elif i_type == "query":
        # Retrieve context
        print(f"[{i_id}/{len(dataset)}] Querying Semantic Router...")
        t0 = time.time()
        try:
            context = memory.retrieve_context(USER_ID, content)
        except Exception as e:
            if "429" in str(e):
                print(">> Rate limit hit during retrieval. Sleeping 60s...")
                time.sleep(60)
                context = memory.retrieve_context(USER_ID, content)
            else:
                raise e
                
        retrieval_latency = (time.time() - t0) * 1000
        
        # Generate Final Answer
        sys_prompt = f"You are a helpful assistant. Answer the user's query strictly using the provided context. If the answer is not in the context, say 'I don't know'.\n\nShort Term Context:\n{context['short_term_history']}\n\nLong Term Context:\n{context['long_term_graph_context']}\n\nGlobal Facts:\n{context['global_state']}"
        t1 = time.time()
        try:
            final_answer = llm_func(sys_prompt, content)
        except Exception as e:
            if "429" in str(e):
                print(">> Rate limit hit during answer generation. Sleeping 60s...")
                time.sleep(60)
                final_answer = llm_func(sys_prompt, content)
            else:
                final_answer = "Error generating answer."
        
        generation_latency = (time.time() - t1) * 1000
        total_latency = (time.time() - t0) * 1000
        
        queries_made += 1
        
        # Log to markdown file
        log_entry = f"### Query #{i_id}\n"
        log_entry += f"**Question:** {content}\n"
        log_entry += f"**Retrieval Latency:** {retrieval_latency:.2f} ms\n"
        log_entry += f"**Generation Latency:** {generation_latency:.2f} ms\n"
        log_entry += f"**Total Latency:** {total_latency:.2f} ms\n"
        log_entry += f"**Ground Truth:** {interaction.get('ground_truth', 'N/A')}\n"
        log_entry += f"**Temporal Test:** {interaction.get('is_temporal_test', False)}\n"
        log_entry += f"**Expanded Intent:** {context.get('expanded_intent', 'N/A')}\n"
        log_entry += f"**Retrieved Context:**\n```text\n{context['long_term_graph_context']}\n```\n"
        log_entry += f"**Final Answer:** {final_answer}\n"
        log_entry += f"---\n\n"
        
        with open(LOG_FILE, "a") as f:
            f.write(log_entry)
            
        # Log to JSONL for AI Evaluator
        jsonl_entry = {
            "id": i_id,
            "query": content,
            "ground_truth": interaction.get("ground_truth", ""),
            "is_temporal_test": interaction.get("is_temporal_test", False),
            "injected_turn_id": interaction.get("injected_turn_id", "N/A"),
            "retrieved_context": context['long_term_graph_context'],
            "final_answer": final_answer,
            "retrieval_latency_ms": round(retrieval_latency, 2),
            "total_latency_ms": round(total_latency, 2)
        }
        with open(JSONL_FILE, "a") as f:
            f.write(json.dumps(jsonl_entry) + "\n")
            
        # Pacing for queries to respect limits (increased sleep due to extra LLM call)
        time.sleep(3)

print("\n=============================================")
print("SIMULATION COMPLETE!")
print(f"Total Statements: {statements_added}")
print(f"Total Queries: {queries_made}")
print(f"Full query logs saved to: {LOG_FILE}")

# Generate Final Telemetry
memory.generate_report(USER_ID, filepath="FINAL_5000_METRICS.md")
print("Final performance metrics saved to FINAL_5000_METRICS.md")
