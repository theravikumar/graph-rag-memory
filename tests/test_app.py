import os
import time
from typing import Optional, Any
from pydantic import BaseModel
import instructor
from groq import Groq
from sentence_transformers import SentenceTransformer

from graph_memory import MemoryClient

# Clean up previous test DB to ensure a fresh graph
if os.path.exists("memory.db"):
    os.remove("memory.db")

# 1. SET API KEY
# Using the key provided by the user for testing
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"

# =====================================================================
# EXAMPLE: HOW A USER WOULD IMPLEMENT THIS WITH GEMINI
# =====================================================================
"""
import google.generativeai as genai
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def gemini_llm_func(sys_prompt: str, user_prompt: str, schema: Optional[BaseModel] = None) -> Any:
    # Gemini 1.5 Pro/Flash supports structured outputs natively via response_schema
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=sys_prompt
    )
    
    if schema:
        # Pass the Pydantic schema class directly to Gemini
        response = model.generate_content(
            user_prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=schema
            )
        )
        # Parse the JSON string back into the Pydantic object
        import json
        return schema(**json.loads(response.text))
    else:
        response = model.generate_content(user_prompt)
        return response.text
"""
# =====================================================================

print("--- 1. Loading Local BYOK Embedder (HuggingFace) ---")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
def embed_func(text: str) -> list[float]:
    return embedder.encode(text).tolist()

print("--- 2. Initializing BYOK LLM (Groq + Instructor) ---")
# We use instructor to easily force Groq to return Pydantic objects.
# The user doesn't NEED instructor, they just need to return the schema instance.
raw_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
groq_client = instructor.from_groq(raw_groq_client)

def llm_func(sys_prompt: str, user_prompt: str, schema: Optional[BaseModel] = None) -> Any:
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    if schema:
        # Request strict Pydantic parsing from Groq
        return groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            response_model=schema,
            messages=messages,
            max_tokens=1000
        )
    else:
        # Standard raw generation using the unpatched client
        response = raw_groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=1000
        )
        return response.choices[0].message.content

print("\n--- 3. Initializing MemoryClient ---")
# We use a very small buffer_size (5) and batch_size (3) to trigger L2 background tests faster.
memory = MemoryClient(
    llm_generate=llm_func,
    embed_text=embed_func,
    use_query_expansion=True,
    buffer_size=5,
    batch_size=3
)

USER_ID = "test_user_001"

print("\n=======================================================")
print("TEST 1: The Fast Path (L1 Cache) - < 50ms Expected")
print("=======================================================")
memory.add_message(USER_ID, "user", "Hi, my name is Ravi.")
memory.add_message(USER_ID, "assistant", "Hello Ravi! How can I help you?")
memory.add_message(USER_ID, "user", "I am working on an AI project.")

start_time = time.time()
context = memory.retrieve_context(USER_ID, "What is my name?")
latency = (time.time() - start_time) * 1000

print(f"-> Retrieval Latency: {latency:.2f} ms")
print(f"-> L1 Buffer (Instant Recall):\n{context['short_term_history']}")
print(f"-> L2 Graph Context (Should be empty): {context['long_term_graph_context']}")


print("\n=======================================================")
print("TEST 2 & 3: Background Batch Trigger & State Extraction")
print("=======================================================")
print("Adding messages to trigger batch...")
memory.add_message(USER_ID, "assistant", "That sounds interesting.")
memory.add_message(USER_ID, "user", "My favorite programming language is Python.")
memory.add_message(USER_ID, "user", "I also use the GROQ API with key 9999.")

print("-> Trigger hit! Waiting 15 seconds for background Graph construction to finish and respect Rate Limits...")
time.sleep(15)

context = memory.retrieve_context(USER_ID, "What language do I like?")
print("-> Global JSON State (Checking for fact extraction):")
print(context['global_state'])
print("\n-> L2 Graph Context (Checking for merged summary):")
print(context['long_term_graph_context'])


print("\n=======================================================")
print("TEST 4: Graph Routing & Dual Semantic Search")
print("=======================================================")
print("Adding completely new topic to trigger branch creation...")
for i in range(5):
    memory.add_message(USER_ID, "user", f"I am adopting a Golden Retriever puppy named Max (part {i}).")
    
print("-> Trigger hit! Waiting 15 seconds for background Graph construction to finish and respect Rate Limits...")
time.sleep(15)

start_time = time.time()
context = memory.retrieve_context(USER_ID, "What kind of dog am I getting?")
latency = (time.time() - start_time) * 1000

print(f"-> Retrieval Latency: {latency:.2f} ms")
print(f"-> Expanded Intent generated by LLM: {context['expanded_intent']}")
print(f"-> Retrieved Graph Context via Dual Search:\n{context['long_term_graph_context']}")
print("\nALL TESTS COMPLETED!")

print("\nGenerating Telemetry Explorer Report...")
memory.generate_report(USER_ID, filepath="METRICS_REPORT.md")
