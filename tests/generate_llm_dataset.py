import os
import json
import random
import time
from typing import List, Optional
from pydantic import BaseModel, Field
import instructor
from groq import Groq

# Ensure API Key is set
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"

print("Initializing Groq LLM for Dataset Synthesis...")
client = instructor.from_groq(Groq(api_key=os.environ.get("GROQ_API_KEY")))

class Message(BaseModel):
    is_query: bool = Field(description="True if the message is a question testing memory.")
    content: str = Field(description="The organic conversational text.")
    established_fact_topic: str = Field(description="If stating a concrete fact, what is the topic? (e.g., 'Dog breed', 'Favorite Color'). Empty if just conversational noise.")
    established_fact_value: str = Field(description="If stating a concrete fact, what is the exact value? (e.g., 'Golden Retriever', 'Blue'). Empty if noise.")

class MessageBatch(BaseModel):
    messages: List[Message]

TOTAL_TURNS = 2000
BATCH_SIZE = 10
ITERATIONS = TOTAL_TURNS // BATCH_SIZE

dataset = []
established_facts = [] # List of dicts: {"topic": ..., "value": ..., "injected_turn": ...}

current_turn = 0

print(f"Starting organic synthesis for {TOTAL_TURNS} turns ({ITERATIONS} batches)...")

for i in range(ITERATIONS):
    # Determine what this batch should organically cover
    context_directive = f"Generate a natural, highly organic batch of {BATCH_SIZE} conversational messages."
    
    forced_query = None
    if len(established_facts) > 0 and random.random() < 0.4:
        # 40% chance per batch to test a previously established fact
        forced_query = random.choice(established_facts)
        context_directive += f"\nCRITICAL: One of the messages MUST be a question asking about '{forced_query['topic']}'. The answer should be '{forced_query['value']}', but do NOT state the answer in the question."
    
    context_directive += "\nThe other messages should casually introduce new concrete facts about your life/work, or just be natural conversational noise and tangents."
    
    success = False
    retries = 3
    while not success and retries > 0:
        try:
            print(f"Synthesizing Batch {i+1}/{ITERATIONS} (Turns {current_turn} to {current_turn+BATCH_SIZE-1})...")
            batch = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                response_model=MessageBatch,
                messages=[
                    {"role": "system", "content": "You are a human user having a flowing, meandering conversation. Output highly organic, unstructured text that sounds like a real person typing."},
                    {"role": "user", "content": context_directive}
                ],
                max_tokens=2000
            )
            
            # Process the LLM output and rigorously track the Temporal logic in Python
            query_injected_this_batch = False
            for msg in batch.messages:
                entry = {
                    "id": current_turn,
                    "type": "query" if msg.is_query else "statement",
                    "content": msg.content,
                    "ground_truth": "",
                    "is_temporal_test": False,
                    "injected_turn_id": "N/A"
                }
                
                if msg.is_query:
                    if forced_query and not query_injected_this_batch:
                        # Map this query to the forced fact to guarantee perfect Ground Truth tracking
                        entry["ground_truth"] = forced_query["value"]
                        entry["is_temporal_test"] = True
                        entry["injected_turn_id"] = forced_query["injected_turn"]
                        query_injected_this_batch = True
                    else:
                        # If the LLM hallucinated an extra query, label it as short-term unknown
                        entry["ground_truth"] = "unknown_short_term"
                else:
                    # It's a statement. Did the LLM introduce a new fact?
                    if msg.established_fact_topic and msg.established_fact_value:
                        # Save it to Python memory so we can query it thousands of turns later!
                        established_facts.append({
                            "topic": msg.established_fact_topic,
                            "value": msg.established_fact_value,
                            "injected_turn": current_turn
                        })
                
                dataset.append(entry)
                current_turn += 1
                
            success = True
            
        except Exception as e:
            print(f"API Error (Rate Limit/Parse): {e}. Retrying in 15s...")
            time.sleep(15)
            retries -= 1
            
    # Respect Groq Rate Limits
    time.sleep(4)

with open("dataset_organic_2000.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("\n=================================")
print(f"Successfully generated 2000 fully organic, LLM-synthesized turns!")
print(f"Total concrete facts established and tracked for temporal testing: {len(established_facts)}")
print(f"Dataset saved to dataset_organic_2000.json")
