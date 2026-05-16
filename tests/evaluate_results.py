import os
import json
import csv
import time
from pydantic import BaseModel, Field
import instructor
from groq import Groq

# Ensure API Key is set
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"

print("Initializing AI Judge (Groq Llama-3.1-8b)...")
client = instructor.from_groq(Groq(api_key=os.environ.get("GROQ_API_KEY")))

class Judgement(BaseModel):
    is_correct: bool = Field(description="True if the retrieved context contains the exact ground truth to answer the question.")
    reasoning: str = Field(description="A brief 1-sentence explanation of why it is correct or incorrect.")

JSONL_FILE = "TEST_RESULTS.jsonl"
CSV_FILE = "EVALUATION_REPORT.csv"

if not os.path.exists(JSONL_FILE):
    print(f"Error: {JSONL_FILE} not found. You must start the run_massive_test.py script first.")
    exit(1)

# Read all queries that have been tested so far
results = []
with open(JSONL_FILE, "r") as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))

print(f"Found {len(results)} queries to evaluate.")

# Initialize CSV
write_header = not os.path.exists(CSV_FILE)
csv_f = open(CSV_FILE, "a", newline="", encoding="utf-8")
writer = csv.writer(csv_f)
if write_header:
    writer.writerow(["Query_ID", "Injected_Turn_ID", "Temporal_Test", "Question", "Ground_Truth", "Retrieval_Latency_ms", "Total_Latency_ms", "Final_Answer", "AI_Judgement", "Reasoning"])

correct_count = 0
temporal_correct = 0
temporal_total = 0

print("Starting Evaluation Pipeline... (Enforcing strict pacing to avoid rate limits)")

for r in results:
    q_id = r["id"]
    query = r["query"]
    gt = r["ground_truth"]
    context = r["retrieved_context"]
    final_answer = r.get("final_answer", "")
    ret_lat = r.get("retrieval_latency_ms", 0)
    tot_lat = r.get("total_latency_ms", 0)
    is_temporal = r.get("is_temporal_test", False)
    inj_turn = r.get("injected_turn_id", "N/A")
    
    if is_temporal:
        temporal_total += 1

    sys_prompt = "You are a strict AI Judge evaluating a RAG memory pipeline."
    user_prompt = f"""
    Evaluate if the RAG system successfully answered the question based on the ground truth.
    
    Question: {query}
    Ground Truth Expected: {gt}
    
    Generated Answer:
    {final_answer}
    
    Task: Did the Generated Answer accurately state the Ground Truth?
    """
    
    try:
        judgement = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            response_model=Judgement,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200
        )
        
        # Log to CSV
        writer.writerow([q_id, inj_turn, is_temporal, query, gt, ret_lat, tot_lat, final_answer, judgement.is_correct, judgement.reasoning])
        csv_f.flush()
        
        if judgement.is_correct:
            correct_count += 1
            if is_temporal:
                temporal_correct += 1
                
        print(f"[{q_id}] (Injected @ {inj_turn}) Correct: {judgement.is_correct} | RetLat: {ret_lat}ms | {judgement.reasoning}")
        
    except Exception as e:
        print(f"[{q_id}] Evaluation Failed (Rate Limit or API Error): {e}")
        writer.writerow([q_id, inj_turn, is_temporal, query, gt, ret_lat, tot_lat, final_answer, "ERROR", str(e)])
        
    # Strict Pacing (2 seconds per evaluation)
    time.sleep(2)

csv_f.close()

print("\n=================================")
print("EVALUATION COMPLETE")
print(f"Total Queries Evaluated: {len(results)}")
print(f"Overall Accuracy: {(correct_count / max(1, len(results))) * 100:.1f}%")
if temporal_total > 0:
    print(f"Long-Term Temporal Accuracy: {(temporal_correct / temporal_total) * 100:.1f}%")
print(f"Detailed report saved to {CSV_FILE}")
