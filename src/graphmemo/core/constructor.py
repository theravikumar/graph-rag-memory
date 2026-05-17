import json
from typing import List, Callable, Dict, Any, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel

from ..schemas import Message, MemoryNode, NodeAction
from ..db.base import MemoryDatabase

class GraphConstructor:
    """
    Handles the asynchronous L2 Background Batch Process.
    Runs LLM prompts to extract quantitative state and safely build the Hierarchical Graph.
    """
    def __init__(
        self, 
        db: MemoryDatabase, 
        llm_generate: Callable[[str, str, Optional[BaseModel]], Any],
        embed_text: Callable[[str], List[float]]
    ):
        """
        :param db: The underlying MemoryDatabase.
        :param llm_generate: A function `(system_prompt, user_prompt, pydantic_schema=None) -> output`.
                             If pydantic_schema is provided, it must return a parsed Pydantic object.
                             Otherwise, it returns a string.
        :param embed_text: A function `(text) -> List[float]` to generate vector embeddings.
        """
        self.db = db
        self.llm_generate = llm_generate
        self.embed_text = embed_text

    def process_batch(self, user_id: str, batch: List[Message]) -> None:
        """
        The main pipeline executed by the background worker.
        """
        if not batch:
            return

        # 1. Format the transcript
        transcript = "\n".join([f"[{msg.timestamp.strftime('%H:%M:%S')}] {msg.role}: {msg.content}" for msg in batch])

        # 2. Extract Quantitative JSON State
        self._extract_and_update_state(user_id, transcript)

        # 3. Summarize the Batch
        summary = self._summarize_batch(transcript)

        # 4. The Retrieve-Then-Decide Pattern
        # Generate an embedding for the summary to find conceptually similar existing nodes.
        summary_vector = self.embed_text(summary)
        top_nodes = self.db.search_nodes(user_id, desc_vector=summary_vector, top_k=3)

        # 5. Determine Graph Action (Hallucination Prevention)
        action = self._determine_graph_action(transcript, summary, top_nodes)

        # 6. Apply the Database Updates safely
        self._apply_graph_action(user_id, action)

    def _extract_and_update_state(self, user_id: str, transcript: str) -> None:
        """Runs the LLM to extract hard quantitative facts as JSON."""
        sys_prompt = (
            "You are a state extraction engine. Analyze the conversation and extract ONLY hard, "
            "quantitative facts (e.g., birthdays, software versions, ticket IDs, explicit preferences). "
            "Return them as a simple dictionary. If no facts exist, return an empty dictionary."
        )
        
        class StateExtraction(BaseModel):
            extracted_facts: Dict[str, str]
            
        try:
            # We use Pydantic/Instructor to guarantee valid JSON syntax
            result = self.llm_generate(sys_prompt, transcript, StateExtraction)
            patches = result.extracted_facts if hasattr(result, 'extracted_facts') else {}
            
            if patches:
                self.db.update_quantitative_state(user_id, patches)
        except Exception as e:
            print(f"[GraphConstructor] State extraction failed: {e}")

    def _summarize_batch(self, transcript: str) -> str:
        """Condenses the raw messages into factual bullet points."""
        sys_prompt = (
            "Summarize the following chat transcript into raw, factual bullet points. "
            "Do not use paragraph form. Retain core entities and user intent. Keep it extremely concise."
        )
        return self.llm_generate(sys_prompt, transcript, None)

    def _determine_graph_action(self, transcript: str, new_summary: str, top_nodes: List[tuple]) -> NodeAction:
        """
        Forces the LLM to choose whether to append, create a child, or branch out,
        providing existing nodes as context to prevent duplication.
        """
        sys_prompt = (
            "You are the Memory Graph Architect. You must route new conversational memory into a Hierarchical Graph.\n"
            "You are given a summary of new conversation, and the Top 3 closest existing Topic Nodes.\n"
            "RULES:\n"
            "1. If the new info perfectly fits an existing node, choose APPEND_TO_NODE and provide the target_node_id.\n"
            "2. If it is a specific sub-category of an existing node, choose CREATE_CHILD_NODE and provide the target_node_id.\n"
            "3. If and ONLY if it is completely unrelated to the existing nodes, choose CREATE_NEW_BRANCH.\n"
            "IMPORTANT: DO NOT generate a JSON schema. Output ONLY the raw JSON object containing the actual fields (action, updated_summary, etc) with their string/array values."
        )
        
        # Format existing nodes for the LLM
        nodes_context = "Top 3 Existing Nodes:\n"
        if not top_nodes:
            nodes_context += "No existing nodes in memory (Empty Graph).\n"
        else:
            for node, score in top_nodes:
                nodes_context += f"ID: {node.node_id} | Label: {node.topic_label} | Desc: {node.topic_description}\n"
        
        user_prompt = f"{nodes_context}\n\nNew Summary to route:\n{new_summary}"
        
        # We pass the NodeAction Pydantic model to force structured output
        return self.llm_generate(sys_prompt, user_prompt, NodeAction)

    def _apply_graph_action(self, user_id: str, action: NodeAction) -> None:
        """Safely mutates the database Graph based on the LLM's structural decision."""
        if action.action == "APPEND_TO_NODE" and action.target_node_id:
            node = self.db.get_node(action.target_node_id)
            if node:
                # We overwrite the summary with the LLM's merged updated_summary
                summary_val = action.updated_summary if isinstance(action.updated_summary, str) else "\n".join(action.updated_summary)
                node.summary = summary_val
                if action.quantitative_patches:
                    node.node_state.update(action.quantitative_patches)
                self.db.update_node(node)
                
        elif action.action == "CREATE_CHILD_NODE" and action.target_node_id:
            # Enforce depth constraints here (pseudocode: if depth(target) >= 4 -> fallback to APPEND)
            new_node = MemoryNode(
                user_id=user_id,
                parent_ids=[action.target_node_id],
                topic_label=action.new_topic_name or "Unknown Subtopic",
                topic_description=action.new_topic_description or "No description",
                node_type="leaf",
                summary=action.updated_summary if isinstance(action.updated_summary, str) else "\n".join(action.updated_summary),
                node_state=action.quantitative_patches,
                label_embedding=self.embed_text(action.new_topic_name or ""),
                description_embedding=self.embed_text(action.new_topic_description or "")
            )
            self.db.add_node(new_node)
            
        elif action.action == "CREATE_NEW_BRANCH":
            new_node = MemoryNode(
                user_id=user_id,
                topic_label=action.new_topic_name or "New Topic",
                topic_description=action.new_topic_description or "No description",
                node_type="root",
                summary=action.updated_summary if isinstance(action.updated_summary, str) else "\n".join(action.updated_summary),
                node_state=action.quantitative_patches,
                label_embedding=self.embed_text(action.new_topic_name or ""),
                description_embedding=self.embed_text(action.new_topic_description or "")
            )
            self.db.add_node(new_node)
