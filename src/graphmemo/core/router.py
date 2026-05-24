import json
from typing import List, Callable, Dict, Any, Optional
from pydantic import BaseModel

from ..schemas import Message, MemoryNode, QuantitativeState
from ..db.base import MemoryDatabase

class SemanticRouter:
    """
    Handles the Synchronous Read Path.
    Retrieves the most relevant context in < 500ms using Dual Semantic Search and Query Expansion.
    """
    def __init__(
        self, 
        db: MemoryDatabase, 
        llm_generate: Callable[[str, str, Optional[BaseModel]], Any],
        embed_text: Callable[[str], List[float]],
        use_query_expansion: bool = True,
        retrieval_top_k: int = 2,
        history_limit: int = 20
    ):
        """
        :param db: The underlying MemoryDatabase.
        :param llm_generate: The LLM function (used for query expansion).
        :param embed_text: The vector embedding function.
        :param use_query_expansion: If True, uses the LLM to expand the query before searching. 
                                    Increases accuracy massively, but adds ~200-400ms latency.
        """
        self.db = db
        self.llm_generate = llm_generate
        self.embed_text = embed_text
        self.use_query_expansion = use_query_expansion
        self.retrieval_top_k = retrieval_top_k
        self.history_limit = history_limit

    def retrieve_context(self, user_id: str, query: str) -> Dict[str, Any]:
        """
        The main retrieval pipeline. Returns a structured dict containing all context
        ready to be injected into the final chatbot prompt.
        """
        # 1. Query Expansion (The Latency Trade-off)
        expanded_instructions = query
        if self.use_query_expansion:
            expanded_instructions = self._expand_query(query)

        # 2. Dual Vector Generation
        # Vector 1 for matching exact Topic Labels
        label_vector = self.embed_text(query)
        # Vector 2 for matching conceptual Topic Descriptions
        desc_vector = self.embed_text(expanded_instructions)

        # 3. Dual Semantic Traversal
        # Returns tuples of (MemoryNode, combined_score)
        top_node_tuples = self.db.search_nodes(
            user_id=user_id, 
            label_vector=label_vector, 
            desc_vector=desc_vector, 
            keyword=query, # Fallback BM25
            top_k=self.retrieval_top_k
        )
        
        nodes = [t[0] for t in top_node_tuples]

        # 4. Graph Cross-Link Traversal
        # Pull in any nodes linked horizontally to our top hits
        extended_nodes = self._traverse_cross_links(nodes)

        # 5. Fetch Quantitative JSON State
        global_state = self.db.get_quantitative_state(user_id)

        # 6. Fetch L1 Short-Term Buffer
        # (This just grabs the immediate recent messages from the DB)
        recent_messages = self.db.get_recent_messages(user_id, limit=self.history_limit)

        # 7. Format the return payload
        return {
            "short_term_history": self._format_messages(recent_messages),
            "global_state": global_state.state_data,
            "long_term_graph_context": self._format_graph_nodes(extended_nodes),
            "expanded_intent": expanded_instructions if self.use_query_expansion else None
        }

    def _expand_query(self, query: str) -> str:
        """Uses a fast LLM call to rewrite the query into detailed instructions/intents."""
        sys_prompt = (
            "You are a query expansion engine for a retrieval system. "
            "Analyze the user's query and output a detailed description of the underlying intent, "
            "synonyms, and broad concepts related to it. Output ONLY the expanded text, no conversational filler."
        )
        try:
            return self.llm_generate(sys_prompt, query, None)
        except Exception as e:
            print(f"[SemanticRouter] Query expansion failed, falling back to raw query. Error: {e}")
            return query

    def _traverse_cross_links(self, primary_nodes: List[MemoryNode]) -> List[MemoryNode]:
        """Fetches nodes that are horizontally linked via related_node_ids to prevent Topic Bleed."""
        all_nodes = {str(n.node_id): n for n in primary_nodes}
        
        for node in primary_nodes:
            for related_id in node.related_node_ids:
                if str(related_id) not in all_nodes:
                    related_node = self.db.get_node(related_id)
                    if related_node:
                        all_nodes[str(related_id)] = related_node
                        
        return list(all_nodes.values())

    def _format_messages(self, messages: List[Message]) -> str:
        """Formats the raw L1 buffer for the LLM prompt."""
        return "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in messages])

    def _format_graph_nodes(self, nodes: List[MemoryNode]) -> str:
        """Formats the retrieved L2 graph nodes into readable text for the LLM."""
        if not nodes:
            return "No relevant long-term memory found."
            
        formatted = ""
        for n in nodes:
            formatted += f"\n--- Topic: {n.topic_label} ---\n"
            formatted += f"Description: {n.topic_description}\n"
            formatted += f"Summary Facts:\n{n.summary}\n"
            if n.node_state:
                formatted += f"Node Specific Data: {json.dumps(n.node_state)}\n"
        return formatted
