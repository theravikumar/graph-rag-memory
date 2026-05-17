from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from ..schemas import Message, QuantitativeState, MemoryNode

class MemoryDatabase(ABC):
    """
    Abstract Base Class defining the required interface for any database backend
    (e.g., Local SQLite+FAISS, PostgreSQL+pgvector).
    """

    # --- Short-Term Buffer Operations ---
    @abstractmethod
    def add_message(self, message: Message) -> None:
        """Appends a new message to the short-term rolling buffer."""
        pass

    @abstractmethod
    def get_recent_messages(self, user_id: str, limit: int = 20) -> List[Message]:
        """Retrieves the most recent messages for a user to serve as immediate context."""
        pass

    @abstractmethod
    def delete_messages(self, message_ids: List[UUID]) -> None:
        """Removes older messages from the buffer after they have been batched and merged."""
        pass

    # --- Quantitative State Operations ---
    @abstractmethod
    def get_quantitative_state(self, user_id: str) -> QuantitativeState:
        """Retrieves the flat/nested JSON state for the user."""
        pass

    @abstractmethod
    def update_quantitative_state(self, user_id: str, patches: Dict[str, Any]) -> None:
        """Applies JSON patches to the user's master state."""
        pass

    # --- Hierarchical Graph Operations ---
    @abstractmethod
    def get_node(self, node_id: UUID) -> Optional[MemoryNode]:
        """Retrieves a specific graph node by ID."""
        pass

    @abstractmethod
    def add_node(self, node: MemoryNode) -> None:
        """Inserts a new node into the Graph (and updates vector indices)."""
        pass

    @abstractmethod
    def update_node(self, node: MemoryNode) -> None:
        """Updates an existing node (e.g., updating its summary or appending to its state)."""
        pass
        
    @abstractmethod
    def add_relationship(self, from_node_id: UUID, to_node_id: UUID, rel_type: str = "parent") -> None:
        """Creates a relationship edge between two nodes."""
        pass

    @abstractmethod
    def get_all_nodes(self, user_id: str) -> List[MemoryNode]:
        """Retrieves all graph nodes for reporting/visualization."""
        pass

    # --- Retrieval Operations (The Semantic Router) ---
    @abstractmethod
    def search_nodes(
        self, 
        user_id: str, 
        label_vector: Optional[List[float]] = None, 
        desc_vector: Optional[List[float]] = None, 
        keyword: Optional[str] = None,
        top_k: int = 3
    ) -> List[Tuple[MemoryNode, float]]:
        """
        The core Dual Semantic Search + Keyword traversal engine.
        Returns a list of tuples containing the Node and its similarity score.
        """
        pass
