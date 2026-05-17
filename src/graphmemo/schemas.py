from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator
from uuid import UUID, uuid4
from datetime import datetime

class Message(BaseModel):
    """Represents a single chat message in the short-term buffer."""
    id: UUID = Field(default_factory=uuid4)
    user_id: str = Field(..., description="The ID of the user this message belongs to")
    role: str = Field(..., description="'user' or 'assistant'")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class QuantitativeState(BaseModel):
    """Holds structured JSON data extracted from conversations."""
    user_id: str
    state_data: Dict[str, Any] = Field(default_factory=dict, description="Flat or nested JSON state")

class MemoryNode(BaseModel):
    """Represents a node in the Hierarchical Topic Graph."""
    node_id: UUID = Field(default_factory=uuid4)
    user_id: str
    parent_ids: List[UUID] = Field(default_factory=list, description="IDs of parent nodes (Graph structure)")
    related_node_ids: List[UUID] = Field(default_factory=list, description="IDs of horizontally linked nodes")
    
    topic_label: str = Field(..., description="Short topic name, < 4 words")
    topic_description: str = Field(..., description="Detailed description of what this node contains")
    node_type: str = Field(..., description="'root', 'branch', or 'leaf'")
    
    summary: str = Field(..., description="Factual bullet points of the conversation")
    node_state: Dict[str, Any] = Field(default_factory=dict, description="Node-specific quantitative JSON")
    
    # We don't store raw vectors in the Pydantic model directly as they are large and DB-specific,
    # but we can represent them as lists if needed for serialization.
    label_embedding: Optional[List[float]] = None
    description_embedding: Optional[List[float]] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class NodeAction(BaseModel):
    """Structured output expected from the LLM during Graph Construction."""
    action: str = Field(..., description="Must be one of: APPEND_TO_NODE, CREATE_CHILD_NODE, CREATE_NEW_BRANCH")
    target_node_id: Optional[UUID] = Field(None, description="Required for APPEND and CREATE_CHILD")
    new_topic_name: Optional[str] = Field(None, description="Required for CREATE_CHILD and CREATE_NEW_BRANCH")
    new_topic_description: Optional[str] = Field(None, description="Required for CREATE_CHILD and CREATE_NEW_BRANCH")
    updated_summary: str | List[str] = Field(..., description="The merged summary factual bullet points")
    quantitative_patches: Dict[str, Any] = Field(default_factory=dict, description="Any new JSON state facts to patch")
    
    @model_validator(mode='before')
    @classmethod
    def unwrap_schema_hallucinations(cls, data: Any) -> Any:
        # Defensive Programming: If the LLM wraps the output in a JSON Schema format
        if isinstance(data, dict):
            if "properties" in data and isinstance(data["properties"], dict):
                return data["properties"]
        return data
