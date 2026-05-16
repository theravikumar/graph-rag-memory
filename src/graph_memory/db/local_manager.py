import os
import uuid
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from datetime import datetime

from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON, ForeignKey, select
from sqlalchemy.orm import declarative_base, sessionmaker

import faiss

from ..schemas import Message, QuantitativeState, MemoryNode
from .base import MemoryDatabase

Base = declarative_base()

class MessageModel(Base):
    __tablename__ = 'messages'
    id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime)

class QuantitativeStateModel(Base):
    __tablename__ = 'quantitative_state'
    user_id = Column(String, primary_key=True)
    state_data = Column(JSON)

class MemoryNodeModel(Base):
    __tablename__ = 'memory_nodes'
    node_id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    topic_label = Column(String)
    topic_description = Column(Text)
    node_type = Column(String)
    summary = Column(Text)
    node_state = Column(JSON)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    # Store integer IDs for FAISS mapping
    faiss_id = Column(String, unique=True, nullable=True)

class NodeEdgeModel(Base):
    __tablename__ = 'node_edges'
    id = Column(String, primary_key=True)
    from_node_id = Column(String, index=True)
    to_node_id = Column(String, index=True)
    rel_type = Column(String)

class LocalMemoryManager(MemoryDatabase):
    """
    Implements the MemoryDatabase interface using SQLite for relational/JSON data
    and FAISS for the Dual Semantic Vectors.
    """
    def __init__(self, db_path: str = "sqlite:///memory.db", vector_dim: int = 384):
        self.engine = create_engine(db_path)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.vector_dim = vector_dim
        
        # Load FAISS indices from disk if they exist, otherwise create new ones
        self.label_index_path = "label_index.faiss"
        self.desc_index_path = "desc_index.faiss"
        
        if os.path.exists(self.label_index_path):
            self.label_index = faiss.read_index(self.label_index_path)
        else:
            self.label_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.vector_dim))
            
        if os.path.exists(self.desc_index_path):
            self.desc_index = faiss.read_index(self.desc_index_path)
        else:
            self.desc_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.vector_dim))
        
        # Automatically resume faiss_id index from existing database to prevent UNIQUE constraint errors
        self._next_faiss_id = 1
        with self.Session() as session:
            try:
                # Query all existing faiss_ids
                existing_ids = session.query(MemoryNodeModel.faiss_id).all()
                if existing_ids:
                    # Filter out None values and convert to integers
                    int_ids = [int(r[0]) for r in existing_ids if r[0] is not None and str(r[0]).isdigit()]
                    if int_ids:
                        self._next_faiss_id = max(int_ids) + 1
            except Exception as e:
                print(f"Warning: Could not resume faiss_id counter: {e}")

    def add_message(self, message: Message) -> None:
        with self.Session() as session:
            msg = MessageModel(
                id=str(message.id),
                user_id=message.user_id,
                role=message.role,
                content=message.content,
                timestamp=message.timestamp
            )
            session.add(msg)
            session.commit()

    def get_recent_messages(self, user_id: str, limit: int = 20) -> List[Message]:
        with self.Session() as session:
            query = select(MessageModel).where(MessageModel.user_id == user_id).order_by(MessageModel.timestamp.desc()).limit(limit)
            results = session.execute(query).scalars().all()
            return [Message(id=UUID(r.id), user_id=r.user_id, role=r.role, content=r.content, timestamp=r.timestamp) for r in reversed(results)]

    def delete_messages(self, message_ids: List[UUID]) -> None:
        with self.Session() as session:
            session.query(MessageModel).filter(MessageModel.id.in_([str(mid) for mid in message_ids])).delete()
            session.commit()

    def get_quantitative_state(self, user_id: str) -> QuantitativeState:
        with self.Session() as session:
            state = session.query(QuantitativeStateModel).filter_by(user_id=user_id).first()
            if state:
                return QuantitativeState(user_id=user_id, state_data=state.state_data)
            return QuantitativeState(user_id=user_id, state_data={})

    def update_quantitative_state(self, user_id: str, patches: Dict[str, Any]) -> None:
        with self.Session() as session:
            state = session.query(QuantitativeStateModel).filter_by(user_id=user_id).first()
            if not state:
                state = QuantitativeStateModel(user_id=user_id, state_data={})
                session.add(state)
            
            # Simple dict update logic (for nested, might need deeper merge)
            current_data = state.state_data.copy() if state.state_data else {}
            current_data.update(patches)
            state.state_data = current_data
            session.commit()

    def get_node(self, node_id: UUID) -> Optional[MemoryNode]:
        with self.Session() as session:
            n = session.query(MemoryNodeModel).filter_by(node_id=str(node_id)).first()
            if not n:
                return None
            
            # Fetch edges
            parents = session.query(NodeEdgeModel).filter_by(to_node_id=str(node_id), rel_type='parent').all()
            related = session.query(NodeEdgeModel).filter_by(from_node_id=str(node_id), rel_type='related').all()
            
            return MemoryNode(
                node_id=UUID(n.node_id),
                user_id=n.user_id,
                topic_label=n.topic_label,
                topic_description=n.topic_description,
                node_type=n.node_type,
                summary=n.summary,
                node_state=n.node_state,
                parent_ids=[UUID(p.from_node_id) for p in parents],
                related_node_ids=[UUID(r.to_node_id) for r in related],
                created_at=n.created_at,
                updated_at=n.updated_at
            )

    def add_node(self, node: MemoryNode) -> None:
        with self.Session() as session:
            faiss_id = self._next_faiss_id
            self._next_faiss_id += 1
            
            db_node = MemoryNodeModel(
                node_id=str(node.node_id),
                user_id=node.user_id,
                topic_label=node.topic_label,
                topic_description=node.topic_description,
                node_type=node.node_type,
                summary=node.summary,
                node_state=node.node_state,
                created_at=node.created_at,
                updated_at=node.updated_at,
                faiss_id=str(faiss_id)
            )
            session.add(db_node)
            
            # Add edges
            for pid in node.parent_ids:
                session.add(NodeEdgeModel(id=str(uuid.uuid4()), from_node_id=str(pid), to_node_id=str(node.node_id), rel_type='parent'))
            for rid in node.related_node_ids:
                session.add(NodeEdgeModel(id=str(uuid.uuid4()), from_node_id=str(node.node_id), to_node_id=str(rid), rel_type='related'))
                
            session.commit()
            
            # Add to FAISS
            if node.label_embedding:
                self.label_index.add_with_ids(np.array([node.label_embedding], dtype=np.float32), np.array([faiss_id]))
            if node.description_embedding:
                self.desc_index.add_with_ids(np.array([node.description_embedding], dtype=np.float32), np.array([faiss_id]))
                
            # Persist FAISS indices to disk
            faiss.write_index(self.label_index, self.label_index_path)
            faiss.write_index(self.desc_index, self.desc_index_path)

    def update_node(self, node: MemoryNode) -> None:
        with self.Session() as session:
            db_node = session.query(MemoryNodeModel).filter_by(node_id=str(node.node_id)).first()
            if db_node:
                db_node.summary = node.summary
                db_node.node_state = node.node_state
                db_node.updated_at = datetime.utcnow()
                session.commit()

    def get_all_nodes(self, user_id: str) -> List[MemoryNode]:
        with self.Session() as session:
            nodes = session.query(MemoryNodeModel).filter_by(user_id=user_id).all()
            result = []
            for n in nodes:
                result.append(MemoryNode(
                    node_id=UUID(n.node_id),
                    user_id=n.user_id,
                    topic_label=n.topic_label,
                    topic_description=n.topic_description,
                    node_type=n.node_type,
                    summary=n.summary,
                    node_state=n.node_state
                ))
            return result

    def add_relationship(self, from_node_id: UUID, to_node_id: UUID, rel_type: str = "parent") -> None:
        with self.Session() as session:
            session.add(NodeEdgeModel(id=str(uuid.uuid4()), from_node_id=str(from_node_id), to_node_id=str(to_node_id), rel_type=rel_type))
            session.commit()

    def search_nodes(
        self, 
        user_id: str, 
        label_vector: Optional[List[float]] = None, 
        desc_vector: Optional[List[float]] = None, 
        keyword: Optional[str] = None,
        top_k: int = 3
    ) -> List[Tuple[MemoryNode, float]]:
        import numpy as np
        
        # Determine which FAISS index to query
        vector_to_search = desc_vector if desc_vector else label_vector
        index_to_search = self.desc_index if desc_vector else self.label_index
        
        if not vector_to_search or index_to_search.ntotal == 0:
            return []
            
        # FAISS search
        query_vector = np.array([vector_to_search], dtype=np.float32)
        distances, faiss_ids = index_to_search.search(query_vector, top_k * 2) # Over-fetch in case of user mismatch
        
        results = []
        with self.Session() as session:
            for i in range(len(faiss_ids[0])):
                f_id = int(faiss_ids[0][i])
                dist = float(distances[0][i])
                
                if f_id == -1:
                    continue
                    
                # Map FAISS ID back to UUID and verify it belongs to user_id
                db_node = session.query(MemoryNodeModel).filter_by(faiss_id=str(f_id), user_id=user_id).first()
                if db_node:
                    node = MemoryNode(
                        node_id=db_node.node_id,
                        user_id=db_node.user_id,
                        topic_label=db_node.topic_label,
                        topic_description=db_node.topic_description,
                        node_type=db_node.node_type,
                        summary=db_node.summary,
                        node_state=db_node.node_state if db_node.node_state else {},
                        created_at=db_node.created_at,
                        updated_at=db_node.updated_at
                    )
                    results.append((node, dist))
                    
                    if len(results) == top_k:
                        break
                        
        return results
