import threading
from typing import List, Callable, Optional
from uuid import UUID

from ..schemas import Message
from ..db.base import MemoryDatabase

class BufferManager:
    """
    Manages the short-term conversational buffer (L1 Cache).
    Stores raw messages for perfect recent recall and triggers the L2 background batch process.
    """
    def __init__(
        self, 
        db: MemoryDatabase, 
        buffer_size: int = 20, 
        batch_size: int = 10,
        batch_callback: Optional[Callable[[str, List[Message]], None]] = None
    ):
        """
        :param db: The underlying database adapter.
        :param buffer_size: Maximum size of the short-term buffer before triggering a batch.
        :param batch_size: Number of messages to extract and send to the background graph constructor.
        :param batch_callback: A function/method to call asynchronously when the buffer fills up.
        """
        self.db = db
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.batch_callback = batch_callback

    def add_message(self, user_id: str, role: str, content: str) -> Message:
        """
        Adds a new message to the user's short term buffer.
        If the buffer reaches `buffer_size`, it asynchronously triggers the batch process.
        """
        msg = Message(user_id=user_id, role=role, content=content)
        self.db.add_message(msg) 
        
        # Check if we need to trigger a batch
        self._check_and_trigger_batch(user_id)
        
        return msg

    def flush(self, user_id: str) -> None:
        """
        Synchronously processes all remaining messages in the buffer.
        Useful when shutting down the system or forcing a graph update.
        """
        recent_messages = self.db.get_recent_messages(user_id, limit=self.buffer_size)
        if not recent_messages:
            return
            
        print(f"  [System]: Flushing {len(recent_messages)} messages to Graph Memory...")
        try:
            if self.batch_callback:
                self.batch_callback(user_id, recent_messages)
            batch_ids = [msg.id for msg in recent_messages]
            self.db.delete_messages(batch_ids)
        except Exception as e:
            print(f"[BufferManager] Error flushing buffer: {e}")

    def get_context(self, user_id: str) -> List[Message]:
        """
        Instantly retrieves the raw short-term memory buffer (L1 Cache) for the user.
        Latency: < 10ms.
        """
        # Fetch up to the buffer limit
        return self.db.get_recent_messages(user_id, limit=self.buffer_size)

    def _check_and_trigger_batch(self, user_id: str) -> None:
        """
        Checks the current buffer size and spins off a background thread to process 
        the oldest messages if the threshold is met.
        """
        recent_messages = self.db.get_recent_messages(user_id, limit=self.buffer_size + 1)
        
        if len(recent_messages) >= self.buffer_size:
            # We have hit the buffer limit. Extract the oldest `batch_size` messages.
            # Since recent_messages is ordered latest-first, the oldest are at the end.
            # Wait, `get_recent_messages` actually returns chronological order in our DB implementation!
            # Let's assume they are chronological (oldest first).
            oldest_batch = recent_messages[:self.batch_size]
            
            if self.batch_callback:
                # Fire and forget on a separate thread to maintain < 400ms user latency
                bg_thread = threading.Thread(
                    target=self._run_batch_task, 
                    args=(user_id, oldest_batch)
                )
                bg_thread.daemon = True
                bg_thread.start()

    def _run_batch_task(self, user_id: str, batch: List[Message]) -> None:
        """
        The wrapper for the background task. It calls the Graph Constructor (batch_callback)
        and upon success, deletes the batched messages from the short-term buffer.
        """
        try:
            # Run the heavy LLM extraction and Graph merging logic
            self.batch_callback(user_id, batch)
            
            # If successful, remove these messages from the short term DB
            batch_ids = [msg.id for msg in batch]
            self.db.delete_messages(batch_ids)
            
        except Exception as e:
            # In a production system, log this error and perhaps implement a retry queue
            print(f"[BufferManager] Error processing background batch for user {user_id}: {e}")
