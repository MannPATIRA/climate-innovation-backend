from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from common.pinecone_store import PineconeStore
from background_process.utils.process_log_manager import ProcessLogManager, ProcessingTask
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import requests
from common.supabase_client import supabase_operation_with_retry

class Processor(ABC):
    def __init__(self, process_log_manager: ProcessLogManager, pinecone_store: PineconeStore, chunk_size: int = 500):
        # Add a ChunkingStrategy class to take in constructor so we can use different chunking strategies LATER
        self.process_log_manager = process_log_manager
        self.pinecone_store = pinecone_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50
        )
        self.task_id = None

    def generate_content_hash(self, content: str) -> str:
        """Generate a hash for the content using SHA-256"""
        return hashlib.sha256(content.encode()).hexdigest()

    @supabase_operation_with_retry(max_retries=3, retry_delay=120)
    def create_task(self, task_type: ProcessingTask) -> int:
        """Create a new task record if it doesn't exist and return its ID"""
        self.task_id = self.process_log_manager.create_task(task_type.value)
        return self.task_id

    @supabase_operation_with_retry(max_retries=3, retry_delay=120)
    def log_progress(self, reference_id: str):
        """Log individual progress for a task"""
        if not self.task_id:
            raise ValueError("No task_id set. Task must be created before logging progress.")
        
        self.process_log_manager.log_progress(self.task_id, reference_id)

    @supabase_operation_with_retry(max_retries=3, retry_delay=120)
    def remove_from_logs(self, reference_id: str):
        """Remove the entry from processing logs once completed"""
        if not self.task_id:
            raise ValueError("No task_id set. Task must be created before removing from logs.")
        
        self.process_log_manager.remove_from_logs(self.task_id, reference_id)

    @abstractmethod
    def process(self, data: Dict[Any, Any]) -> Tuple[str, Dict[str, Any]]:
        pass 