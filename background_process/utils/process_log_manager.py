from supabase import Client
from enum import Enum
from typing import List

class ProcessingTask(Enum):
    REPORT_FETCHING = "report_fetching"
    REPORT_PROCESSING = "report_processing"
    PAPER_PROCESSING = "paper_processing" 
    TOPIC_PROCESSING = "topic_processing"

class ProcessLogManager:
    """
    Manages process logging operations in the database.
    Abstracts database operations from the processor classes.
    """
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
    
    def create_task(self, task_type: str) -> int:
        """Create a new task record if it doesn't exist and return its ID"""
        # Check for existing task
        response = self.supabase.table('processor_progress') \
            .select("*") \
            .eq('task', task_type) \
            .execute()
        
        if response.data:
            # Return ID of existing task
            return response.data[0]["id"]
        
        # Create new task if none exists
        response = self.supabase.table('processor_progress').insert({
            "task": task_type
        }).execute()
        return response.data[0]["id"]

    def log_progress(self, task_id: int, reference_id: str):
        """Log individual progress for a task"""
        self.supabase.table('process_progress_logs').insert({
            "task_id": task_id,
            "reference_id": reference_id
        }).execute()

    def remove_from_logs(self, task_id: int, reference_id: str):
        """Remove the entry from processing logs once completed"""
        self.supabase.table('process_progress_logs') \
            .delete() \
            .eq('task_id', task_id) \
            .eq('reference_id', reference_id) \
            .execute()
            
    def is_already_processed(self, task_id: int, reference_id: str) -> bool:
        """Check if a reference ID has already been processed for a given task"""
        response = self.supabase.table('process_progress_logs') \
            .select("*") \
            .eq('task_id', task_id) \
            .eq('reference_id', reference_id) \
            .execute()
        
        return len(response.data) > 0
        
    def get_all_processed_references(self, task_id: int) -> List[str]:
        """Get all reference IDs that have been processed for a given task"""
        response = self.supabase.table('process_progress_logs') \
            .select("reference_id") \
            .eq('task_id', task_id) \
            .execute()
        
        return [item["reference_id"] for item in response.data] 