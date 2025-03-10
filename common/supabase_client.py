import os
import threading
import time
from functools import wraps
from typing import Callable, Any

from supabase import create_client, Client, create_async_client, AsyncClient
from dotenv import load_dotenv
import postgrest
from httpx import RemoteProtocolError, ReadTimeout, ConnectTimeout

# Global lock and flag for retry operations
_retry_lock = threading.Lock()
_is_waiting = False

def supabase_operation_with_retry(max_retries=3, retry_delay=120):
    """
    Decorator for Supabase operations that handles timeout errors
    by waiting and retrying the operation. Uses a global lock
    and flag to ensure all threads wait when one thread encounters a timeout.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global _is_waiting
            
            # Check if we're in a waiting period before even trying
            if _is_waiting:
                print("Supabase is in cooldown period, waiting before execution...")
                with _retry_lock:
                    pass  # Just wait for lock to be released
            
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (postgrest.exceptions.APIError,
                        ReadTimeout, ConnectTimeout, TimeoutError) as e:
                    # Check if it's an APIError but not the specific connection pool timeout
                    if isinstance(e, postgrest.exceptions.APIError):
                        # Check if it's a retryable error
                        error_code = getattr(e, 'code', None)
                        error_message = str(e).lower()
                        
                        # List of retryable error codes
                        retryable_codes = ['PGRST002', 'PGRST003', '57014', '25P02']
                        
                        # Check if it's not a retryable error
                        if (error_code not in retryable_codes and
                            'connection pool' not in error_message and
                            'statement timeout' not in error_message and
                            'transaction is aborted' not in error_message and
                            'timing out' not in error_message):
                            # It's a different API error, so raise it immediately
                            raise
                    
                    retries += 1
                    print(f"Supabase service error: {str(e)}")
                    print(f"Retry attempt {retries}/{max_retries}")
                    
                    if retries < max_retries:
                        # Try to acquire the lock - if we can't, another thread is already waiting
                        if not _retry_lock.acquire(blocking=False):
                            print("Another thread is already waiting for Supabase, joining wait...")
                            # Wait for the lock to be released by the thread doing the waiting
                            with _retry_lock:
                                pass  # Just wait for lock to be released
                        else:
                            try:
                                # Set the waiting flag so new operations will wait
                                _is_waiting = True
                                print(f"Waiting {retry_delay} seconds before retrying...")
                                time.sleep(retry_delay)
                            finally:
                                # Clear the waiting flag
                                _is_waiting = False
                                # Release the lock when done waiting
                                _retry_lock.release()
                    else:
                        print("Max retries reached. Operation failed.")
                        raise
            return None
        return wrapper
    return decorator

def init_supabase() -> Client:
    """Initialize and return a Supabase client instance."""
    load_dotenv(override=True)
    supabase_url = os.getenv("REMOTE_SUPABASE_URL")
    supabase_key = os.getenv("REMOTE_SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("Supabase credentials not found in environment variables!")
    
    return create_client(supabase_url, supabase_key)

async def init_supabase_async() -> AsyncClient:
    """Initialize and return an async Supabase client instance."""
    load_dotenv(override=True)
    supabase_url = os.getenv("REMOTE_SUPABASE_URL")
    supabase_key = os.getenv("REMOTE_SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("Supabase credentials not found in environment variables!")
    
    return await create_async_client(supabase_url, supabase_key)