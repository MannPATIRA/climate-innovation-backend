import os
from supabase import create_client, Client, create_async_client, AsyncClient
from dotenv import load_dotenv

def init_supabase() -> Client:
    load_dotenv(override=True)
    """Initialize and return a Supabase client instance."""
    supabase_url = os.getenv("REMOTE_SUPABASE_URL")
    supabase_key = os.getenv("REMOTE_SUPABASE_KEY")
    # print(supabase_key)
    # print(supabase_url)
    
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