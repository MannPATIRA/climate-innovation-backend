import os
from supabase import create_client, Client
from dotenv import load_dotenv

def init_supabase() -> Client:
    load_dotenv()
    """Initialize and return a Supabase client instance."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("Supabase credentials not found in environment variables!")
    
    return create_client(supabase_url, supabase_key)