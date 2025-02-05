import os
from dotenv import load_dotenv
from supabase import create_client
from common.pinecone_store import PineconeStore
from background_process.fetchers import PyAlexFetcher
from background_process.processors import PaperProcessor
from background_process.orchestrators import PaperOrchestrator

def test_paper_ingestion():
    # Load environment variables
    load_dotenv(override=True)
    
    # Initialize clients
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    supabase = create_client(supabase_url, supabase_key)
    pinecone_store = PineconeStore(index_name="climate-index")
    
    # Initialize components
    fetcher = PyAlexFetcher(supabase)
    processor = PaperProcessor(
        supabase, 
        pinecone_store,
        chunk_size=750,
        max_workers=10
    )
    
    # Create and run orchestrator
    orchestrator = PaperOrchestrator(
        fetcher=fetcher,
        processor=processor,
        batch_size=1000
    )
    
    orchestrator.run(country="GB")

if __name__ == "__main__":
    test_paper_ingestion() 