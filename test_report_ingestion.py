import os
from dotenv import load_dotenv
from supabase import create_client
from common.pinecone_store import PineconeStore
from background_process.fetchers import LocalPDFFetcher
from background_process.processors import ReportProcessor
from background_process.orchestrators import ReportOrchestrator

def test_report_ingestion():
    # Load environment variables
    load_dotenv(override=True)
    
    # Initialize clients
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    supabase = create_client(supabase_url, supabase_key)
    pinecone_store = PineconeStore(index_name="climate-index")
    
    # Initialize components
    fetcher = LocalPDFFetcher("./test_reports")
    processor = ReportProcessor(
        supabase, 
        pinecone_store,
        chunk_size=750,
    )
    
    # Create and run orchestrator
    orchestrator = ReportOrchestrator(
        fetcher=fetcher,
        processor=processor
    )
    
    orchestrator.run()

if __name__ == "__main__":
    test_report_ingestion() 