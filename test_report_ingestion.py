import os
from dotenv import load_dotenv
from supabase import create_client
from common.pinecone_store import PineconeStore
from background_process.fetchers import LocalPDFFetcher, WebScrapingReportFetcher
from background_process.utils.process_log_manager import ProcessLogManager
from background_process.processors import ReportProcessor, SummaryProcessor
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
    # fetcher = LocalPDFFetcher("./test_reports")
    process_log_manager = ProcessLogManager(supabase)
    fetcher = WebScrapingReportFetcher(
        process_log_manager,
        llm_model="gpt-4o-mini",
        max_depth=1,
        use_cache=False
    )
    processor = ReportProcessor(
        supabase,
        process_log_manager,
        pinecone_store,
        chunk_size=750,
    )
    summarizer = SummaryProcessor(
        supabase,
        process_log_manager,
        pinecone_store,
        chunk_size=4000,
    )
    
    # Create and run orchestrator
    orchestrator = ReportOrchestrator(
        fetcher=fetcher,
        processor=processor,
        summarizer=summarizer
    )
    
    orchestrator.run()

if __name__ == "__main__":
    test_report_ingestion() 