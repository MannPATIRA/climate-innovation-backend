from supabase import create_client
from common.pinecone_store import PineconeStore
from .fetchers import LocalPDFFetcher
from .processors import ReportProcessor
from .summary_processors import SummaryProcessor
from .orchestrators import ReportOrchestrator
import os
from common.supabase_client import init_supabase
from dotenv import load_dotenv
from .web_fetcher import WebReportFetcher

def main():
    # Initialize clients
    supabase = init_supabase()

    pinecone_store = PineconeStore(
        index_name="test-index",
        model="multilingual-e5-large"
    )

    # Initialize components
    fetcher = LocalPDFFetcher("./test_reports")
    report_processor = ReportProcessor(supabase, pinecone_store)
    summary_processor = SummaryProcessor(None, supabase, pinecone_store)  # Summarizer to be implemented later

    # Create and run orchestrator
    orchestrator = ReportOrchestrator(fetcher, report_processor, None)
    orchestrator.run()
    
    base_url = "https://iris.thegiin.org/share/id/47226x678e3dca05e43/"
    
    with WebReportFetcher(base_url) as fetcher:
        for report_path in fetcher.fetch():
            if report_path:
                print(f"Successfully downloaded: {report_path}")


if __name__ == "__main__":
    main() 