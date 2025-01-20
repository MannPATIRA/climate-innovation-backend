from supabase import create_client
from common.pinecone_store import PineconeStore
from .fetchers import LocalPDFFetcher
from .processors import ReportProcessor
from .summary_processors import SummaryProcessor
from .orchestrators import Orchestrator, ReportOrchestrator
import os
from dotenv import load_dotenv

def main():
    load_dotenv()

    # Initialize clients
    supabase = create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_KEY")
    )

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

if __name__ == "__main__":
    main() 