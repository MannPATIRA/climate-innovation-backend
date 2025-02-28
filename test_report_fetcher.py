from background_process.fetchers.report_fetcher import WebScrapingReportFetcher
from background_process.utils.process_log_manager import ProcessLogManager
from supabase import create_client
from dotenv import load_dotenv
import os

def test_first_pdf():
    load_dotenv(override=True)
    """Test WebScrapingReportFetcher by getting first PDF"""
    
    print("\nInitializing WebScrapingReportFetcher...")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    supabase = create_client(supabase_url, supabase_key)
    process_log_manager = ProcessLogManager(supabase)
    fetcher = WebScrapingReportFetcher(process_log_manager, llm_model="gpt-4o-mini", max_depth=1, use_cache=False)
    
    print("\nStarting fetch process...")
    try:
        # Get just the first PDF
        for pdf_path in fetcher.fetch():
            print(f"\nSuccessfully found and downloaded PDF:")
            print(f"Path: {pdf_path}")
            break  # Stop after first PDF
            
    except Exception as e:
        print(f"\nError during fetch: {str(e)}")
    finally:
        print("\nCleaning up...")
        fetcher.cleanup()

if __name__ == "__main__":
    test_first_pdf() 