from background_process.fetchers.report_fetcher import WebScrapingReportFetcher
from dotenv import load_dotenv

def test_first_pdf():
    load_dotenv(override=True)
    """Test WebScrapingReportFetcher by getting first PDF"""
    
    print("\nInitializing WebScrapingReportFetcher...")
    fetcher = WebScrapingReportFetcher(llm_model="gpt-4o-mini", max_depth=2, use_cache=False)
    
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