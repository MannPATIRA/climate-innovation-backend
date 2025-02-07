from background_process.fetchers import PyAlexFetcher

def test_pyalex_fetcher():
    # Initialize the fetcher
    fetcher = PyAlexFetcher()
    
    # Test with a specific country (using Australia as an example)
    country = "US"
    
    # Counter to limit the number of results for testing
    count = 0
    max_results = 450
    
    print(f"\nFetching papers from {country}...")
    for abstract, metadata in fetcher.fetch(country=country):
        print("\n---Paper Details---")
        print(f"Title: {metadata['title']}")
        print(f"DOI: {metadata['doi']}")
        print(f"ID: {metadata['id']}")
        print("\nAbstract preview:", abstract, "...")
        
        count += 1
        if count >= max_results:
            break

if __name__ == "__main__":
    test_pyalex_fetcher() 