import os
from dotenv import load_dotenv
from supabase import create_client
from common.pinecone_store import PineconeStore
from background_process.fetchers import PyAlexFetcher
from background_process.processors import PaperProcessor
from itertools import islice
from typing import List, Dict, Any, Tuple

def batch_generator(generator, batch_size: int):
    """Convert a generator into batches"""
    while True:
        batch = list(islice(generator, batch_size))
        if not batch:
            break
        yield batch

def format_paper_data(paper_tuple: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Convert paper tuple from fetcher to dictionary format"""
    abstract, metadata = paper_tuple
    return {
        'abstract': abstract,
        'metadata': metadata
    }

def test_paper_ingestion():
    # Load environment variables
    load_dotenv()
    
    # Initialize clients
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    supabase = create_client(supabase_url, supabase_key)
    pinecone_store = PineconeStore(index_name="test-index")
    
    # Initialize components
    fetcher = PyAlexFetcher(supabase)
    processor = PaperProcessor(
        supabase, 
        pinecone_store,
        chunk_size=750,
        max_workers=5
    )
    
    # Get papers generator
    papers_generator = fetcher.fetch(country="GB")  # Example with US papers
    
    # Process in batches of 1000 papers
    BATCH_SIZE = 1000
    total_processed = 0
    
    for batch in batch_generator(papers_generator, BATCH_SIZE):
        print(f"\nProcessing batch of {len(batch)} papers...")
        
        # Convert tuples to dictionaries before processing
        formatted_batch = [format_paper_data(paper_tuple) for paper_tuple in batch]
        
        # Process the batch using thread pool
        batch_results = processor.process_batch(formatted_batch)
        
        # Print batch results
        total_processed += len(batch_results)
        print(f"Successfully processed {len(batch_results)} papers in this batch")
        print(f"Total papers processed: {total_processed}")
        
        # Optional: Print some details about processed papers
        for paper, record in batch_results[:5]:  # Show first 5 papers
            print(f"Processed: {paper.title}")
            print(f"ID: {record['id']}")
            print("-" * 50)

if __name__ == "__main__":
    test_paper_ingestion() 