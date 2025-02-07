import os
import asyncio
from dotenv import load_dotenv
from supabase import create_client
from background_process.fetchers import TopicFetcher
from background_process.processors import TopicProcessor
import json
from itertools import islice

async def batch_generator(generator, batch_size):
    """Convert a generator into batches"""
    batch = []
    for item in generator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # Don't forget the last partial batch
        yield batch

async def main():
    # Load environment variables
    load_dotenv(override=True)
    
    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    supabase = create_client(supabase_url, supabase_key)
    
    # Initialize processor and fetcher
    processor = TopicProcessor(supabase_client=supabase)
    fetcher = TopicFetcher()
    
    # Get topics generator
    topic_generator = fetcher.fetch()
    
    # Process in batches of 5 topics
    BATCH_SIZE = 10
    
    async for batch in batch_generator(topic_generator, BATCH_SIZE):
        print(f"\nProcessing batch of {len(batch)} topics...")
        
        # Process the batch asynchronously
        batch_results = await processor.process_batch(batch)
        
        # Print batch results
        for assessment, record in batch_results:
            if assessment:  # Skip if None (already processed)
                print(f"Processed topic {record['id']}")
                print(f"Climate Relevant: {assessment.is_climate_relevant}")
                print(f"Analysis: {assessment.analysis[:200]}...")
                print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main()) 