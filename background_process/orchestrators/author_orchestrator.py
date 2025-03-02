from itertools import islice
from typing import Dict, Any
from .base import Orchestrator
from ..fetchers.author_fetcher import AuthorFetcher
from ..processors.author_processor import AuthorProcessor

class AuthorOrchestrator(Orchestrator):
    def __init__(
            self,
            fetcher: AuthorFetcher,
            processor: AuthorProcessor,
            batch_size: int = 1000
    ):
        super().__init__(fetcher, processor)
        self.batch_size = batch_size

    def _batch_generator(self, generator, batch_size: int):
        """Convert a generator into batches"""
        while True:
            batch = list(islice(generator, batch_size))
            if not batch:
                break
            yield batch

    def run(self, country: str = None):
        """Main process to fetch and process authors in batches"""
        total_processed = 0
        
        # Get authors generator
        authors_generator = self.fetcher.fetch(country=country)
        
        # Process in batches
        for batch in self._batch_generator(authors_generator, self.batch_size):
            print(f"\nProcessing batch of {len(batch)} authors...")
            
            try:
                # Process the batch using thread pool
                batch_results = self.processor.process_batch(batch)
                
                # Print batch results
                total_processed += len(batch_results)
                print(f"Successfully processed {len(batch_results)} authors in this batch")
                print(f"Total authors processed: {total_processed}")
                
                # Mark batch as complete by updating the main cursor
                self.fetcher.mark_batch_complete()
                
                # Print some details about processed authors (first 5 in batch)
                for author, record in batch_results[:5]:
                    print(f"Processed: {author.display_name}")
                    print(f"ID: {record['id']}")
                    print(f"Institutions: {author.institutions_str}")
                    print("-" * 50)
                    
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue