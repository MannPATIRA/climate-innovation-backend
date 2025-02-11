from itertools import islice
from typing import Dict, Any, Tuple
from .base import Orchestrator
from ..fetchers.paper_fetcher import PaperFetcher
from ..processors.paper_processor import PaperProcessor
from ..summary_processors import SummaryProcessor
from typing import Optional


class PaperOrchestrator(Orchestrator):
    def __init__(
            self,
            fetcher: PaperFetcher,
            processor: PaperProcessor,
            summarizer: Optional[SummaryProcessor] = None,
            batch_size: int = 1000
    ):
        super().__init__(fetcher, processor, summarizer)
        self.batch_size = batch_size

    def _batch_generator(self, generator, batch_size: int):
        """Convert a generator into batches"""
        while True:
            batch = list(islice(generator, batch_size))
            if not batch:
                break
            yield batch

    def _format_paper_data(self, paper_tuple: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Convert paper tuple from fetcher to dictionary format"""
        abstract, metadata = paper_tuple
        return {
            'abstract': abstract,
            'metadata': metadata
        }

    def run(self, country: str = "GB"):
        """Main process to fetch and process papers in batches"""
        total_processed = 0
        
        # Get papers generator
        papers_generator = self.fetcher.fetch(country=country)
        
        # Process in batches
        for batch in self._batch_generator(papers_generator, self.batch_size):
            print(f"\nProcessing batch of {len(batch)} papers...")
            
            # Convert tuples to dictionaries before processing
            formatted_batch = [self._format_paper_data(paper_tuple) for paper_tuple in batch]
            
            try:
                # Process the batch using thread pool
                batch_results = self.processor.process_batch(formatted_batch)
                
                # Print batch results
                total_processed += len(batch_results)
                print(f"Successfully processed {len(batch_results)} papers in this batch")
                print(f"Total papers processed: {total_processed}")
                
                # Mark batch as complete by updating the main cursor
                self.fetcher.mark_batch_complete()
                
                # Print some details about processed papers (first 5 in batch)
                for paper, record in batch_results[:5]:
                    print(f"Processed: {paper.title}")
                    print(f"ID: {record['id']}")
                    print("-" * 50)
                    
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue 