from .fetchers import ReportFetcher, Fetcher, PaperFetcher
from .processors import ReportProcessor, Processor, PaperProcessor
from .summary_processors import SummaryProcessor, Summarizer
from typing import Optional, Dict, Any, Tuple
import os
from abc import ABC, abstractmethod
from itertools import islice


class Orchestrator(ABC):

    def __init__(
            self,
            fetcher: Fetcher,
            processor: Processor,
            summarizer: Optional[SummaryProcessor] = None
    ):
        self.fetcher = fetcher
        self.processor = processor
        self.summarizer = summarizer

    @abstractmethod
    def run(self):
        """Main process to orchestrate the ingestion and vectorising of documents"""


class ReportOrchestrator(Orchestrator):

    def process_single_report(self, report_path: str) -> None:
        """Process a single report"""
        try:

            data = {'report_path': report_path}

            (document, report_record) = self.processor.process(data)

            # Process summary if summary processor exists
            if self.summarizer:
                try:
                    self.summarizer.summarize(document, report_record, report_path)
                except Exception as e:
                    print(f"Error processing summary for {report_path}: {str(e)}")


        except Exception as e:
            print(f"Error processing report {report_path}: {str(e)}")
        
        finally:
            # Clean up the PDF file
            if os.path.exists(report_path):
                os.remove(report_path)

    def run(self):
        """Main process to fetch and process reports"""
        for report_path in self.fetcher.fetch():
            self.process_single_report(report_path)


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




