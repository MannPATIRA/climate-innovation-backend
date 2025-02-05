from .fetchers import ReportFetcher, Fetcher, PaperFetcher
from .processors import ReportProcessor, Processor, PaperProcessor
from .summary_processors import SummaryProcessor, Summarizer
from typing import Optional, Dict, Any
import os
from abc import ABC, abstractmethod


class Orchestrator(ABC):

    def __init__(
            self,
            fetcher: Fetcher,
            processor: Processor,
            summarizer: Optional[SummaryProcessor]
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

    def process_single_paper(self, abstract: str, metadata: Dict[str, Any]) -> None:
        """Process a single paper"""
        try:
            data = {
                'abstract': abstract,
                'metadata': metadata
            }
            
            (paper, paper_record) = self.processor.process(data)

        except Exception as e:
            print(f"Error processing paper {metadata['title']}: {str(e)}")

    def run(self):
        """Main process to fetch and process papers"""
        for abstract, metadata in self.fetcher.fetch(country="GB"):  # Fetch uk paper
            self.process_single_paper(abstract, metadata)

