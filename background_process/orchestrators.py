from .fetchers import ReportFetcher, Fetcher, PaperFetcher
from .processors import ReportProcessor, Processor, PaperProcessor
from .summary_processors import SummaryProcessor, Summarizer
from typing import Optional
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

            (content, report_record) = self.processor.process(report_path)

            # Process summary if summary processor exists
            if self.summarizer:
                try:
                    self.summarizer.summarize(content, report_record, report_path)
                except Exception as e:
                    print(f"Error processing summary for {report_path}: {str(e)}")

            # Clean up the PDF file
            if os.path.exists(report_path):
                os.remove(report_path)

        except Exception as e:
            print(f"Error processing report {report_path}: {str(e)}")

    def run(self):
        """Main process to fetch and process reports"""
        for report_path in self.fetcher.fetch():
            self.process_single_report(report_path)


class PaperOrchestrator(Orchestrator):

    @abstractmethod
    def run(self):
        """Main process to orchestrate the ingestion and vectorising of documents"""
