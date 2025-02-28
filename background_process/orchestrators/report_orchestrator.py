import os
from .base import Orchestrator
from ..processors.summary_processor import SummaryProcessor


class ReportOrchestrator(Orchestrator):

    def __init__(self, fetcher, processor, summarizer=None):
        super().__init__(fetcher, processor, summarizer)

    def process_single_report(self, report_path: str) -> None:
        """Process a single report"""
        try:
            data = {'report_path': report_path}

            # Process the report
            (document, report_record) = self.processor.process(data)

            # Process summary if summary processor exists
            if self.summarizer:
                try:
                    # Create data for summary processor
                    summary_data = {'report_id': report_record["id"]}
                    
                    # Process the summary
                    summaries, summary_records = self.summarizer.process(summary_data)
                    
                    print(f"Generated {len(summaries)} summaries for report {report_record['report_title']}")
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