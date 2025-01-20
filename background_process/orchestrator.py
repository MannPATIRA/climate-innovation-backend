from .fetchers import AbstractReportFetcher
from .report_processor import ReportProcessor
from .summary_processor import SummaryProcessor
from typing import Optional
import os

class Orchestrator:
    def __init__(
        self,
        fetcher: AbstractReportFetcher,
        report_processor: ReportProcessor,
        summary_processor: Optional[SummaryProcessor]
    ):
        self.fetcher = fetcher
        self.report_processor = report_processor
        self.summary_processor = summary_processor

    def process_single_report(self, report_path: str) -> None:
        """Process a single report"""
        try:
            # Convert PDF to text and get content hash
            content = self.report_processor.convert_pdf_to_text(report_path)
            content_hash = self.report_processor.generate_content_hash(content)
            print("content hash: ", content_hash)
            # Check if report exists and get data if it does
            existing_report = self.report_processor.get_report(content_hash)
            if not existing_report:
                # Add to Supabase
                report_record = self.report_processor.add_report_to_db(content, content_hash)
                
                # Add to Pinecone
                report_metadata = {
                    "report_id": report_record["id"],
                    "content_hash": content_hash,
                }
                self.report_processor.chunk_and_embed(content, report_metadata)
            else:
                print(f"Report {report_path} already processed.")
                report_record = existing_report[0]
            
            # Process summary if summary processor exists
            if self.summary_processor:
                try:
                    summary = self.summary_processor.generate_summary(content)
                    summary_hash = self.summary_processor.generate_content_hash(summary)
                    
                    # Only process summary if it hasn't been processed before
                    if not self.summary_processor.get_summary(summary_hash):
                        # Add to Supabase
                        summary_record = self.summary_processor.add_summary_to_db(
                            summary, 
                            report_record["id"],
                            summary_hash
                        )
                        
                        # Add to Pinecone
                        summary_metadata = {
                            "report_id": report_record["id"],
                            "summary_id": summary_record["id"],
                            "content_hash": summary_hash,
                            "type": "summary"
                        }
                        self.summary_processor.chunk_and_embed(summary, summary_metadata)
                    else:
                        print(f"Summary for report {report_path} already exists.")
                except Exception as e:
                    print(f"Error processing summary for {report_path}: {str(e)}")

            # Clean up the PDF file
            if os.path.exists(report_path):
                os.remove(report_path)

        except Exception as e:
            print(f"Error processing report {report_path}: {str(e)}")

    def run(self):
        """Main process to fetch and process reports"""
        for report_path in self.fetcher.fetch_reports():
            self.process_single_report(report_path) 