import os
from .base import Orchestrator


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