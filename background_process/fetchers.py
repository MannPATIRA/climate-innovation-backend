from abc import ABC, abstractmethod
from typing import Generator
import os
import shutil

class AbstractReportFetcher(ABC):
    @abstractmethod
    def fetch_reports(self) -> Generator[str, None, None]:
        """
        Yields one report path/url at a time.
        This allows for processing one report at a time and cleaning up after.
        """
        pass

class LocalPDFFetcher(AbstractReportFetcher):
    def __init__(self, directory: str):
        self.directory = directory
        self.temp_directory = os.path.join(os.path.dirname(directory), "_temp")
        if not os.path.exists(self.temp_directory):
            os.makedirs(self.temp_directory)

    def fetch_reports(self) -> Generator[str, None, None]:
        for filename in os.listdir(self.directory):
            if filename.lower().endswith('.pdf'):
                print("considering file: ", filename)
                # Create temp copy
                source_path = os.path.join(self.directory, filename)
                temp_path = os.path.join(self.temp_directory, filename)
                shutil.copy2(source_path, temp_path)
                yield temp_path

    def __del__(self):
        """Cleanup temporary directory when the fetcher is destroyed"""
        if os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory) 