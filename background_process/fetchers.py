from abc import ABC, abstractmethod
from typing import Generator
import os

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

    def fetch_reports(self) -> Generator[str, None, None]:
        for filename in os.listdir(self.directory):
            if filename.lower().endswith('.pdf'):
                print("considering file: ", filename)
                yield os.path.join(self.directory, filename) 