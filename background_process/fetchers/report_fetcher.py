import os
import shutil
from typing import Generator
from abc import ABC
from .base import Fetcher


class ReportFetcher(Fetcher, ABC):
    pass


class LocalPDFFetcher(ReportFetcher):
    def __init__(self, directory: str):
        self.directory = directory
        self.temp_directory = os.path.join(os.path.dirname(directory), "processing_temp")
        if not os.path.exists(self.temp_directory):
            os.makedirs(self.temp_directory)

    def fetch(self) -> Generator[str, None, None]:
        for filename in os.listdir(self.directory):
            if filename.lower().endswith('.pdf'):
                print("considering file: ", filename)
                # Create temp copy
                source_path = os.path.join(self.directory, filename)
                temp_path = os.path.join(self.temp_directory, filename)
                shutil.copy2(source_path, temp_path)
                yield temp_path
                
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        """Cleanup temporary directory"""
        if os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory)


    def __del__(self):
        """Cleanup temporary directory when the fetcher is destroyed"""
        self.cleanup() 