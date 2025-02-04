from abc import ABC, abstractmethod
from itertools import chain
from typing import Generator, Any, Dict, Tuple
import os
import shutil
from pyalex import Works


class Fetcher(ABC):

    @abstractmethod
    def fetch(self, **kwargs) -> Generator[Any, None, None]:
        """
        Yields one path / url at a time.
        This allows for processing one document at a time and cleaning up after.
        """
        pass


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

class PaperFetcher(Fetcher, ABC):
    pass


class PyAlexFetcher(PaperFetcher):
    def fetch(self, country: str, **kwargs) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """
        Generates paper abstracts using the pyalex library.

        Args:
            country: The country to filter research papers by.
            **kwargs: Additional filters for the search query.

        Yields:
            Tuple containing:
                - abstract (str): The paper's abstract
                - metadata (Dict): Dictionary containing id, doi, and title of the paper
        """
        # Build the query with filters
        query = Works() \
            .filter(
            authorships={"institutions": {"country_code": country}}
        ) \
            .filter(
            type="article|preprint|book-chapter|dissertation"
        ) \
            .filter(
            authorships={"is_corresponding": "true"}
        ) \
            .filter(
            authorships={"affiliations": {
                "institution_ids": "https://openalex.org/I82284825|https://openalex.org/I47508984|https://openalex"
                                   ".org/I98677209|https://openalex.org/I130828816|https://openalex.org/I241749|https"
                                   "://openalex.org/I4210092773"}}
        ) \
            .filter(
            publication_year=">1999"
        ) \
            .filter(
            primary_topic={"domain": {"id": "!2"}}
        ) \
            .filter(
            primary_topic={"domain": {"id": "!4"}}
        )

        # Use pagination to get all results
        for page in chain(query.paginate(per_page=200)):
            for paper in page:
                #abstract = paper.get("abstract", "None")
                abstract = self._get_abstract(paper)
                if abstract:  # Only yield papers with abstracts
                    metadata = {
                        'id': paper.get('id'),
                        'doi': paper.get('doi'),
                        'title': paper.get('title')
                    }
                    yield abstract, metadata

    def _get_abstract(self, work):
        # Try the v3 index first
        inverted_index = work.get('abstract_inverted_index_v3') or work.get('abstract_inverted_index')
        
        if not inverted_index:
            return None
        
        # Reconstruct the abstract from the inverted index
        # The index is a dict where keys are words and values are lists of positions
        # We need to create a list long enough to hold all words
        max_position = max(pos for positions in inverted_index.values() for pos in positions)
        words = [''] * (max_position + 1)
        
        # Place each word in its correct position(s)
        for word, positions in inverted_index.items():
            for position in positions:
                words[position] = word
        
        # Join the words to form the complete abstract
        return ' '.join(words)
