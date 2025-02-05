from abc import ABC, abstractmethod
from itertools import chain
from typing import Generator, Any, Dict, Tuple, List
import os
import shutil
from pyalex import Works, Topics


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
    def __init__(self, supabase_client):
        self.supabase = supabase_client

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
        # Get climate-relevant topic IDs
        climate_relevant_topics = self._get_climate_relevant_topics()
        
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
                publication_year=">2015"
            ) \
            .filter(
                cited_by_count=">3"
            ) \
            .filter(
                primary_location={"source": {"type": "journal|repository"}}
            ) \
            .sort(cited_by_count="desc")  # Get most cited papers first
        res, meta = query.get(per_page=1, return_meta=True)
        print("paper meta info: ")
        print(meta)
        # Use pagination to get all results
        for page in chain(query.paginate(per_page=200)):
            print("Another 200 papers fetched")
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

    
    def _get_climate_relevant_topics(self) -> List[str]:
        """Get list of topic IDs that were assessed as climate-relevant"""
        response = self.supabase.table('openalex_topic_assessments') \
            .select('topic_id') \
            .eq('is_climate_relevant', True) \
            .execute()
        return [record['topic_id'] for record in response.data]

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

class TopicFetcher(Fetcher):
    def fetch(self) -> Generator[Dict[str, Any], None, None]:
        """
        Fetches topics and sample works from OpenAlex.
        
        Yields:
            Dict containing topic info and sample works
        """
        cursor = "*"
        while cursor:
            topics, meta = Topics().get(per_page=200, cursor=cursor, return_meta=True)
            cursor = meta["next_cursor"]
            print("number of topics: ", len(topics))
            for topic in topics:
                # Get 3 random sample works for this topic
                sample_works = Works() \
                    .filter(topics={'id': topic['id']}) \
                    .sort(cited_by_count="desc") \
                    .select(['id', 'title', 'abstract_inverted_index_v3', 'abstract_inverted_index']) \
                    .paginate(per_page=3)
                    
                # Get the first page of results (3 works)
                sample_abstracts = []
                for page in chain(sample_works):
                    for work in page:
                        if abstract := self._get_abstract(work):
                            sample_abstracts.append({
                                'title': work.get('title'),
                                'abstract': abstract
                            })
                    break # break after first page (we have already seen 3 papers)
                yield {
                    'topic_id': topic['id'],
                    'topic_name': topic['display_name'],
                    'topic_description': topic.get('description', ''),
                    'sample_works': sample_abstracts
                }
            

    def _get_abstract(self, work):
        # Reuse the abstract extraction logic from PyAlexFetcher
        inverted_index = work.get('abstract_inverted_index_v3') or work.get('abstract_inverted_index')
        
        if not inverted_index:
            return None
        
        max_position = max(pos for positions in inverted_index.values() for pos in positions)
        words = [''] * (max_position + 1)
        
        for word, positions in inverted_index.items():
            for position in positions:
                words[position] = word
        
        return ' '.join(words)
