from abc import ABC
from typing import Generator, Any, Dict, Tuple, List
from itertools import chain
from pyalex import Works, Topics
from tenacity import retry, stop_after_attempt, wait_exponential
from ..processors.base import ProcessingTask
from .base import Fetcher


class PaperFetcher(Fetcher, ABC):
    pass


class PyAlexFetcher(PaperFetcher):
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.task_id = self._get_paper_processing_task_id()
        self.cursor = self._get_main_cursor()
        self.current_cursor = self._get_current_cursor()
        # Store climate relevant topics as set for O(1) lookup
        self.climate_relevant_topics = set(self._get_climate_relevant_topics())
        print("number of climate relevant topics")
        print(len(self.climate_relevant_topics))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_paper_processing_task_id(self) -> int:
        """Create a new task record if it doesn't exist and return its ID"""
        # Check for existing task
        response = self.supabase.table('processor_progress') \
            .select("*") \
            .eq('task', ProcessingTask.PAPER_PROCESSING.value) \
            .execute()
        
        if response.data:
            # Return ID of existing task
            return response.data[0]["id"]
        
        # Create new task if none exists
        response = self.supabase.table('processor_progress').insert({
            "task": ProcessingTask.PAPER_PROCESSING.value
        }).execute()
        return response.data[0]["id"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_main_cursor(self) -> str:
        """Get the main cursor from the processing_tasks table"""
        response = self.supabase.table('processor_progress') \
            .select('cursor') \
            .eq('id', self.task_id) \
            .execute()
        
        return response.data[0].get('cursor', '*') if response.data else '*'

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_current_cursor(self) -> str:
        """Get the current_cursor from the processing_tasks table"""
        response = self.supabase.table('processor_progress') \
            .select('current_cursor') \
            .eq('id', self.task_id) \
            .execute()
        
        # If no current_cursor exists, use the main cursor value
        if not response.data or response.data[0].get('current_cursor') is None:
            return self._get_main_cursor()
        
        return response.data[0].get('current_cursor')

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _update_current_cursor(self, cursor: str):
        """Update the current_cursor in the processing_tasks table"""
        self.supabase.table('processor_progress') \
            .update({'current_cursor': cursor}) \
            .eq('id', self.task_id) \
            .execute()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _update_main_cursor(self):
        """Update the main cursor with the current_cursor value"""
        print("updating main cursor to: ", self.current_cursor)
        self.supabase.table('processor_progress') \
            .update({'cursor': self.current_cursor}) \
            .eq('id', self.task_id) \
            .execute()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_failed_papers(self) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Fetch and yield papers that failed to process previously"""
        # Get all paper IDs from processing logs for this task
        response = self.supabase.table('process_progress_logs') \
            .select('reference_id') \
            .eq('task_id', self.task_id) \
            .execute()

        if not response.data:
            return
        
        print("The number of failed papers retrieved: ", len(response.data))
        for record in response.data:
            openalex_id = record['reference_id']
            # Fetch the specific paper from OpenAlex
            work = Works()[openalex_id]
            print("Re Yielding this work: ", work.get('id'))
            if work:
                abstract = self._get_abstract(work)
                if abstract:
                    primary_topic = work.get('primary_topic', {})
                    if primary_topic is not None:
                        primary_topic_id = primary_topic.get('id')
                        if primary_topic_id and primary_topic_id in self.climate_relevant_topics:
                            metadata = {
                                'id': work.get('id'),
                                'doi': work.get('doi'),
                                'title': work.get('title')
                            }
                            yield abstract, metadata
                    else:
                        print("primary topic is null: here are topics: ")
                        print(work.get('topics', "No topics"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_climate_relevant_topics(self) -> List[str]:
        """Get list of topic IDs that were assessed as climate-relevant"""
        all_topics = []
        page = 0
        page_size = 1000
        
        while True:
            response = self.supabase.table('openalex_topic_assessments') \
                .select('topic_id') \
                .eq('is_climate_relevant', True) \
                .range(page * page_size, (page + 1) * page_size - 1) \
                .execute()
            
            if not response.data:  # No more results
                break
                
            all_topics.extend(record['topic_id'] for record in response.data)
            page += 1
            
        return all_topics

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
        # First yield any failed papers
        #yield from self._get_failed_papers()

        # Continue with normal fetching process
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
                publication_year=">2009"
            ) \
            .filter(
                primary_location={"source": {"type": "journal|repository"}}
            ) \
            .filter(
                primary_topic={"domain": {"id": "1|3"}}
            ) \
            .sort(publication_date="asc")
        

        res, meta = query.get(per_page=1, return_meta=True)
        print("paper meta info: ")
        print(meta)


        # Use cursor to get all results
        cursor = self.cursor
        while cursor:
            works, meta = query.get(per_page=200, cursor=cursor, return_meta=True)
            cursor = meta.get('next_cursor')
            print(f"Another 200 papers fetched with cursor")
            papers_yielded = 0
            
            for paper in works:
                abstract = self._get_abstract(paper)
                if abstract:  # Only yield papers with abstracts
                    primary_topic = paper.get('primary_topic', {})
                    if primary_topic is not None:
                        primary_topic_id = primary_topic.get('id')
                        if primary_topic_id and primary_topic_id in self.climate_relevant_topics:
                            metadata = {
                                'id': paper.get('id'),
                                'doi': paper.get('doi'),
                                'title': paper.get('title')
                            }
                            papers_yielded += 1
                            yield abstract, metadata
                    else:
                        print("primary topic is null: here are topics: ")
                        print(paper.get('topics', "No topics"))
            
            print(f"Yielded {papers_yielded} out of 200 papers in this batch")
            
            # Update the current cursor
            if cursor:
                self.current_cursor = cursor
                self._update_current_cursor(cursor)

    
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

    def mark_batch_complete(self):
        """Mark the current batch as complete by updating the main cursor"""
        self._update_main_cursor() 