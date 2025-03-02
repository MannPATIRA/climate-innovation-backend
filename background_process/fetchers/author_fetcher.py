from abc import ABC
from typing import Generator, Any, Dict, List
from pyalex import Authors
from tenacity import retry, stop_after_attempt, wait_exponential
from ..processors.base import ProcessingTask
from .base import Fetcher

class AuthorFetcher(Fetcher, ABC):
    pass

class PyAlexAuthorFetcher(AuthorFetcher):
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.task_id = self._get_author_processing_task_id()
        self.cursor = self._get_main_cursor()
        self.current_cursor = self._get_current_cursor()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_author_processing_task_id(self) -> int:
        """Create a new task record if it doesn't exist and return its ID"""
        response = self.supabase.table('processor_progress') \
            .select("*") \
            .eq('task', ProcessingTask.AUTHOR_PROCESSING.value) \
            .execute()
        
        if response.data:
            return response.data[0]["id"]
        
        response = self.supabase.table('processor_progress').insert({
            "task": ProcessingTask.AUTHOR_PROCESSING.value
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

    def _format_institutions_list(self, affiliations: List[Dict]) -> List[Dict]:
        """Format institutions from affiliations data"""
        institutions = []
        for affiliation in affiliations:
            if 'institution' in affiliation:
                inst = affiliation['institution']
                institutions.append({
                    'id': inst.get('id'),
                    'display_name': inst.get('display_name'),
                    'country_code': inst.get('country_code'),
                    'type': inst.get('type'),
                    'years': affiliation.get('years', [])
                })
        return institutions

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _author_exists_in_db(self, openalex_id: str) -> bool:
        """Check if author already exists in database"""
        response = self.supabase.table('authors') \
            .select('id') \
            .eq('openalex_id', openalex_id) \
            .execute()
        return bool(response.data)

    def fetch(self, country: str = None) -> Generator[Dict[str, Any], None, None]:
        """
        Generates author metadata using the pyalex library.
        
        Args:
            country: Optional country code to filter authors by.
            
        Yields:
            Dict containing author metadata
        """
        query = Authors()
        if country:
            query = query.filter(last_known_institution={'country_code': country})

        cursor = self.cursor
        while cursor:
            authors, meta = query.get(per_page=200, cursor=cursor, return_meta=True)
            cursor = meta.get('next_cursor')
            print(f"Fetched batch of 200 authors with cursor")
            authors_yielded = 0
            
            for author in authors:
                openalex_id = author.get('id')
                print(f"Processing author: {author.get('display_name')} ({openalex_id})")
                
                # Only yield authors that exist in our database
                if self._author_exists_in_db(openalex_id):
                    metadata = {
                        'id': openalex_id,
                        'display_name': author.get('display_name'),
                        'orcid': author.get('orcid'),
                        'institutions': self._format_institutions_list(author.get('affiliations', [])),
                        'topics': author.get('topics', []),
                        'works_count': author.get('works_count', 0),
                        'cited_by_count': author.get('cited_by_count', 0),
                        'h_index': author.get('summary_stats', {}).get('h_index', 0)
                    }
                    authors_yielded += 1
                    yield metadata
            
            print(f"Yielded {authors_yielded} out of 200 authors in this batch")
            
            if cursor:
                self.current_cursor = cursor
                self._update_current_cursor(cursor)

    def mark_batch_complete(self):
        """Mark the current batch as complete by updating the main cursor"""
        self._update_main_cursor() 