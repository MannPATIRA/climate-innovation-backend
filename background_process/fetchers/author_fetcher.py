import time
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
        self.page_size = 1000  # Supabase max page size
        self.rate_limit = 0.1  # 10 requests per second for OpenAlex

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
    def _get_unprocessed_authors_page(self, page: int) -> List[Dict[str, Any]]:
        """Get a page of unprocessed authors from Supabase"""
        response = self.supabase.table('authors') \
            .select('openalex_id') \
            .eq('author_processed', False) \
            .range(page * self.page_size, (page + 1) * self.page_size - 1) \
            .execute()
        return response.data

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_author_from_openalex(self, openalex_id: str) -> Dict[str, Any]:
        """Fetch single author directly from OpenAlex"""
        author = Authors()[openalex_id]
        time.sleep(self.rate_limit)  # Rate limiting
        return author

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
        Generates author metadata by fetching unprocessed authors from Supabase
        and then getting their details from OpenAlex.
        
        Args:
            country: Optional country code to filter authors by (not used in this version)
            
        Yields:
            Dict containing author metadata
        """
        page = 0
        while True:
            # Get page of unprocessed authors
            authors = self._get_unprocessed_authors_page(page)
            
            # Break if no more authors
            if not authors:
                break
                
            print(f"Fetched {len(authors)} unprocessed authors from page {page}")
            
            for author_data in authors:
                openalex_id = author_data['openalex_id']
                print(f"Fetching author details for: {openalex_id}")
                
                try:
                    author = self._get_author_from_openalex(openalex_id)
                    
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
                    yield metadata
                except Exception as e:
                    print(f"Error fetching author {openalex_id}: {type(e).__name__} - {str(e)}")
                    continue
            
            page += 1

    def mark_batch_complete(self):
        """Mark the current batch as complete by updating the main cursor"""
        self._update_main_cursor() 