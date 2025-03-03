import time
import asyncio
import aiohttp
from abc import ABC
from typing import Generator, Any, Dict, List
from pyalex import Authors
from tenacity import retry, stop_after_attempt, wait_exponential
from ..processors.base import ProcessingTask
from .base import Fetcher

class AuthorFetcher(Fetcher, ABC):
    pass

class PyAlexAuthorFetcher(AuthorFetcher):
    def __init__(self, supabase_client, page_size: int = 1000, batch_size: int = 100, openalex_key: str = None):
        self.supabase = supabase_client
        self.openalex_key = openalex_key
        self.task_id = self._get_author_processing_task_id()
        self.page_size = page_size  # Supabase max page size
        self.batch_size = batch_size  # Number of concurrent requests
        self.rate_limit = 1  # Wait 1 second between batches

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

    async def _get_author_from_openalex_async(self, session: aiohttp.ClientSession, openalex_id: str) -> Dict[str, Any]:
        """Fetch single author directly from OpenAlex asynchronously"""
        # Extract the ID from the full URL if needed
        author_id = openalex_id.split('/')[-1] if '/' in openalex_id else openalex_id
        url = f"https://api.openalex.org/authors/{author_id}"
        
        # Add email authentication header if premium key is available
        headers = {}
        if self.openalex_key:
            headers['Authorization'] = f'Bearer {self.openalex_key}'
            
        async with session.get(url, headers=headers) as response:
            return await response.json()

    async def _fetch_batch_async(self, authors_batch: List[Dict]) -> List[Dict[str, Any]]:
        """Fetch a batch of authors concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for author_data in authors_batch:
                openalex_id = author_data['openalex_id']
                tasks.append(self._get_author_from_openalex_async(session, openalex_id))
            return await asyncio.gather(*tasks, return_exceptions=True)

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

    def fetch(self) -> Generator[Dict[str, Any], None, None]:
        """
        Generates author metadata by fetching unprocessed authors from Supabase
        and then getting their details from OpenAlex in concurrent batches.
        """
        page = 0
        while True:
            # Get page of unprocessed authors
            authors = self._get_unprocessed_authors_page(page)
            
            # Break if no more authors
            if not authors:
                page = 0
                time.sleep(10) # wait 10 seconds before starting over
                continue
                
            print(f"Fetched {len(authors)} unprocessed authors from page {page}")
            
            # Process authors in batches
            for i in range(0, len(authors), self.batch_size):
                batch = authors[i:i + self.batch_size]
                
                # Fetch batch concurrently
                results = asyncio.run(self._fetch_batch_async(batch))
                print(f"Fetched {len(results)} authors")
                # Process results
                for author_data, author in zip(batch, results):
                    openalex_id = author_data['openalex_id']
                    
                    if isinstance(author, Exception):
                        print(f"Error fetching author {openalex_id}: {type(author).__name__} - {str(author)}")
                        continue
                        
                    try:
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
                        print(f"Error processing author {openalex_id}: {type(e).__name__} - {str(e)}")
                        continue
                
                # Rate limiting between batches
                time.sleep(self.rate_limit)
            
            page += 1

    def mark_batch_complete(self):
        """Mark the current batch as complete by updating the main cursor"""
        pass