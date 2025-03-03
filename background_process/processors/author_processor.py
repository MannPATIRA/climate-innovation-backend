from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import postgrest
from httpx import RemoteProtocolError
from .base import Processor, ProcessingTask
from background_process.utils.process_log_manager import ProcessLogManager
from common.neo4j_client import Neo4jClient

@dataclass
class Author:
    openalex_id: str
    display_name: str
    orcid: str
    institutions_str: str
    h_index: int
    citations: int
    topics: List[Dict]

class AuthorProcessor(Processor):
    def __init__(self, supabase_client, neo4j_client: Optional[Neo4jClient] = None, max_workers: int = 5):
        super().__init__(ProcessLogManager(supabase_client), None)
        self.supabase = supabase_client
        self.neo4j = neo4j_client
        self.max_workers = max_workers
        self.task_id = self.create_task(ProcessingTask.AUTHOR_PROCESSING)

    def _format_institutions_string(self, institutions: List[Dict]) -> str:
        """Convert institutions list to formatted string"""
        return ", ".join(
            inst['display_name'] for inst in institutions 
            if inst.get('display_name')
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((postgrest.exceptions.APIError, TimeoutError, RemoteProtocolError)),
        reraise=True
    )
    def update_author_in_db(self, author: Author) -> Dict[str, Any]:
        """Update author record in Supabase DB with retry logic"""
        data = {
            "institutions": author.institutions_str,
            "h_index": author.h_index,
            "citations": author.citations,
            "author_processed": True
        }
        response = self.supabase.table('authors') \
            .update(data) \
            .eq('openalex_id', author.openalex_id) \
            .execute()
        return response.data[0]

    def process_single_author(self, data: Dict[str, Any]) -> Tuple[Author, Dict[str, Any]]:
        """Process a single author with error handling"""
        try:
            # Extract openalex_id for logging
            openalex_id = data['id']
            
            # Log that we're starting to process this author
            self.log_progress(openalex_id)
            
            author = Author(
                openalex_id=openalex_id,
                display_name=data['display_name'],
                orcid=data.get('orcid'),
                institutions_str=self._format_institutions_string(data.get('institutions', [])),
                h_index=data.get('h_index', 0),
                citations=data.get('cited_by_count', 0),
                topics=data.get('topics', [])
            )
            
            # Neo4j operations only if client exists
            if self.neo4j:
                try:
                    # Update author record in Neo4j with full properties
                    self.neo4j.merge_author_node(
                        author_id=author.openalex_id,
                        display_name=author.display_name,
                        orcid=author.orcid,
                        h_index=author.h_index,
                        citations=author.citations
                    )
                    
                    # Process topics in Neo4j
                    for topic in author.topics:
                        self.neo4j.merge_author_topic_relationship(
                            author_id=author.openalex_id,
                            topic_id=topic['id'],
                            topic_name=topic['display_name'],
                            paper_count=topic.get('works_count', 0)
                        )
                    
                    # Process institutions in Neo4j
                    for institution in data.get('institutions', []):
                        if institution.get('id'):
                            self.neo4j.merge_author_institution_relationship(
                                author_id=author.openalex_id,
                                institution_id=institution['id'],
                                institution_name=institution.get('display_name', ''),
                                country_code=institution.get('country_code', ''),
                                institution_type=institution.get('type', ''),
                                years=institution.get('years', [])
                            )
                except Exception as e:
                    print(f"Neo4j Error - Failed to process relationships: {type(e).__name__} - {str(e)}")
                    # Continue as we'll still try to update Supabase
            
            # Update author record in database after processing neo4j stuff
            try:
                author_record = self.update_author_in_db(author)
            except Exception as e:
                print(f"Supabase Error - Failed to update author: {type(e).__name__} - {str(e)}")
                return None, None

            # Remove from processing logs after successful processing
            self.remove_from_logs(openalex_id)
            
            return author, author_record
        except Exception as e:
            print(f"Unexpected Error processing author: {type(e).__name__} - {str(e)}")
            return None, None

    def process_batch(self, authors_data: List[Dict[str, Any]]) -> List[Tuple[Author, Dict[str, Any]]]:
        """Process a batch of authors using thread pool executor"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_author = {
                executor.submit(self.process_single_author, author_data): author_data 
                for author_data in authors_data
            }
            
            for future in as_completed(future_to_author):
                author_data = future_to_author[future]
                try:
                    author, record = future.result()
                    if author and record:
                        results.append((author, record))
                    else:
                        print(f"Failed to process author: {author_data.get('display_name', 'Unknown')}")
                except Exception as e:
                    print(f"Exception processing author: {type(e).__name__} - {str(e)}")
                    continue
                
        return results

    def process(self, data: Dict[str, Any]) -> Tuple[Author, Dict[str, Any]]:
        """Process single author"""
        return self.process_single_author(data) 