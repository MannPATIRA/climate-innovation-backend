from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Dict, Any, List, Tuple, Optional
from .base import Processor, ProcessingTask
from background_process.utils.process_log_manager import ProcessLogManager
from common.neo4j_client import Neo4jClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import postgrest
from httpx import RemoteProtocolError

@dataclass
class Paper:
    abstract: str
    openalex_id: str
    doi: str
    title: str
    publication_date: str


class PaperProcessor(Processor):
    def __init__(self, supabase_client, pinecone_store, neo4j_client: Optional[Neo4jClient] = None,
                 chunk_size: int = 500, max_workers: int = 5):
        super().__init__(ProcessLogManager(supabase_client), pinecone_store, chunk_size=chunk_size)
        self.supabase = supabase_client
        self.neo4j = neo4j_client
        self.max_workers = max_workers
        self.task_id = self.create_task(ProcessingTask.PAPER_PROCESSING)


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((postgrest.exceptions.APIError, TimeoutError, RemoteProtocolError)),
        reraise=True
    )
    def add_paper_to_db(self, paper: Paper) -> Dict[str, Any]:
        """Add paper to Supabase DB with retry logic"""
        data = {
            "openalex_id": paper.openalex_id,
            "doi": paper.doi,
            "title": paper.title,
            "abstract": paper.abstract,
            "publication_date": paper.publication_date
        }
        response = self.supabase.table('papers').insert(data).execute()
        return response.data[0]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((postgrest.exceptions.APIError, TimeoutError, RemoteProtocolError)),
        reraise=True
    )
    def get_paper(self, openalex_id: str) -> Dict[str, Any]:
        """Get paper from Supabase DB with retry logic"""
        response = self.supabase.table('papers') \
            .select("*") \
            .eq('openalex_id', openalex_id) \
            .execute()
        return response.data

    def chunk_and_embed(self, papers: List[Paper], metadata_list: List[Dict[str, Any]]) -> bool:
        """Batch add paper abstracts to Pinecone with retry logic"""
        try:
            all_chunks = []
            all_metadata = []
            
            for paper, metadata in zip(papers, metadata_list):
                chunks = self.text_splitter.split_text(paper.abstract)
                chunk_metadata = [{
                    **metadata,
                    "content": chunk,
                } for chunk in chunks]
                
                all_chunks.extend(chunks)
                all_metadata.extend(chunk_metadata)
            
            success = self.pinecone_store.add_chunks(
                chunks=all_chunks,
                metadata=all_metadata,
                namespace="papers"
            )
            return success
        except Exception as e:
            print(f"Error adding to Pinecone: {str(e)}")
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((postgrest.exceptions.APIError, TimeoutError, RemoteProtocolError)),
        reraise=True
    )
    def add_author_to_db(self, author: Dict[str, Any]) -> Dict[str, Any]:
        """Add author to Supabase DB with retry logic"""
        data = {
            "openalex_id": author['id'],
            "display_name": author['display_name'],
            "orcid": author['orcid']
        }
        response = self.supabase.table('authors').upsert(data).execute()
        return response.data[0]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((postgrest.exceptions.APIError, TimeoutError, RemoteProtocolError)),
        reraise=True
    )
    def add_paper_author_relation(self, paper_id: int, author_id: int, position: str, is_corresponding: bool):
        """Add paper-author relationship to Supabase DB"""
        data = {
            "paper_id": paper_id,
            "author_id": author_id,
            "position": position,
            "is_corresponding": is_corresponding
        }
        self.supabase.table('paper_authors').insert(data).execute()

    def process_single_paper(self, data: Dict[str, Any]) -> Tuple[Paper, Dict[str, Any]]:
        """Process a single paper with error handling"""
        try:
            # Extract openalex_id for logging
            openalex_id = data['metadata']['id']
            
            # Log that we're starting to process this paper
            self.log_progress(openalex_id)
            
            abstract = data['abstract']
            metadata = data['metadata']
            authors = data['authors']
            topics = data['topics']
            paper = Paper(
                abstract=abstract,
                openalex_id=openalex_id,
                doi=metadata.get('doi'),
                title=metadata.get('title'),
                publication_date=metadata.get('publication_date')
            )
            
            # Check if paper exists in Supabase
            try:
                existing_paper = self.get_paper(paper.openalex_id)
            except Exception as e:
                print(f"Supabase Error - Failed to check existing paper: {type(e).__name__} - {str(e)}")
                return None, None

            if not existing_paper:
                # Add to Supabase
                try:
                    paper_record = self.add_paper_to_db(paper)
                except Exception as e:
                    print(f"Supabase Error - Failed to add paper to DB: {type(e).__name__} - {str(e)}")
                    return None, None

                # Neo4j operations only if client exists
                if self.neo4j:
                    try:
                        # Add to Neo4j
                        self.neo4j.merge_paper_node(
                            paper_id=paper.openalex_id,
                            title=paper.title,
                            year=metadata.get('publication_year'),
                            citations=metadata.get('cited_by_count', 0)
                        )

                        # Process authors in Neo4j
                        for author in authors:
                            self.neo4j.merge_author_paper_relationship(
                                author_id=author['id'],
                                paper_id=paper.openalex_id,
                                position=author['position'],
                                is_corresponding=author['is_corresponding'],
                                author_name=author['display_name']
                            )

                        # Process topics in Neo4j
                        for topic in topics:
                            self.neo4j.merge_paper_topic_relationship(
                                paper_id=paper.openalex_id,
                                topic_id=topic['id'],
                                topic_name=topic['display_name'],
                                score=topic.get('score', 0.0)
                            )
                    except Exception as e:
                        print(f"Neo4j Error - Failed to process relationships: {type(e).__name__} - {str(e)}")
                        # Continue processing as Supabase operations were successful

                # Process authors in Supabase
                try:
                    for author in authors:
                        author_record = self.add_author_to_db(author)
                        self.add_paper_author_relation(
                            paper_record['id'],
                            author_record['id'],
                            author['position'],
                            author['is_corresponding']
                        )
                except Exception as e:
                    print(f"Supabase Error - Failed to process author relationships: {type(e).__name__} - {str(e)}")
                    return None, None

                # Add to Pinecone
                try:
                    paper_metadata = {
                        "paper_id": paper_record["id"],
                        "openalex_id": paper.openalex_id,
                    }
                    if paper.doi:  # Only add doi if it exists and is not None
                        paper_metadata["doi"] = paper.doi
                    self.chunk_and_embed([paper], [paper_metadata])
                except Exception as e:
                    print(f"Pinecone Error - Failed to add embeddings: {type(e).__name__} - {str(e)}")
                    # Continue as core data was saved
            else:
                print(f"Paper {paper.title[:30]}... already processed.")
                paper_record = existing_paper[0]

            # Remove from processing logs after successful processing
            self.remove_from_logs(openalex_id)
            
            time.sleep(0.1)
            return paper, paper_record
        except Exception as e:
            print(f"Unexpected Error processing paper: {type(e).__name__} - {str(e)}")
            return None, None

    def process_batch(self, papers_data: List[Dict[str, Any]]) -> List[Tuple[Paper, Dict[str, Any]]]:
        """Process a batch of papers using thread pool executor"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all papers to the thread pool
            future_to_paper = {
                executor.submit(self.process_single_paper, paper_data): paper_data 
                for paper_data in papers_data
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_paper):
                paper_data = future_to_paper[future]
                try:
                    paper, record = future.result()
                    if paper and record:
                        results.append((paper, record))
                    else:
                        print(f"Failed to process paper: {paper_data.get('metadata', {}).get('title', 'Unknown')}")
                except Exception as e:
                    print(f"Exception processing paper: {str(e)}")
                    continue
                
        return results
    
    def process(self, data: Dict[str, Any]) -> Tuple[Paper, Dict[str, Any]]:
        """
        Synchronous process method to satisfy abstract class.
        For single topic processing, use this.
        For batch processing, use process_batch.
        """
        # Run the async process in the event loop
        return asyncio.run(self.process_single_paper(data)) 