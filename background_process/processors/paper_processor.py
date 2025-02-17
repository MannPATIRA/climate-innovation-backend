from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Dict, Any, List, Tuple
from .base import Processor, ProcessingTask


@dataclass
class Paper:
    abstract: str
    openalex_id: str
    doi: str
    title: str


class PaperProcessor(Processor):
    def __init__(self, supabase_client, pinecone_store, chunk_size: int = 500, 
                 max_workers: int = 5):
        super().__init__(supabase_client, pinecone_store, chunk_size=chunk_size)
        self.max_workers = max_workers
        self.task_id = self.create_task(ProcessingTask.PAPER_PROCESSING)


    def add_paper_to_db(self, paper: Paper) -> Dict[str, Any]:
        """Add paper to Supabase DB with retry logic"""
        data = {
            "openalex_id": paper.openalex_id,
            "doi": paper.doi,
            "title": paper.title
        }
        response = self.supabase.table('papers').insert(data).execute()
        return response.data[0]

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

    def process_single_paper(self, data: Dict[str, Any]) -> Tuple[Paper, Dict[str, Any]]:
        """Process a single paper with error handling"""
        try:
            # Extract openalex_id for logging
            openalex_id = data['metadata']['id']
            
            # Log that we're starting to process this paper
            self.log_progress(openalex_id)
            
            abstract = data['abstract']
            metadata = data['metadata']
            
            paper = Paper(
                abstract=abstract,
                openalex_id=openalex_id,
                doi=metadata.get('doi'),
                title=metadata['title'],
            )
            
            # Check if paper exists
            existing_paper = self.get_paper(paper.openalex_id)
            if not existing_paper:
                # Add to Supabase
                paper_record = self.add_paper_to_db(paper)

                # Add to Pinecone
                paper_metadata = {
                    "paper_id": paper_record["id"],
                    "openalex_id": paper.openalex_id,
                }
                if paper.doi:  # Only add doi if it exists and is not None
                    paper_metadata["doi"] = paper.doi
                self.chunk_and_embed([paper], [paper_metadata])
            else:
                print(f"Paper {paper.title[:30]}... already processed.")
                paper_record = existing_paper[0]

            # Remove from processing logs after successful processing
            self.remove_from_logs(openalex_id)
            
            time.sleep(0.1)
            return paper, paper_record
        except Exception as e:
            print(f"Error processing paper: {str(e)}")
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