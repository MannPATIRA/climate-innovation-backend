from abc import ABC, abstractmethod
from supabase import Client
from common.pinecone_store import PineconeStore
from typing import Dict, Any, List, Tuple, AsyncIterator
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
from .base import Processor, ProcessingTask
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ..prompts import SUMMARY_GENERATION_PROMPT, DocumentSummary
import asyncio
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict, Annotated
import operator
from background_process.utils.process_log_manager import ProcessLogManager
from tenacity import retry, stop_after_attempt, wait_exponential

# Define the state for our LangGraph
class SummaryState(TypedDict):
    chunks: List[str]
    summaries: Annotated[List[str], operator.add]
    successful_indices: Annotated[List[int], operator.add]

class Summarizer(ABC):
    @abstractmethod
    async def generate_summary(self, text: str) -> str:
        """Generates a summary for the given text."""
        pass

    def generate_content_hash(self, content: str) -> str:
        """Generate a hash for the content using SHA-256"""
        return hashlib.sha256(content.encode()).hexdigest()


class LLMSummarizer(Summarizer):
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SUMMARY_GENERATION_PROMPT),
            ("human", "{text}")
        ])
        self.model_with_structure = self.llm.with_structured_output(DocumentSummary)
    

    async def generate_summary(self, text: str) -> str:
        """Asynchronously generates a summary for the given text using an LLM."""
        try:
            result = await self.model_with_structure.ainvoke(
                self.prompt.format(text=text)
            )
            return result.summary
        except Exception as e:
            raise Exception(f"Error generating summary with LLM: {str(e)}")


class SummaryProcessor(Processor):
    def __init__(self, supabase_client, process_log_manager: ProcessLogManager, pinecone_store: PineconeStore, summarizer: Summarizer = None, chunk_size: int = 500):
        super().__init__(process_log_manager=process_log_manager, pinecone_store=pinecone_store, chunk_size=chunk_size)
        self.task_id = self.create_task(ProcessingTask.SUMMARY_PROCESSING)
        self.summarizer = summarizer or LLMSummarizer()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0
        )
        # Create the LangGraph for summarization
        self.summary_graph = self._create_summary_graph()
        self.supabase = supabase_client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=15, max=60),
        reraise=True
    )
    async def _generate_summary_with_semaphore(self, chunk: str, semaphore: asyncio.Semaphore) -> str:
        """Generate a summary for a chunk with semaphore control and retry logic"""
        try:
            async with semaphore:
                return await self.summarizer.generate_summary(chunk)
        except Exception as e:
            print(f"Summary generation failed: {str(e)}")
            raise

    async def summarize_chunks(self, state: SummaryState):
        """Process chunks with semaphore control"""
        # Create semaphore in the current event loop
        semaphore = asyncio.Semaphore(10)
        
        tasks = [self._generate_summary_with_semaphore(chunk, semaphore) for chunk in state["chunks"]]
        summaries = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed summaries and track successful indices
        successful_summaries = []
        successful_indices = []
        
        for i, summary in enumerate(summaries):
            if not isinstance(summary, Exception):
                successful_summaries.append(summary)
                successful_indices.append(i)
        
        return {
            "summaries": successful_summaries,
            "successful_indices": successful_indices
        }

    def _create_summary_graph(self) -> StateGraph:
        """Create a LangGraph for summarizing chunks in parallel."""
        # Build the graph
        builder = StateGraph(SummaryState)
        builder.add_node("summarize", self.summarize_chunks)
        builder.add_edge(START, "summarize")
        
        return builder.compile()

    def get_report_by_id(self, report_id: int) -> Dict[str, Any]:
        """Get report from Supabase DB by ID"""
        response = self.supabase.table('reports') \
            .select("*") \
            .eq('id', report_id) \
            .execute()
        if not response.data:
            raise Exception(f"Report with ID {report_id} not found")
        return response.data[0]

    def add_summaries_to_db(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add multiple summaries to Supabase DB in a single batch operation"""
        response = self.supabase.table('summaries').insert(items).execute()
        return response.data
    
    def add_summary_to_db(self, original_text: str, summary: str, report_id: int, 
                          chunk_index: int, content_hash: str) -> Dict[str, Any]:
        """Add summary to Supabase DB"""
        data = {
            "original_content": original_text,
            "summary_content": summary,
            "content_hash": content_hash,
            "report_id": report_id,
            "chunk_index": chunk_index
        }
        response = self.supabase.table('summaries').insert(data).execute()
        return response.data[0]

    def get_summary(self, content_hash: str) -> Dict[str, Any]:
        """Get summary from Supabase DB by content hash"""
        response = self.supabase.table('summaries') \
            .select("*") \
            .eq('content_hash', content_hash) \
            .execute()
        return response.data

    def chunk_and_embed(self, summaries: List[str], original_chunks: List[str], 
                        metadata_base: Dict[str, Any]) -> bool:
        """Add summaries to Pinecone with original content in metadata"""
        try:
            # Create metadata for each summary chunk
            metadatas = []
            for i, (summary, original) in enumerate(zip(summaries, original_chunks)):
                metadatas.append({
                    **metadata_base,
                    "content": original,  # Original text in metadata
                    "summary": summary, # The summary that was generated
                    "chunk_index": i
                })
            
            # Add summaries as the actual chunks to embed
            success = self.pinecone_store.add_chunks(
                chunks=summaries,  # Summaries are what gets embedded
                metadata=metadatas,  # Original content in metadata
                namespace="report_summaries"
            )
            return success
        except Exception as e:
            raise Exception(f"Error adding summaries to Pinecone: {str(e)}")

    def process(self, data: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process a report by ID, chunk it, summarize each chunk, and store in DB and vector DB
        
        Args:
            data: Dictionary containing report_id
            
        Returns:
            Tuple of (summaries, summary_records)
        """
        report_id = data['report_id']
        
        # Get report from database
        report = self.get_report_by_id(report_id)
        report_content = report['content']
        report_title = report['report_title']
        
        # Split report into chunks
        chunks = self.text_splitter.split_text(report_content)
        print(f"Split report into {len(chunks)} chunks")
        
        # Filter out chunks that already have summaries
        chunks_to_process = []
        chunk_indices = []
        chunk_hashes = []  # Store hashes to avoid recomputing them
        
        for i, chunk in enumerate(chunks):
            # Generate hash for this chunk
            chunk_hash = self.generate_content_hash(chunk)
            
            # Check if this chunk has already been processed
            existing_summary = self.get_summary(chunk_hash)
            
            if not existing_summary:
                chunks_to_process.append(chunk)
                chunk_indices.append(i)
                chunk_hashes.append(chunk_hash)  # Store the hash
            else:
                print(f"Summary for chunk {i} of report {report_id} already exists.")
        
        # If there are chunks to process, use LangGraph to generate summaries in parallel
        summaries = []
        summary_records = []
        
        if chunks_to_process:
            # Run the graph to generate summaries
            result = asyncio.run(self.summary_graph.ainvoke({
                "chunks": chunks_to_process,
                "summaries": [],
                "successful_indices": []
            }))
            
            # Get the results
            generated_summaries = result["summaries"]
            successful_indices = result["successful_indices"]
            
            # Prepare batch data for database insertion
            print(f"Generated {len(generated_summaries)} summaries successfully")
            batch_data = []
            for summary, idx in zip(generated_summaries, successful_indices):
                batch_data.append({
                    "original_content": chunks_to_process[idx],
                    "summary_content": summary,
                    "content_hash": chunk_hashes[idx],
                    "report_id": report_id,
                    "chunk_index": chunk_indices[idx]
                })
                summaries.append(summary)
            
            # Add all summaries to database in a single batch operation
            if batch_data:
                summary_records = self.add_summaries_to_db(batch_data)
                
                # Add to vector database using the correct chunks
                metadata_base = {
                    "report_id": report_id,
                    "report_title": report_title,
                }
                
                successful_original_chunks = [chunks_to_process[idx] for idx in successful_indices]
                self.chunk_and_embed(generated_summaries, successful_original_chunks, metadata_base)
        
        return summaries, summary_records
