from abc import ABC, abstractmethod
from supabase import Client
from common.pinecone_store import PineconeStore
from typing import Dict, Any, List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
from .base import Processor, ProcessingTask
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ..prompts import SUMMARY_GENERATION_PROMPT, DocumentSummary


class Summarizer(ABC):
    @abstractmethod
    def generate_summary(self, text: str) -> str:
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
    
    def generate_summary(self, text: str) -> str:
        """Generates a summary for the given text using an LLM."""
        try:
            result = self.model_with_structure.invoke(
                self.prompt.format(text=text)
            )
            return result.summary
        except Exception as e:
            raise Exception(f"Error generating summary with LLM: {str(e)}")


class SummaryProcessor(Processor):
    def __init__(self, supabase_client, pinecone_store, summarizer: Summarizer = None, chunk_size: int = 500):
        super().__init__(supabase_client, pinecone_store, chunk_size=chunk_size)
        self.task_id = self.create_task(ProcessingTask.SUMMARY_PROCESSING)
        self.summarizer = summarizer or LLMSummarizer()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0  # No overlap as requested
        )

    def get_report_by_id(self, report_id: int) -> Dict[str, Any]:
        """Get report from Supabase DB by ID"""
        response = self.supabase.table('reports') \
            .select("*") \
            .eq('id', report_id) \
            .execute()
        if not response.data:
            raise Exception(f"Report with ID {report_id} not found")
        return response.data[0]

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
        
        # Process each chunk
        summaries = []
        summary_records = []
        
        for i, chunk in enumerate(chunks):
            # Generate hash for this chunk
            chunk_hash = self.generate_content_hash(chunk)
            
            # Check if this chunk has already been processed
            existing_summary = self.get_summary(chunk_hash)
            
            if not existing_summary:
                # Generate summary for this chunk
                summary = self.summarizer.generate_summary(chunk)
                
                # Add to database
                summary_record = self.add_summary_to_db(
                    original_text=chunk,
                    summary=summary,
                    report_id=report_id,
                    chunk_index=i,
                    content_hash=chunk_hash
                )
                
                summaries.append(summary)
                summary_records.append(summary_record)
            else:
                print(f"Summary for chunk {i} of report {report_id} already exists.")
        
        # Add all summaries to vector database
        metadata_base = {
            "report_id": report_id,
            "report_title": report_title,
        }
        
        self.chunk_and_embed(summaries, chunks, metadata_base)
        
        return summaries, summary_records
