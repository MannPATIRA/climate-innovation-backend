from abc import ABC, abstractmethod
from supabase import Client
from common.pinecone_store import PineconeStore
from typing import Dict, Any, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib


class Summarizer(ABC):
    @abstractmethod
    def generate_summary(self, text: str) -> str:
        """Generates a summary for the given text."""
        pass

    def generate_content_hash(self, content: str) -> str:
        """Generate a hash for the content using SHA-256"""
        return hashlib.sha256(content.encode()).hexdigest()


class SummaryProcessor:
    def __init__(self, summarizer: Summarizer, supabase_client: Client, pinecone_store: PineconeStore):
        self.summarizer = summarizer
        self.supabase = supabase_client
        self.pinecone_store = pinecone_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def generate_summary(self, text: str) -> str:
        """Generate summary using the provided summarizer"""
        return self.summarizer.generate_summary(text)

    def add_summary_to_db(self, summary: str, report_id: int, content_hash: str) -> Dict[str, Any]:
        """Add summary to Supabase DB"""
        data = {
            "content": summary,
            "content_hash": content_hash,
            "report_id": report_id
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

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using LangChain's text splitter"""
        return self.text_splitter.split_text(text)

    def chunk_and_embed(self, summary: str, metadata: Dict[str, Any]) -> bool:
        """Add summary to Pinecone with metadata"""
        try:
            chunks = self.chunk_text(summary)
            # Duplicate metadata for each chunk
            metadatas = [metadata for _ in chunks]
            success = self.pinecone_store.add_chunks(
                chunks=chunks,
                metadata=metadatas,
                namespace="summaries"
            )
            return success
        except Exception as e:
            raise Exception(f"Error adding summary to Pinecone: {str(e)}")

    def summarize(self, content: str, report_record: Dict[str, Any], report_path: str):
        summary = self.summarizer.generate_summary(content)
        summary_hash = self.summarizer.generate_content_hash(summary)

        # Only process summary if it hasn't been processed before
        if not self.get_summary(summary_hash):
            # Add to Supabase
            summary_record = self.add_summary_to_db(
                summary,
                report_record["id"],
                summary_hash
            )

            # Add to Pinecone
            summary_metadata = {
                "report_id": report_record["id"],
                "summary_id": summary_record["id"],
                "content_hash": summary_hash,
            }
            self.chunk_and_embed(summary, summary_metadata)

        else:
            print(f"Summary for report {report_path} already exists.")
