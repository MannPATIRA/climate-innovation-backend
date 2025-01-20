from abc import ABC, abstractmethod

from supabase import Client
from common.pinecone_store import PineconeStore
from PyPDF2 import PdfReader
import hashlib
from typing import Dict, Any, List
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Processor(ABC):
    def __init__(self, supabase_client: Client, pinecone_store: PineconeStore):
        # Add a ChunkingStrategy class to take in constructor so we can use different chunking strategies LATER
        self.supabase = supabase_client
        self.pinecone_store = pinecone_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def generate_content_hash(self, content: str) -> str:
        """Generate a hash for the content using SHA-256"""
        return hashlib.sha256(content.encode()).hexdigest()

    @abstractmethod
    def process(self, data: Dict[Any, Any]) -> (str, Dict[str, Any]):
        pass


class ReportProcessor(Processor):

    def convert_pdf_to_text(self, pdf_path: str) -> str:
        """Converts PDF to text using PyPDF2"""
        text_content = ''
        try:
            with open(pdf_path, 'rb') as pdf_file:
                reader = PdfReader(pdf_file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + '\n'
            return text_content
        except Exception as e:
            raise Exception(f"Error converting PDF to text: {str(e)}")

    def add_report_to_db(self, content: str, content_hash: str) -> Dict[str, Any]:
        """Add report to Supabase DB"""
        data = {
            "content": content,
            "content_hash": content_hash
        }
        response = self.supabase.table('reports').insert(data).execute()
        return response.data[0]

    def get_report(self, content_hash: str) -> Dict[str, Any]:
        """Get report from Supabase DB by content hash"""
        response = self.supabase.table('reports') \
            .select("*") \
            .eq('content_hash', content_hash) \
            .execute()
        return response.data

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using LangChain's text splitter"""
        return self.text_splitter.split_text(text)

    def chunk_and_embed(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Add report content to Pinecone with metadata"""
        try:
            chunks = self.chunk_text(content)
            # Duplicate metadata for each chunk
            print("number of chunks in report to embed: ", len(chunks))
            metadatas = [metadata for _ in chunks]
            success = self.pinecone_store.add_chunks(
                chunks=chunks,
                metadata=metadatas,
                namespace="reports"
            )
            return success
        except Exception as e:
            raise Exception(f"Error adding to Pinecone: {str(e)}")

    def process(self, data):

        report_path = data['report_path']

        # Convert PDF to text and get content hash
        content = self.convert_pdf_to_text(report_path)
        content_hash = self.generate_content_hash(content)
        print("content hash: ", content_hash)
        # Check if report exists and get data if it does
        existing_report = self.get_report(content_hash)
        if not existing_report:
            # Add to Supabase
            report_record = self.add_report_to_db(content, content_hash)

            # Add to Pinecone
            report_metadata = {
                "report_id": report_record["id"],
                "content_hash": content_hash,
            }
            self.chunk_and_embed(content, report_metadata)
        else:
            print(f"Report {report_path} already processed.")
            report_record = existing_report[0]

        return content, report_record


class PaperProcessor(Processor):
    def process(self, data) -> (str, Dict[str, Any]):
        pass
