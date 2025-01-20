from supabase import Client
from common.pinecone_store import PineconeStore
from PyPDF2 import PdfReader
import hashlib
from typing import Dict, Any, List
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ReportProcessor:
    def __init__(self, supabase_client: Client, pinecone_store: PineconeStore):
        self.supabase = supabase_client
        self.pinecone_store = pinecone_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

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

    def generate_content_hash(self, content: str) -> str:
        """Generate a hash for the content using SHA-256"""
        return hashlib.sha256(content.encode()).hexdigest()

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
