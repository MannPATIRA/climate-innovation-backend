from dataclasses import dataclass
import os
import boto3
from urllib.parse import quote_plus
from typing import Dict, Any, List, Tuple
from PyPDF2 import PdfReader
from .base import Processor, ProcessingTask


@dataclass
class PDFDocument:
    content: str
    title: str
    content_hash: str


class ReportProcessor(Processor):
    def __init__(self, supabase_client, pinecone_store, chunk_size: int = 500):
        super().__init__(supabase_client, pinecone_store, chunk_size=chunk_size)
        self.task_id = self.create_task(ProcessingTask.REPORT_PROCESSING)
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'eu-north-1')
        )
        self.bucket_name = "climate-innovation-bucket"
        self.region = "eu-north-1"

    def convert_pdf_to_text(self, pdf_path: str) -> PDFDocument:
        """Converts PDF to text using PyPDF2 and extracts title from filename"""
        try:
            # Extract title from path (filename without extension)
            title = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # Convert PDF content
            text_content = ''
            with open(pdf_path, 'rb') as pdf_file:
                reader = PdfReader(pdf_file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + '\n'
            
            # Generate hash and return PDFDocument
            content_hash = self.generate_content_hash(text_content)
            return PDFDocument(
                content=text_content,
                title=title,
                content_hash=content_hash
            )
        except Exception as e:
            raise Exception(f"Error converting PDF to text: {str(e)}")

    def upload_to_s3(self, file_path: str, file_name: str) -> str:
        """Upload file to S3 and return the object URL"""
        try:
            # Upload file
            self.s3_client.upload_file(file_path, self.bucket_name, file_name)
            # Construct and return the object URL
            object_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{quote_plus(file_name)}"
            print(f"Object url: {object_url}")
            return object_url
        except Exception as e:
            raise Exception(f"Error uploading to S3: {str(e)}")

    def add_report_to_db(self, pdf_doc: PDFDocument, object_url: str) -> Dict[str, Any]:
        """Add report to Supabase DB"""
        data = {
            "content": pdf_doc.content,
            "content_hash": pdf_doc.content_hash,
            "report_title": pdf_doc.title,
            "object_url": object_url  # Add the S3 object URL
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

    def chunk_and_embed(self, pdf_doc: PDFDocument, metadata: Dict[str, Any]) -> bool:
        """Add report content to Pinecone with metadata"""
        try:
            chunks = self.chunk_text(pdf_doc.content)
            # Add title to metadata for each chunk
            metadatas = [{
                **metadata,
                "content": chunk,
            } for chunk in chunks]
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

        # Convert PDF to text and get PDFDocument
        pdf_doc = self.convert_pdf_to_text(report_path)
        
        # Check if report exists and get data if it does
        existing_report = self.get_report(pdf_doc.content_hash)
        if not existing_report:
            # Upload to S3 first
            file_name = os.path.basename(report_path)
            object_url = self.upload_to_s3(report_path, file_name)
            
            # Add to Supabase with object_url
            report_record = self.add_report_to_db(pdf_doc, object_url)

            # Add to Pinecone
            report_metadata = {
                "report_id": report_record["id"],
                "content_hash": pdf_doc.content_hash,
                "report_title": pdf_doc.title,
                "object_url": object_url
            }
            self.chunk_and_embed(pdf_doc, report_metadata)
        else:
            print(f"Report {report_path} already processed.")
            report_record = existing_report[0]

        return pdf_doc, report_record 