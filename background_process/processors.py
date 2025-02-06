from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from enum import Enum
import boto3
from urllib.parse import quote_plus
from supabase import Client
from common.pinecone_store import PineconeStore
from PyPDF2 import PdfReader
import hashlib
from typing import Dict, Any, List, Tuple, Set
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from .prompts import CLIMATE_RELEVANCE_PROMPT, TopicAssessment



class ProcessingTask(Enum):
    REPORT_PROCESSING = "report_processing"
    PAPER_PROCESSING = "paper_processing" 
    TOPIC_PROCESSING = "topic_processing"

class Processor(ABC):
    def __init__(self, supabase_client: Client, pinecone_store: PineconeStore, chunk_size: int = 500):
        # Add a ChunkingStrategy class to take in constructor so we can use different chunking strategies LATER
        self.supabase = supabase_client
        self.pinecone_store = pinecone_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50
        )
        self.task_id = None

    def generate_content_hash(self, content: str) -> str:
        """Generate a hash for the content using SHA-256"""
        return hashlib.sha256(content.encode()).hexdigest()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def create_task(self, task_type: ProcessingTask) -> int:
        """Create a new task record if it doesn't exist and return its ID"""
        # Check for existing task
        response = self.supabase.table('processor_progress') \
            .select("*") \
            .eq('task', task_type.value) \
            .execute()
        
        if response.data:
            # Return ID of existing task
            return response.data[0]["id"]
        
        # Create new task if none exists
        response = self.supabase.table('processor_progress').insert({
            "task": task_type.value
        }).execute()
        return response.data[0]["id"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def log_progress(self, reference_id: str):
        """Log individual progress for a task"""
        if not self.task_id:
            raise ValueError("No task_id set. Task must be created before logging progress.")
        
        self.supabase.table('process_progress_logs').insert({
            "task_id": self.task_id,
            "reference_id": reference_id
        }).execute()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def remove_from_logs(self, reference_id: str):
        """Remove the entry from processing logs once completed"""
        if not self.task_id:
            raise ValueError("No task_id set. Task must be created before removing from logs.")
        
        self.supabase.table('process_progress_logs') \
            .delete() \
            .eq('task_id', self.task_id) \
            .eq('reference_id', reference_id) \
            .execute()

    @abstractmethod
    def process(self, data: Dict[Any, Any]) -> Tuple[str, Dict[str, Any]]:
        pass


@dataclass
class PDFDocument:
    content: str
    title: str
    content_hash: str

class ReportProcessor(Processor):
    def __init__(self, supabase_client: Client, pinecone_store: PineconeStore, chunk_size: int = 500):
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


@dataclass
class Paper:
    abstract: str
    openalex_id: str
    doi: str
    title: str

class PaperProcessor(Processor):
    def __init__(self, supabase_client: Client, pinecone_store: PineconeStore, chunk_size: int = 500, 
                 max_workers: int = 5):
        super().__init__(supabase_client, pinecone_store, chunk_size=chunk_size)
        self.max_workers = max_workers
        self.task_id = self.create_task(ProcessingTask.PAPER_PROCESSING)


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def add_paper_to_db(self, paper: Paper) -> Dict[str, Any]:
        """Add paper to Supabase DB with retry logic"""
        data = {
            "openalex_id": paper.openalex_id,
            "doi": paper.doi,
            "title": paper.title
        }
        response = self.supabase.table('papers').insert(data).execute()
        return response.data[0]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_paper(self, openalex_id: str) -> Dict[str, Any]:
        """Get paper from Supabase DB with retry logic"""
        response = self.supabase.table('papers') \
            .select("*") \
            .eq('openalex_id', openalex_id) \
            .execute()
        return response.data

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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


class TopicProcessor(Processor):
    def __init__(self, supabase_client, model_name: str = "gpt-4o-mini"):
        super().__init__(supabase_client, None)  # No pinecone store needed
        self.evaluator = ChatOpenAI(
            model=model_name,
            temperature=0.2
        ).with_structured_output(TopicAssessment)
        self.task_id = self.create_task(ProcessingTask.TOPIC_PROCESSING)

    def format_sample_works(self, works: list) -> str:
        """Format sample works for prompt"""
        formatted = ""
        for i, work in enumerate(works, 1):
            formatted += f"\nWork {i}:\n"
            formatted += f"Title: {work['title']}\n"
            formatted += f"Abstract: {work['abstract'][:500]}...\n"  # Truncate long abstracts
        return formatted

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_topic_assessment(self, topic_id: str) -> Dict[str, Any]:
        """Get existing topic assessment from Supabase DB by topic_id"""
        response = self.supabase.table('openalex_topic_assessments') \
            .select("*") \
            .eq('topic_id', topic_id) \
            .execute()
        return response.data

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def save_to_db(self, assessment: TopicAssessment, topic_id: str) -> Dict[str, Any]:
        """Save assessment to Supabase"""
        data = {
            "topic_id": topic_id,
            "is_climate_relevant": assessment.is_climate_relevant,
            "analysis": assessment.analysis
        }
        response = self.supabase.table('openalex_topic_assessments').insert(data).execute()
        return response.data[0]

    def process(self, data: Dict[str, Any]) -> Tuple[TopicAssessment, Dict[str, Any]]:
        """
        Synchronous process method to satisfy abstract class.
        For single topic processing, use this.
        For batch processing, use process_batch.
        """
        # Run the async process in the event loop
        return asyncio.run(self.process_single_topic(data))

    async def process_batch(self, topics: List[Dict[str, Any]]) -> List[Tuple[TopicAssessment, Dict[str, Any]]]:
        """Process a batch of topics concurrently"""
        tasks = []
        for topic in topics:
            task = asyncio.create_task(self.process_single_topic(topic))
            tasks.append(task)
        
        return await asyncio.gather(*tasks)

    async def process_single_topic(self, data: Dict[str, Any]) -> Tuple[TopicAssessment, Dict[str, Any]]:
        """Process a single topic asynchronously"""
        # Check if topic has already been processed
        existing_assessment = self.get_topic_assessment(data['topic_id'])
        if existing_assessment:
            print(f"Topic {data['topic_name']} already processed.")
            return None, existing_assessment[0]

        # Format the prompt
        sample_works_text = self.format_sample_works(data['sample_works'])
        chain = CLIMATE_RELEVANCE_PROMPT | self.evaluator
        
        # Get structured assessment from LLM
        assessment = await chain.ainvoke({
            "topic_name": data['topic_name'],
            "topic_description": data['topic_description'],
            "sample_works": sample_works_text
        })

        # Store in database
        record = self.save_to_db(assessment, data['topic_id'])
        
        return assessment, record
