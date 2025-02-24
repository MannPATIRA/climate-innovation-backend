import os
import shutil
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from urllib.parse import urljoin, urlparse
from typing import Generator, List, Set, Optional
from abc import ABC
from .base import Fetcher
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from ..prompts import SEARCH_QUERY_SYSTEM_PROMPT, SearchQueries
from common.pinecone_store import PineconeStore
from collections import deque
from dataclasses import dataclass


class ReportFetcher(Fetcher, ABC):
    pass


class LocalPDFFetcher(ReportFetcher):
    def __init__(self, directory: str):
        self.directory = directory
        self.temp_directory = os.path.join(os.path.dirname(directory), "processing_temp")
        if not os.path.exists(self.temp_directory):
            os.makedirs(self.temp_directory)

    def fetch(self) -> Generator[str, None, None]:
        for filename in os.listdir(self.directory):
            if filename.lower().endswith('.pdf'):
                print("considering file: ", filename)
                # Create temp copy
                source_path = os.path.join(self.directory, filename)
                temp_path = os.path.join(self.temp_directory, filename)
                shutil.copy2(source_path, temp_path)
                yield temp_path
                
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        """Cleanup temporary directory"""
        if os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory)


    def __del__(self):
        """Cleanup temporary directory when the fetcher is destroyed"""
        self.cleanup()


@dataclass
class URLNode:
    url: str
    depth: int
    parent_url: Optional[str] = None

class WebScrapingReportFetcher(ReportFetcher):
    def __init__(self, llm_model: str = "gpt-4o-mini", max_depth: int = 2, use_cache: bool = True):
        self.search_operators = [
            "site:gov.uk",
            "site:theccc.org.uk",
            "site:metoffice.gov.uk",
            "filetype:pdf"
        ]
        self.llm = ChatOpenAI(model=llm_model)
        self.max_depth = max_depth
        self.use_cache = use_cache
        if use_cache:
            self.store = PineconeStore(index_name="climate-index")
        self.temp_directory = "temp_pdfs"
        if not os.path.exists(self.temp_directory):
            os.makedirs(self.temp_directory)
        
        # Google Custom Search API configuration
        self.search_api_key = os.getenv('GOOGLE_API_KEY')
        print("search api key")
        print(self.search_api_key)
        self.search_engine_id = os.getenv('SEARCH_ENGINE_ID')
        print("search engine id")
        print(self.search_engine_id)
        self.search_url = 'https://www.googleapis.com/customsearch/v1'

    def is_pdf_link(self, url: str) -> bool:
        """Check if URL points to a PDF file"""
        return url.lower().endswith('.pdf')
    
    def download_pdf(self, url: str) -> str:
        """Download PDF and return local path"""
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                filename = os.path.join(self.temp_directory, url.split('/')[-1])
                with open(filename, 'wb') as f:
                    f.write(response.content)
                return filename
        except Exception as e:
            print(f"Error downloading PDF {url}: {e}")
        return None
    
    def get_pdf_links(self, url: str, visited: Set[str]) -> List[str]:
        """Extract all links (both PDF and non-PDF) from webpage"""
        links = []
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a'):
                href = link.get('href')
                if not href:
                    continue
                    
                full_url = urljoin(url, href)
                links.append(full_url)
                
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            
        return links
    
    def generate_search_queries(self, input_text: str) -> List[str]:
        # Create messages list
        messages = [
            ("system", SEARCH_QUERY_SYSTEM_PROMPT),
            ("human", input_text)
        ]
        
        model_with_structure = self.llm.with_structured_output(SearchQueries)
        unique_queries = []
        
        while len(unique_queries) < 10:
            # Generate queries
            result = model_with_structure.invoke(messages)
            print("results from model saerch qeury gen")
            print(result)
            # Check each query for similarity
            for query in result.queries:
                # Skip if we already have enough queries
                if len(unique_queries) >= 10:
                    break
                    
                # Check similarity with existing queries in vector DB if caching enabled
                is_unique = True
                if self.use_cache:
                    similar_results = self.store.query_chunk(
                        query, 
                        top_k=1, 
                        namespace="search_queries"
                    )
                    
                    for result in similar_results:
                        if result.score > 0.9:
                            is_unique = False
                            break
                
                if is_unique:
                    unique_queries.append(query)
                    # Store in vector DB if caching enabled
                    if self.use_cache:
                        self.store.add_chunks(
                            [query], 
                            [{"content": query}], 
                            namespace="search_queries"
                        )
            
            # If we need more queries, ask LLM again
            if len(unique_queries) < 10:
                current_queries = "\n".join(unique_queries)
                messages.extend([
                    ("assistant", f"Here are the queries I've generated so far:\n{current_queries}"),
                    ("human", "Please generate more different queries that cover other aspects of climate change")
                ])
            else:
                break
        
        # Expand unique queries with search operators
        expanded_queries = []
        for query in unique_queries:
            for operator in self.search_operators:
                expanded_queries.append(f"{query} {operator}")
        
        return expanded_queries

    def get_search_results(self, query: str, num_results: int = 10) -> List[str]:
        """Get search results using Google Custom Search API"""
        params = {
            'q': query,
            'key': self.search_api_key,
            'cx': self.search_engine_id
        }
        
        try:
            response = requests.get(self.search_url, params=params)
            results = response.json()
            if 'items' in results:
                return [item['link'] for item in results['items']]
            return []
            
        except Exception as e:
            print(f"Error in Google Search API: {e}")
            return []

    def fetch(self) -> Generator[str, None, None]:
        """Fetch PDFs using BFS with configurable depth"""
        visited_urls = set()
        queries = self.generate_search_queries("Generate the climate change reports search queries")
        
        for query in queries[:2]:
            try:
                # Get initial search results using Google Custom Search API
                search_results = self.get_search_results(query)
                
                # Initialize BFS queue with search results
                queue = deque(
                    URLNode(url=url, depth=0) 
                    for url in search_results
                )
                
                # BFS traversal
                while queue:
                    node = queue.popleft()
                    
                    # Skip if URL already visited or depth exceeded
                    if node.url in visited_urls or node.depth > self.max_depth:
                        continue
                        
                    visited_urls.add(node.url)
                    print(f"Processing URL (depth {node.depth}): {node.url}")
                    
                    try:
                        # Handle PDF URLs
                        if self.is_pdf_link(node.url):
                            pdf_path = self.download_pdf(node.url)
                            if pdf_path:
                                yield pdf_path
                        # Handle web pages
                        else:
                            # Get all links from the page
                            for link in self.get_pdf_links(node.url, visited_urls):
                                # Add unvisited links to queue with incremented depth
                                if link not in visited_urls:
                                    queue.append(URLNode(
                                        url=link,
                                        depth=node.depth + 1,
                                        parent_url=node.url
                                    ))
                                    
                    except Exception as e:
                        print(f"Error processing URL {node.url}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error processing query {query}: {e}")
                continue
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self):
        """Clean up temporary directory"""
        if os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory)
    
    def __del__(self):
        self.cleanup()