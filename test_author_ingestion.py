import os
from dotenv import load_dotenv
from supabase import create_client
from background_process.fetchers import PyAlexAuthorFetcher
from background_process.processors import AuthorProcessor
from background_process.orchestrators import AuthorOrchestrator
from common.neo4j_client import Neo4jClient

def test_author_ingestion():
    # Load environment variables
    load_dotenv(override=True)
    
    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    supabase = create_client(supabase_url, supabase_key)
    
    # Initialize Neo4j client
    neo4j_client = Neo4jClient(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    
    # Initialize components
    fetcher = PyAlexAuthorFetcher(supabase)
    processor = AuthorProcessor(
        supabase,
        neo4j_client,
        max_workers=5  # Lower number of workers due to rate limiting
    )
    
    # Create and run orchestrator
    orchestrator = AuthorOrchestrator(
        fetcher=fetcher,
        processor=processor,
        batch_size=100  # Smaller batch size due to rate limiting
    )
    
    # Run the orchestrator (no country filter needed as we're processing existing authors)
    orchestrator.run()

if __name__ == "__main__":
    test_author_ingestion() 