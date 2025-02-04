import os
from dotenv import load_dotenv
from supabase import create_client
from background_process.fetchers import TopicFetcher
from background_process.processors import TopicProcessor
import json

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    supabase = create_client(supabase_url, supabase_key)
    
    # Initialize processor
    processor = TopicProcessor(supabase_client=supabase)
    
    # Initialize fetcher
    fetcher = TopicFetcher()
    
    topic_generator = fetcher.fetch()
    print("first topic processed: ", record["id"])
    for topic in topic_generator:
        assessment, record = processor.process(topic)
        print("processed ", record["id"])
    print(assessment)
    # Print results in a readable format
    print("\n=== Topic Analysis Results ===")
    print("\nAssessment:")
    print(f"Climate Relevant: {assessment.is_climate_relevant}")
    print(f"\nAnalysis:\n{assessment.analysis}")
    
    # Save results to JSON file for reference
    output = {
        "assessment": {
            "is_climate_relevant": assessment.is_climate_relevant,
            "analysis": assessment.analysis
        }
    }
    
    print("\nResults have been saved to topic_assessment_test.json")

if __name__ == "__main__":
    main() 