import os
from supabase import create_client, Client
from datetime import datetime
import hashlib
from dotenv import load_dotenv
load_dotenv(override=True)

# Initialize Supabase client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def generate_content_hash(content: str) -> str:
    """Generate a hash for the content using SHA-256"""
    return hashlib.sha256(content.encode()).hexdigest()

def main():
    # 1. Delete all existing rows
    print("Deleting all existing rows...")
    supabase.table('reports').delete().neq('id', 0).execute()  # Delete where id is not equal to 0 (deletes all rows)

    # 2. Create sample data
    sample_reports = [
        {
            "content": "This is the first test report",
            "content_hash": generate_content_hash("This is the first test report")
        },
        {
            "content": "Another test report with different content",
            "content_hash": generate_content_hash("Another test report with different content")
        },
        {
            "content": "Third report for testing",
            "content_hash": generate_content_hash("Third report for testing")
        }
    ]

    # 3. Insert sample data
    print("\nInserting new reports...")
    for report in sample_reports:
        result = supabase.table('reports').insert(report).execute()
        print(f"Inserted report with hash: {report['content_hash'][:8]}...")

    # 4. Query and display all reports
    print("\nRetrieving all reports:")
    response = supabase.table('reports').select("*").execute()
    
    for record in response.data:
        print("\nReport:")
        print(f"ID: {record['id']}")
        print(f"Created at: {record['created_at']}")
        print(f"Content: {record['content']}")
        print(f"Content hash: {record['content_hash'][:8]}...")

if __name__ == "__main__":
    main() 