from common.neo4j_client import Neo4jClient
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import uuid

load_dotenv(override=True)

# Initialize the client
client = Neo4jClient(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD"),
    ssh_host=os.getenv("REMOTE_SERVER_HOST"),
    ssh_user=os.getenv("REMOTE_SERVER_USER"),
    ssh_password=os.getenv("REMOTE_SERVER_PASSWORD")
)

# Test connection
print("Testing Neo4j connection...")
with client.driver.session() as session:
    print("Connection successful!")

# Function to perform a single Neo4j operation
def perform_neo4j_operation(operation_id):
    try:
        # Generate unique IDs for this operation
        paper_id = f"paper_{operation_id}_{uuid.uuid4().hex[:8]}"
        author_id = f"author_{operation_id}_{uuid.uuid4().hex[:8]}"
        topic_id = f"topic_{operation_id}_{uuid.uuid4().hex[:8]}"
        institution_id = f"inst_{operation_id}_{uuid.uuid4().hex[:8]}"
        
        # Add a paper
        client.merge_paper_node(
            paper_id=paper_id,
            title=f"Test Paper {operation_id}",
            year=2024,
            citations=random.randint(0, 100)
        )
        
        # Add an author
        client.merge_author_node(
            author_id=author_id,
            display_name=f"Author {operation_id}",
            orcid=f"0000-0001-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}",
            h_index=random.randint(1, 50),
            citations=random.randint(10, 1000),
            works_count=random.randint(5, 100)
        )
        
        # Link author to paper
        client.merge_author_paper_relationship(
            author_id=author_id,
            paper_id=paper_id,
            position="first" if random.random() > 0.5 else "middle",
            is_corresponding=random.random() > 0.7,
            author_name=f"Author {operation_id}"
        )
        
        # Add a topic
        client.merge_paper_topic_relationship(
            paper_id=paper_id,
            topic_id=topic_id,
            topic_name=f"Topic {operation_id}",
            score=random.random()
        )
        
        # Add institution
        client.merge_author_institution_relationship(
            author_id=author_id,
            institution_id=institution_id,
            institution_name=f"Institution {operation_id}",
            country_code="US",
            institution_type="University",
            years=[2023, 2024]
        )
        
        # Verify the data was added correctly
        with client.driver.session() as session:
            # Check paper
            paper_result = session.run(
                "MATCH (w:Work {id: $paper_id}) RETURN w.title",
                paper_id=paper_id
            )
            paper_title = paper_result.single()[0]
            
            # Check author
            author_result = session.run(
                "MATCH (a:Author {id: $author_id}) RETURN a.name",
                author_id=author_id
            )
            author_name = author_result.single()[0]
            
            # Check relationship
            rel_result = session.run(
                """
                MATCH (a:Author {id: $author_id})-[r:AUTHORED]->(w:Work {id: $paper_id})
                RETURN r.authorPosition
                """,
                author_id=author_id, paper_id=paper_id
            )
            position = rel_result.single()[0]
        
        return {
            "operation_id": operation_id,
            "paper_id": paper_id,
            "paper_title": paper_title,
            "author_id": author_id,
            "author_name": author_name,
            "position": position,
            "status": "success"
        }
    except Exception as e:
        # Print detailed exception information
        import traceback
        print(f"\nDetailed exception for operation {operation_id}:")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception module: {type(e).__module__}")
        print(f"Exception message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        
        return {
            "operation_id": operation_id,
            "status": "failed",
            "error": f"{type(e).__module__}.{type(e).__name__}: {str(e)}"
        }

# Main execution
def main():
    num_operations = 100
    max_workers = 10
    
    print(f"Starting {num_operations} Neo4j operations with {max_workers} workers...")
    start_time = time.time()
    
    successful_operations = 0
    failed_operations = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all operations to the thread pool
        future_to_operation = {
            executor.submit(perform_neo4j_operation, i): i 
            for i in range(num_operations)
        }
        
        # Process results as they complete
        for future in as_completed(future_to_operation):
            operation_id = future_to_operation[future]
            try:
                result = future.result()
                if result["status"] == "success":
                    successful_operations += 1
                    print(f"Operation {operation_id} succeeded:")
                    print(f"  Paper: {result['paper_title']} ({result['paper_id']})")
                    print(f"  Author: {result['author_name']} ({result['author_id']})")
                    print(f"  Position: {result['position']}")
                else:
                    failed_operations += 1
                    print(f"Operation {operation_id} failed: {result['error']}")
            except Exception as e:
                failed_operations += 1
                print(f"Operation {operation_id} raised exception: {str(e)}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\nSummary:")
    print(f"Total operations: {num_operations}")
    print(f"Successful: {successful_operations}")
    print(f"Failed: {failed_operations}")
    print(f"Total time: {duration:.2f} seconds")
    print(f"Average time per operation: {duration/num_operations:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
    finally:
        # Always close the client
        client.close()
        print("\nNeo4j client closed.") 