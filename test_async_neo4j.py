import asyncio
import os
import random
import time
import uuid
from dotenv import load_dotenv

from common.async_neo4j_client import AsyncNeo4jClient

load_dotenv(override=True)

async def perform_neo4j_operation(client: AsyncNeo4jClient, operation_id):
    try:
        # Generate unique IDs for this operation
        paper_id = f"paper_{operation_id}_{uuid.uuid4().hex[:8]}"
        author_id = f"author_{operation_id}_{uuid.uuid4().hex[:8]}"
        topic_id = f"topic_{operation_id}_{uuid.uuid4().hex[:8]}"
        institution_id = f"inst_{operation_id}_{uuid.uuid4().hex[:8]}"
        
        # Add a paper
        await client.merge_paper_node(
            paper_id=paper_id,
            title=f"Test Paper {operation_id}",
            year=2024,
            citations=random.randint(0, 100)
        )
        
        # Add an author
        await client.merge_author_node(
            author_id=author_id,
            display_name=f"Author {operation_id}",
            orcid=f"0000-0001-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}",
            h_index=random.randint(1, 50),
            citations=random.randint(10, 1000),
            works_count=random.randint(5, 100)
        )
        
        # Link author to paper
        await client.merge_author_paper_relationship(
            author_id=author_id,
            paper_id=paper_id,
            position="first" if random.random() > 0.5 else "middle",
            is_corresponding=random.random() > 0.7,
            author_name=f"Author {operation_id}"
        )
        
        # Add a topic
        await client.merge_paper_topic_relationship(
            paper_id=paper_id,
            topic_id=topic_id,
            topic_name=f"Topic {operation_id}",
            score=random.random()
        )
        
        # Add institution
        await client.merge_author_institution_relationship(
            author_id=author_id,
            institution_id=institution_id,
            institution_name=f"Institution {operation_id}",
            country_code="US",
            institution_type="University",
            years=[2023, 2024]
        )
        
        # Verify the data was added correctly
        async with client.driver.session() as session:
            # Check paper
            paper_result = await session.run(
                "MATCH (w:Work {id: $paper_id}) RETURN w.title",
                paper_id=paper_id
            )
            paper_record = await paper_result.single()
            paper_title = paper_record[0]
            
            # Check author
            author_result = await session.run(
                "MATCH (a:Author {id: $author_id}) RETURN a.name",
                author_id=author_id
            )
            author_record = await author_result.single()
            author_name = author_record[0]
            
            # Check relationship
            rel_result = await session.run(
                """
                MATCH (a:Author {id: $author_id})-[r:AUTHORED]->(w:Work {id: $paper_id})
                RETURN r.authorPosition
                """,
                author_id=author_id, paper_id=paper_id
            )
            rel_record = await rel_result.single()
            position = rel_record[0]
        
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

async def main():
    # Initialize the client
    client: AsyncNeo4jClient = AsyncNeo4jClient(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
        ssh_host=os.getenv("REMOTE_SERVER_HOST"),
        ssh_user=os.getenv("REMOTE_SERVER_USER"),
        ssh_password=os.getenv("REMOTE_SERVER_PASSWORD")
    )
    await client.initialize()
    
    try:
        # Test connection
        print("Testing Neo4j connection...")
        async with client.driver.session() as session:
            await session.run("RETURN 1")
            print("Connection successful!")
        
        num_operations = 100
        concurrency = 10
        
        print(f"Starting {num_operations} Neo4j operations with concurrency {concurrency}...")
        start_time = time.time()
        
        # Create tasks in batches to control concurrency
        tasks = []
        results = []
        
        for i in range(0, num_operations, concurrency):
            batch = range(i, min(i + concurrency, num_operations))
            batch_tasks = [perform_neo4j_operation(client, j) for j in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
        
        successful_operations = 0
        failed_operations = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_operations += 1
                print(f"Operation {i} raised exception: {str(result)}")
            elif result["status"] == "success":
                successful_operations += 1
                print(f"Operation {result['operation_id']} succeeded:")
                print(f"  Paper: {result['paper_title']} ({result['paper_id']})")
                print(f"  Author: {result['author_name']} ({result['author_id']})")
                print(f"  Position: {result['position']}")
            else:
                failed_operations += 1
                print(f"Operation {result['operation_id']} failed: {result['error']}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\nSummary:")
        print(f"Total operations: {num_operations}")
        print(f"Successful: {successful_operations}")
        print(f"Failed: {failed_operations}")
        print(f"Total time: {duration:.2f} seconds")
        print(f"Average time per operation: {duration/num_operations:.2f} seconds")
    
    finally:
        # Always close the client
        await client.close()
        print("\nNeo4j client closed.")

if __name__ == "__main__":
    asyncio.run(main()) 