from common.neo4j_client import Neo4jClient
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# Initialize the client (replace with your credentials)
client = Neo4jClient(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

# Add a paper
print("Adding a paper...")
client.merge_paper_node(
    paper_id="paper123",
    title="Understanding Graph Databases",
    year=2024,
    citations=42
)

# Add an author
print("Adding an author...")
client.merge_author_node(
    author_id="author123",
    display_name="Alice Smith",
    orcid="0000-0001-2345-6789",
    h_index=10,
    citations=500
)

# Link author to paper
print("Linking author to paper...")
client.merge_author_paper_relationship(
    author_id="author123",
    paper_id="paper123",
    position="first",
    is_corresponding=True
)

# Add a topic
print("Adding paper-topic relationship...")
client.merge_paper_topic_relationship(
    paper_id="paper123",
    topic_id="topic123",
    score=0.9
)

# Add institution
print("Adding author-institution relationship...")
client.merge_author_institution_relationship(
    author_id="author123",
    institution_id="inst123",
    institution_name="Graph University",
    years=[2023, 2024]
)

# Query and print results
print("\nQuerying the database...")
with client.driver.session() as session:
    # Query paper details
    print("\nPaper details:")
    result = session.run("""
        MATCH (w:Work {id: 'paper123'})
        RETURN w.title, w.publicationYear, w.citationCount
    """)
    for record in result:
        print(f"Title: {record['w.title']}")
        print(f"Year: {record['w.publicationYear']}")
        print(f"Citations: {record['w.citationCount']}")

    # Query author and their papers
    print("\nAuthor and their papers:")
    result = session.run("""
        MATCH (a:Author {id: 'author123'})-[r:AUTHORED]->(w:Work)
        RETURN a.name, w.title, r.authorPosition
    """)
    for record in result:
        print(f"Author: {record['a.name']}")
        print(f"Wrote: {record['w.title']}")
        print(f"Position: {record['r.authorPosition']}")

    # Query author's institution
    print("\nAuthor's institution:")
    result = session.run("""
        MATCH (a:Author {id: 'author123'})-[r:AFFILIATED_WITH]->(i:Institution)
        RETURN i.name, r.years
    """)
    for record in result:
        print(f"Institution: {record['i.name']}")
        print(f"Years: {record['r.years']}")

client.close()
print("\nDone!") 