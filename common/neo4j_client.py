from typing import Dict, List, Optional
from neo4j import GraphDatabase

class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_constraints_and_indexes()

    def close(self):
        self.driver.close()

    def _ensure_constraints_and_indexes(self):
      """Create necessary constraints and indexes"""
      with self.driver.session() as session:
          # Create unique constraints on IDs
          constraints = [
              "CREATE CONSTRAINT IF NOT EXISTS FOR (w:Work) REQUIRE w.id IS UNIQUE",
              "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE",
              "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
              "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Institution) REQUIRE i.id IS UNIQUE"
          ]
          
          for constraint in constraints:
              session.run(constraint)
              
          # Wait for indexes to be online
          session.run("CALL db.awaitIndexes()")


    def merge_paper_node(self, paper_id: str, title: str, year: Optional[int], citations: int = 0):
        with self.driver.session() as session:
            query = """
            MERGE (w:Work {id: $paper_id})
            ON CREATE SET 
                w.title = $title,
                w.publicationYear = $year,
                w.cited_by_count = $citations
            """
            session.run(query, paper_id=paper_id, title=title, year=year, citations=citations)

    def merge_author_paper_relationship(self, author_id: str, paper_id: str, 
                                      position: str, is_corresponding: bool,
                                      author_name: str):
        with self.driver.session() as session:
            query = """
            MERGE (a:Author {id: $author_id})
            ON CREATE SET a.name = $author_name
            MERGE (w:Work {id: $paper_id})
            MERGE (a)-[r:AUTHORED]->(w)
            ON CREATE SET 
                r.authorPosition = $position,
                r.isCorresponding = $is_corresponding
            """
            session.run(query, 
                       author_id=author_id, 
                       paper_id=paper_id, 
                       position=position, 
                       is_corresponding=is_corresponding,
                       author_name=author_name)

    def merge_paper_topic_relationship(self, paper_id: str, topic_id: str, topic_name: str, score: float):
        with self.driver.session() as session:
            query = """
            MERGE (w:Work {id: $paper_id})
            MERGE (t:Topic {id: $topic_id})
            ON CREATE SET t.name = $topic_name
            MERGE (w)-[r:MENTIONS]->(t)
            ON CREATE SET r.score = $score
            """
            session.run(query, paper_id=paper_id, topic_id=topic_id, topic_name=topic_name, score=score)

    def merge_author_topic_relationship(self, author_id: str, topic_id: str, topic_name: str, paper_count: int):
        with self.driver.session() as session:
            query = """
            MERGE (a:Author {id: $author_id})
            MERGE (t:Topic {id: $topic_id})
            ON CREATE SET t.name = $topic_name
            MERGE (a)-[r:RESEARCHES]->(t)
            ON CREATE SET r.paperCount = $paper_count
            ON MATCH SET r.paperCount = $paper_count
            """
            session.run(query, author_id=author_id, topic_id=topic_id, topic_name=topic_name, paper_count=paper_count)

    def merge_author_institution_relationship(self, author_id: str, institution_id: str, 
                                           institution_name: str, country_code: str,
                                           institution_type: str, years: List[int]):
        with self.driver.session() as session:
            query = """
            MERGE (a:Author {id: $author_id})
            MERGE (i:Institution {id: $institution_id})
            ON CREATE SET 
                i.name = $institution_name,
                i.country_code = $country_code,
                i.type = $institution_type
            MERGE (a)-[r:AFFILIATED_WITH]->(i)
            ON CREATE SET r.years = $years
            ON MATCH SET r.years = $years
            """
            session.run(query, 
                       author_id=author_id, 
                       institution_id=institution_id,
                       institution_name=institution_name, 
                       country_code=country_code,
                       institution_type=institution_type,
                       years=years)

    def merge_author_node(self, author_id: str, display_name: str, orcid: Optional[str], 
                         h_index: int, citations: int, works_count: int):
        """Update author node with full properties during author processing"""
        with self.driver.session() as session:
            query = """
            MERGE (a:Author {id: $author_id})
            SET 
                a.name = $display_name,
                a.orcid = $orcid,
                a.h_index = $h_index,
                a.cited_by_count = $citations,
                a.works_count = $works_count,
                a.last_updated = datetime()
            """
            session.run(query, 
                       author_id=author_id,
                       display_name=display_name,
                       orcid=orcid,
                       h_index=h_index,
                       citations=citations,
                       works_count=works_count) 