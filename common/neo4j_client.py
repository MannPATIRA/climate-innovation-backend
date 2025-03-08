from typing import Dict, List, Optional, Callable
from neo4j import GraphDatabase
import paramiko
import time
import os
import threading
from functools import wraps
from neo4j.exceptions import ServiceUnavailable, DatabaseUnavailable

def neo4j_operation_with_retry(max_retries=3, retry_delay=60):
    """
    Decorator for Neo4j operations that handles ServiceUnavailable errors
    by attempting to restart the service and retry the operation.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self: 'Neo4jClient', *args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(self, *args, **kwargs)
                except (ServiceUnavailable, DatabaseUnavailable) as e:
                    retries += 1
                    print(f"Neo4j service unavailable: {str(e)}")
                    print(f"Retry attempt {retries}/{max_retries}")
                    
                    if retries < max_retries:
                        # Try to restart Neo4j service
                        self.restart_neo4j_service()
                        # Wait before retrying
                        time.sleep(retry_delay)
                    else:
                        print("Max retries reached. Operation failed.")
                        raise
            return None
        return wrapper
    return decorator

class Neo4jClient:
    # Class-level lock for restart operations
    _restart_lock = threading.Lock()
    
    def __init__(self, uri: str, user: str, password: str, 
                 ssh_host: Optional[str] = None, 
                 ssh_user: Optional[str] = None, 
                 ssh_password: Optional[str] = None):
        self.uri = uri
        self.user = user
        self.password = password
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ssh_password = ssh_password
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._ensure_constraints_and_indexes()

    def restart_neo4j_service(self) -> bool:
        """
        Restart Neo4j service via SSH with thread safety.
        Only one thread will perform the actual restart while others wait.
        """
        # Try to acquire the lock - if we can't, another thread is already restarting
        if not Neo4jClient._restart_lock.acquire(blocking=False):
            print("Another thread is already restarting Neo4j, waiting...")
            # Wait for the lock to be released by the thread doing the restart
            with Neo4jClient._restart_lock:
                pass  # Just wait for lock to be released
            return True
        
        # We have the lock, so we're responsible for restarting
        try:
            print("Attempting to restart Neo4j service...")
            if not (self.ssh_host and self.ssh_user and self.ssh_password):
                print("SSH credentials not provided, cannot restart Neo4j")
                return False
                
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.ssh_host, username=self.ssh_user, password=self.ssh_password)
            
            # Start Neo4j service
            stdin, stdout, stderr = ssh.exec_command("sudo systemctl start neo4j")
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                print(f"Error starting Neo4j: {stderr.read().decode()}")
                return False
                
            # Enable Neo4j service
            stdin, stdout, stderr = ssh.exec_command("sudo systemctl enable neo4j")
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                print(f"Error enabling Neo4j: {stderr.read().decode()}")
                return False
                
            # Check Neo4j status
            for _ in range(5):  # Try checking status a few times
                stdin, stdout, stderr = ssh.exec_command("sudo systemctl status neo4j")
                status_output = stdout.read().decode()
                if "Active: active" in status_output:
                    print("Neo4j service is now active")
                    ssh.close()
                    return True
                time.sleep(5) # wait 
                
            print("Neo4j service did not become active in the expected time")
            ssh.close()
            return False
            
        except Exception as e:
            print(f"SSH connection error: {str(e)}")
            return False
        finally:
            # Release the lock when done
            Neo4jClient._restart_lock.release()

    def close(self):
        self.driver.close()

    @neo4j_operation_with_retry()
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

    @neo4j_operation_with_retry()
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

    @neo4j_operation_with_retry()
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

    @neo4j_operation_with_retry()
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

    @neo4j_operation_with_retry()
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

    @neo4j_operation_with_retry()
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

    @neo4j_operation_with_retry()
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

    @neo4j_operation_with_retry()
    def get_coauthor_network(self, author_id: str, limit: int = 50):
        """
        Get the coauthor network for a given author, including shared works.
        Returns a graph structure showing how authors are connected through papers.
        """
        with self.driver.session() as session:
            query = """
            MATCH (a:Author {id: $author_id})-[auth1:AUTHORED]->(w:Work)<-[auth2:AUTHORED]-(co:Author)
            WHERE a <> co
            RETURN DISTINCT a, auth1, w, auth2, co
            LIMIT $limit
            """
            result = session.run(query, author_id=author_id, limit=limit)
            return result.graph()

    @neo4j_operation_with_retry()
    def get_topic_network(self, author_id: str, limit: int = 50):
        """
        Get the topic-based network for a given author, showing connections
        through shared research topics with other authors.
        """
        with self.driver.session() as session:
            query = """
            MATCH (a:Author {id: $author_id})-[r1:RESEARCHES]->(t:Topic)<-[r2:RESEARCHES]-(other:Author)
            WHERE a <> other
            RETURN DISTINCT a, r1, t, r2, other
            LIMIT $limit
            """
            result = session.run(query, author_id=author_id, limit=limit)
            return result.graph()

    @neo4j_operation_with_retry()
    def get_author_topics(self, author_id: str, limit: int = 20):
        """
        Get all topics researched by an author with their paper counts,
        returned as a graph structure showing author-topic relationships
        """
        with self.driver.session() as session:
            query = """
            MATCH (a:Author {id: $author_id})-[r:RESEARCHES]->(t:Topic)
            RETURN DISTINCT a, r, t
            ORDER BY r.paperCount DESC
            LIMIT $limit
            """
            result = session.run(query, author_id=author_id, limit=limit)
            return result.graph()

    @neo4j_operation_with_retry()
    def execute_custom_query(self, query: str, params: Dict = None, limit: int = 50):
        """
        Execute a custom Cypher query and return the graph result.
        The query should return a graph structure.
        
        Args:
            query: The Cypher query to execute
            params: Optional parameters for the query
            limit: Maximum number of results to return
        
        Returns:
            Neo4j graph object containing nodes and relationships
        """
        with self.driver.session() as session:
            # Add LIMIT clause if not present in query
            if "LIMIT" not in query.upper():
                query = f"{query}\nLIMIT {limit}"
            
            # Ensure params is a dictionary
            params = params or {}
            print("QUERY params: ", params)
            try:
                result = session.run(query, params)
                return result.graph()
            except Exception as e:
                print(f"Error executing Cypher query: {str(e)}")
                raise Exception(f"Error executing Cypher query: {str(e)}")

    @neo4j_operation_with_retry()
    def get_node_by_id(self, node_type: str, node_id: str) -> dict:
        """
        Get properties of a node by its ID.
        
        Args:
            node_type: One of 'Work', 'Author', 'Topic', or 'Institution'
            node_id: OpenAlex URL ID
        
        Returns:
            Dictionary of node properties
        """
        with self.driver.session() as session:
            query = f"""
            MATCH (n:{node_type} {{id: $node_id}})
            RETURN properties(n) as props
            """
            result = session.run(query, node_id=node_id)
            record = result.single()
            return record["props"] if record else None 