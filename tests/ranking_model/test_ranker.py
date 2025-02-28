import pytest
import numpy as np
from unittest.mock import MagicMock
from ranking_model.OnlineSVMRanker import OnlineSVMRanker
from ranking_model.author import Author
from ranking_model.paper import Paper
from ranking_model.RegressionRanker import RegressionRanker

from tests.ranking_model.common_fixtures import test_authors, test_papers

class MockSupabaseClient:
    def __init__(self):
        self.execute_return_value = MagicMock(data=[{'model_data': '{"author_weights": {"citations": 0.5, "hindex": 0.5, "total_grant_value": 0.1, "num_grants": 0.1, "works_count": 0.1}, "paper_weights": {"relevancy": 1.0}}'}])
    def table(self, table_name):
        return self
    def update(self, data):
        return self
    def insert(self, data):
        return self
    def eq(self, column_name, value):
        return self
    def select(self, columns):
        return self
    def execute(self):
        return self.execute_return_value

@pytest.fixture
def model_name():
    return "test_model"

@pytest.fixture
def mock_supabase_client(model_name):
    return MockSupabaseClient()

@pytest.fixture
def online_ranksvm_ranker(mock_supabase_client, model_name):
    # High learning rate = 0.5
    ranker = OnlineSVMRanker(mock_supabase_client, model_name, learning_rate=0.5)
    return ranker

@pytest.fixture
def regression_ranker(mock_supabase_client, model_name):
    # High learning rate for testing
    ranker = RegressionRanker(supabase_client=mock_supabase_client, model_name=model_name, learning_rate=0.5)
    return ranker

class TestRegressionRanker:
    def test_rank_authors(self, regression_ranker):
        authors = [
            Author(name="Alice", citations=100, dob="1990-01-01", organisation_history=[], orcid="", hindex=10, grants=[], grant_org_name="", website="", openAlexid="", works_count=100),
            Author(name="Bob", citations=50, dob="1990-01-01", organisation_history=[], orcid="", hindex=5, grants=[], grant_org_name="", website="", openAlexid="", works_count=50),
        ]
        ranked_authors = regression_ranker.rank_authors(authors)
        assert ranked_authors[0].name == "Alice"
        assert ranked_authors[1].name == "Bob"
        assert ranked_authors[0].score >= ranked_authors[1].score

    def test_rank_papers(self, regression_ranker):
        authors = [
            Author(name="Alice", citations=100, dob="1990-01-01", organisation_history=[], orcid="", hindex=10, grants=[], grant_org_name="", website="", openAlexid="", works_count=100),
            Author(name="Bob", citations=50, dob="1990-01-01", organisation_history=[], orcid="", hindex=5, grants=[], grant_org_name="", website="", openAlexid="", works_count=50),
        ]
        paper1 = Paper(paper_id=1, openalex_id="", title="Paper 1", relevancy=0.8, authors=authors, doi="", abstract="", publication_date="")
        paper2 = Paper(paper_id=2, openalex_id="", title="Paper 2", relevancy=0.6, authors=authors, doi="", abstract="", publication_date="")
        papers = [paper1, paper2]
        ranked_papers = regression_ranker.rank_papers(papers)
        assert ranked_papers[0].title == "Paper 1"
        assert ranked_papers[1].title == "Paper 2"
        assert ranked_papers[0].score >= ranked_papers[1].score

    def test_update_author_model(self, regression_ranker):
        initial_citations_weight = float(regression_ranker.author_weights['citations'])
        author = Author(name="Alice", citations=100, dob="1990-01-01", organisation_history=[], orcid="", hindex=10, grants=[], grant_org_name="", website="", openAlexid="", works_count=100)
        regression_ranker.update_author_model(author, label=0)  # Changed label to 0
        assert float(regression_ranker.author_weights['citations']) != initial_citations_weight

    def test_update_paper_model(self, regression_ranker):
        initial_relevancy_weight = regression_ranker.paper_weights['relevancy']
        paper = Paper(paper_id=1, openalex_id="", title="Paper 1", relevancy=0.8, authors=[], doi="", abstract="", publication_date="")
        regression_ranker.update_paper_model(paper, label=1)
        assert regression_ranker.paper_weights['relevancy'] != initial_relevancy_weight

    def test_save_load_model(self, regression_ranker, mock_supabase_client):
        # Update mock to return the new weights
        mock_supabase_client.execute_return_value = MagicMock(data=[{'model_data': '{"author_weights": {"citations": 0.6, "hindex": 0.5, "total_grant_value": 0.1, "num_grants": 0.1, "works_count": 0.1}, "paper_weights": {"relevancy": 1.2}}'}])
        
        regression_ranker.author_weights['citations'] = 0.6
        regression_ranker.paper_weights['relevancy'] = 1.2

        regression_ranker.save_model()

        new_ranker = RegressionRanker(supabase_client=mock_supabase_client, model_name=regression_ranker.model_name)
        new_ranker.load_model()

        assert new_ranker.author_weights['citations'] == 0.6
        assert new_ranker.paper_weights['relevancy'] == 1.2

    def test_rank_authors_empty_list(self, regression_ranker):
        ranked_authors = regression_ranker.rank_authors([])
        assert ranked_authors == []

    def test_rank_authors_single_author(self, regression_ranker):
        author = Author(name="Alice", citations=100, dob="1990-01-01", 
                       organisation_history=[], orcid="", hindex=10, grants=[], 
                       grant_org_name="", website="", openAlexid="", works_count=100)
        ranked_authors = regression_ranker.rank_authors([author])
        assert len(ranked_authors) == 1
        assert ranked_authors[0].score is not None

class TestOnlineSVMRanker:
    def test_online_ranksvm_initial_ranking(self, online_ranksvm_ranker, test_papers):
        ranked_papers = online_ranksvm_ranker.rank_papers(test_papers)
        assert len(ranked_papers) == 3
        # Check papers are ranked by score in descending order
        for i in range(len(ranked_papers) - 1):
            assert ranked_papers[i].score >= ranked_papers[i + 1].score

    def test_online_ranksvm_single_update(self, online_ranksvm_ranker, test_papers, test_authors):
        paper1 = test_papers[0]
        authors1 = test_authors[0]
        bob = authors1[1]

        # Rank papers initially and get Bob's initial score
        online_ranksvm_ranker.rank_papers(test_papers)
        initial_score = bob.score

        # Delete author
        online_ranksvm_ranker.accept_author(paper1, bob)

        # Rank papers again to update scores
        online_ranksvm_ranker.rank_papers(test_papers)

        # Check that Bob's score has increased from initial score
        assert bob.score > initial_score

    def test_online_ranksvm_multiple_updates(self, online_ranksvm_ranker, test_papers, test_authors):
        paper2 = test_papers[1]
        paper3 = test_papers[2]
        authors2 = test_authors[1]
        eve = authors2[1]

        # Rank papers initially
        ranked_papers = online_ranksvm_ranker.rank_papers(test_papers)
        initial_score = eve.score

        authors1 = test_authors[0]
        bob = authors1[1]
        # Perform model initialisation
        online_ranksvm_ranker.accept_author(paper2, bob)
        online_ranksvm_ranker.accept_paper(paper2)

        # Rank paper copy to apply the changes
        ranked_papers = online_ranksvm_ranker.rank_papers(test_papers.copy())
        online_ranksvm_ranker.delete_author(paper2, eve)
        
        ranked_papers = online_ranksvm_ranker.rank_papers(test_papers.copy())
        # Check that Eve's score has increased
        assert eve.score > initial_score

        online_ranksvm_ranker.delete_paper(paper2)
        ranked_papers = online_ranksvm_ranker.rank_papers(test_papers)

    def test_online_ranksvm_model_consistency(self, online_ranksvm_ranker, test_papers):
        paper1 = test_papers[0]
        new_author = Author(
            name="NewBob", 
            citations=100, 
            dob="1980-03-22", 
            hindex=15,
            organisation_history=["Uni X"],
            orcid="0000-0009-9012-3456",
            grants=[],
            grant_org_name="New Foundation",
            website="https://newbob.com",
            openAlexid="NB12345",
            works_count=30
        )
        paper1.authors.append(new_author)
        ranked_papers = online_ranksvm_ranker.rank_papers(test_papers)
        assert len(ranked_papers) == 3
        assert hasattr(new_author, 'score')

    def test_online_ranksvm_incremental_learning(self, online_ranksvm_ranker, test_papers, test_authors):
        paper2 = test_papers[1]
        authors2 = test_authors[1]
        eve = authors2[1]
        online_ranksvm_ranker.rank_papers(test_papers)
        initial_score = eve.score
        
        for i in range(3):
            online_ranksvm_ranker.accept_author(paper2, eve)
            ranked_papers = online_ranksvm_ranker.rank_papers(test_papers)
            assert eve.score >= initial_score
            initial_score = eve.score