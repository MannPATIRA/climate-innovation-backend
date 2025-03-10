import pytest
from unittest.mock import MagicMock
from ranking_model.SevenQRanker import SevenQRanker
from tests.ranking_model.common_fixtures import test_authors, test_papers

@pytest.fixture
def mock_supabase_client():
    class MockSupabaseClient:
        def table(self, table_name):
            return self
        def upsert(self, data):
            return self
        def execute(self):
            return None
    return MockSupabaseClient()

@pytest.fixture
def seven_q_ranker(mock_supabase_client):
    return SevenQRanker(
        supabase_client=mock_supabase_client,
        model_name="gpt-4o-mini",
        learning_rate=0.01
    )

class TestSevenQRanker:
    def test_rank_papers(self, seven_q_ranker, test_papers):
        ranked_papers = seven_q_ranker.rank_papers(test_papers)
        
        assert len(ranked_papers) == 3
        assert all(hasattr(p, 'score') for p in ranked_papers)
        
        # Check papers are ranked by score in descending order
        for i in range(len(ranked_papers) - 1):
            assert ranked_papers[i].score >= ranked_papers[i + 1].score

    def test_rank_papers_empty_list(self, seven_q_ranker):
        ranked_papers = seven_q_ranker.rank_papers([])
        assert ranked_papers == []

    def test_rank_papers_single_paper(self, seven_q_ranker, test_papers):
        single_paper = test_papers[0]
        ranked_papers = seven_q_ranker.rank_papers([single_paper])
        assert len(ranked_papers) == 1
        assert hasattr(ranked_papers[0], 'score')

    def test_rank_papers_score_calculation(self, seven_q_ranker, test_papers):
        ranked_papers = seven_q_ranker.rank_papers(test_papers)
        for paper in ranked_papers:
            assert isinstance(paper.score, (int, float))
            assert -8 <= paper.score <= 5

    def test_rank_papers_maintains_data(self, seven_q_ranker, test_papers):
        original_titles = [p.title for p in test_papers]
        ranked_papers = seven_q_ranker.rank_papers(test_papers)
        ranked_titles = [p.title for p in ranked_papers]
        assert set(original_titles) == set(ranked_titles)