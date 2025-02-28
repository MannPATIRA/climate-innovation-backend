from collections import defaultdict
import json
import pytest
from unittest.mock import MagicMock
from typing import List
from supabase import Client

from ranking_model.ranker_manager import RankerManager
from ranking_model.ranker import Ranker
from ranking_model.paper import Paper
from ranking_model.author import Author
from tests.ranking_model.common_fixtures import test_authors, test_papers

class MockRankerBase(Ranker):
    def __init__(self, supabase_client, model_name, learning_rate=0.01, reverse: bool = False):
        super().__init__(supabase_client, model_name, learning_rate)
        self.reverse = reverse
        self.calls = {"update_author": 0, "update_paper": 0,
                      "delete_author": 0, "accept_author": 0,
                      "delete_paper": 0, "accept_paper": 0}

    def rank_papers(self, papers: List[Paper]) -> List[Paper]:
        return sorted(papers, key=lambda p: p.paper_id, reverse=self.reverse)

    def rank_authors(self, authors: List[Author]) -> List[Author]:
        return sorted(authors, key=lambda a: a.openAlexid, reverse=self.reverse)

    def update_author_model(self, author: Author, label: int):
        self.calls["update_author"] += 1

    def update_paper_model(self, paper: Paper, label: int):
        self.calls["update_paper"] += 1

    def delete_author(self, paper: Paper, author: Author):
        self.calls["delete_author"] += 1

    def accept_author(self, paper: Paper, author: Author):
        self.calls["accept_author"] += 1

    def delete_paper(self, paper: Paper):
        self.calls["delete_paper"] += 1

    def accept_paper(self, paper: Paper):
        self.calls["accept_paper"] += 1

    def save_model(self):
        pass

    def load_model(self) -> bool:
        return True

class MockRankerAsc(MockRankerBase):
    def __init__(self, supabase_client, model_name, learning_rate=0.01):
        super().__init__(supabase_client, model_name, learning_rate, reverse=False)

class MockRankerDesc(MockRankerBase):
    def __init__(self, supabase_client, model_name, learning_rate=0.01):
        super().__init__(supabase_client, model_name, learning_rate, reverse=True)

@pytest.fixture
def mock_supabase_client():
    mock = MagicMock()
    mock.table.return_value = mock
    mock.update.return_value = mock
    mock.insert.return_value = mock
    mock.eq.return_value = mock
    mock.select.return_value = mock
    mock_response = MagicMock(data=[{'model_data': json.dumps({
          "paper_weights": {"mock1": 0.3, "mock2": 0.7},
          "author_weights": {"mock1": 0.3, "mock2": 0.7}
        })}])
    mock.execute.return_value = mock_response
    return mock

@pytest.fixture
def ranker_classes():
    return {
        "mock1": MockRankerAsc,
        "mock2": MockRankerDesc,
    }

@pytest.fixture
def ranker_manager(mock_supabase_client, ranker_classes):
    model_name = "test_ranker_manager_model"
    manager = RankerManager(supabase_client=mock_supabase_client,
                            model_name=model_name,
                            ranker_classes=ranker_classes,
                            learning_rate=0.05)
    return manager

class TestRankerManager:
  def test_ensemble_rank_papers(self, ranker_manager, test_papers):
      ensemble_ranked = ranker_manager.rank_papers(test_papers.copy())
      scores = {p.paper_id: p.score for p in ensemble_ranked}
      sorted_scores = sorted(scores.values(), reverse=True)
      assert all(earlier >= later for earlier, later in zip(sorted_scores, sorted_scores[1:]))

  def test_ensemble_rank_authors(self, ranker_manager, test_authors):
      authors1, _, _ = test_authors
      ensemble_ranked = ranker_manager.rank_authors(authors1.copy())
      scores = {a.openAlexid: a.score for a in ensemble_ranked}
      assert scores["B12345"] < scores["A12345"]
      assert scores["B12345"] < scores["C12345"]
      sorted_scores = sorted(scores.values(), reverse=True)
      assert all(earlier >= later for earlier, later in zip(sorted_scores, sorted_scores[1:]))

  def test_update_and_delete_propagation(self, ranker_manager, test_papers, test_authors):
      paper = test_papers[0]
      authors1, _, _ = test_authors
      author = authors1[0]
      
      ranker_manager.update_author_model(author, label=1)
      ranker_manager.update_paper_model(paper, label=0)
      ranker_manager.accept_author(paper, author)
      ranker_manager.delete_author(paper, author)
      ranker_manager.accept_paper(paper)
      ranker_manager.delete_paper(paper)
      
      for r in ranker_manager.rankers.values():
          assert r.calls["update_author"] == 1
          assert r.calls["update_paper"] == 1
          assert r.calls["accept_author"] == 1
          assert r.calls["delete_author"] == 1
          assert r.calls["accept_paper"] == 1
          assert r.calls["delete_paper"] == 1

  def test_save_model(self, ranker_manager, mock_supabase_client):
      mock_supabase_client.table.reset_mock()
      ranker_manager.save_model()
      mock_supabase_client.table.assert_called_with('ranker_models')
      mock_supabase_client.update.assert_called()

  def test_load_model(self, ranker_manager, mock_supabase_client):
      success = ranker_manager.load_model()
      assert success is True
      assert ranker_manager.paper_weights == {"mock1": 0.3, "mock2": 0.7}
      assert ranker_manager.author_weights == {"mock1": 0.3, "mock2": 0.7}

  def test_store_feedback_exceptions(monkeypatch, ranker_manager, test_papers, test_authors):
      def fake_insert_fail(data):
          raise Exception("Insert failed")
      ranker_manager.supabase.table.return_value.insert.side_effect = fake_insert_fail

      ranker_manager.papers = test_papers
      ranker_manager.accepted_papers = test_papers[:1]
      ranker_manager.rejected_papers = test_papers[1:]
      try:
          ranker_manager._store_ranking_papers_feedback()
      except Exception:
          pytest.fail("Feedback storage for papers raised an exception.")

      authors1, _, _ = test_authors
      ranker_manager.authors = authors1
      ranker_manager.accepted_authors = authors1[:1]
      ranker_manager.rejected_authors = authors1[1:]
      try:
          ranker_manager._store_ranking_authors_feedback()
      except Exception:
          pytest.fail("Feedback storage for authors raised an exception.")