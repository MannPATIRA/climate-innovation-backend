import pytest
import requests
from unittest.mock import Mock, patch

from backend_server.gatherers.SemanticScholarInformationGatherer import SemanticScholarInformationGatherer
from backend_server.gatherers.OpenAlexInformationGatherer import OpenAlexInformationGatherer

# Test data
MOCK_ARXIV_DOI = "2303.11366"
MOCK_REGULAR_DOI = "10.1234/example"
MOCK_AUTHOR_ID = "2212367248"
MOCK_PAPER_ID = "https://openalex.org/W2105503244"

# Mock responses
MOCK_AUTHORS_RESPONSE = {
    "data": [
        {"authorId": "author1", "name": "John Doe"},
        {"authorId": "author2", "name": "Jane Smith"}
    ]
}

MOCK_AUTHOR_INFO = {
    "name": "John Doe",
    "affiliations": ["Imperial College London"],
    "paperCount": 42,
    "citationCount": 1000,
    "hIndex": 20,
    "papers": [{"paperId": "paper1"}, {"paperId": "paper2"}],
    "externalIds": {"ORCID": "0000-0000-0000-0000"}
}

MOCK_PAPER_INFO = {
    "data": {
        "externalIds": {
            "DOI": "10.1234/example",
            "arXiv": "2303.11366"
        }
    }
}

class TestSemanticScholarInformationGatherer:
    def test_get_authors_from_arxiv_doi(self, requests_mock):
        requests_mock.get(
            f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{MOCK_ARXIV_DOI}/authors?fields=authorId,name",
            json=MOCK_AUTHORS_RESPONSE
        )
        
        result = SemanticScholarInformationGatherer.get_authors_from_doi(f"arXiv.{MOCK_ARXIV_DOI}")
        assert result == MOCK_AUTHORS_RESPONSE["data"]
    
    def test_get_authors_from_regular_doi(self, requests_mock):
        requests_mock.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{MOCK_REGULAR_DOI}/authors?fields=authorId,name",
            json=MOCK_AUTHORS_RESPONSE
        )
        
        result = SemanticScholarInformationGatherer.get_authors_from_doi(f"doi.org/{MOCK_REGULAR_DOI}")
        assert result == MOCK_AUTHORS_RESPONSE["data"]
    
    def test_get_author_info(self, requests_mock):
        fields = "name,affiliations,paperCount,citationCount,hIndex,papers.paperId,externalIds"
        requests_mock.get(
            f"https://api.semanticscholar.org/graph/v1/author/{MOCK_AUTHOR_ID}?fields={fields}",
            json=MOCK_AUTHOR_INFO
        )
        
        result = SemanticScholarInformationGatherer.get_author_info(MOCK_AUTHOR_ID)
        assert result == MOCK_AUTHOR_INFO
    
    def test_get_author_info_with_exclude(self, requests_mock):
        exclude = ["paperCount", "citationCount"]
        fields = "name,affiliations,hIndex,papers.paperId,externalIds"
        requests_mock.get(
            f"https://api.semanticscholar.org/graph/v1/author/{MOCK_AUTHOR_ID}?fields={fields}",
            json=MOCK_AUTHOR_INFO
        )
        
        result = SemanticScholarInformationGatherer.get_author_info(MOCK_AUTHOR_ID, exclude=exclude)
        assert result == MOCK_AUTHOR_INFO
    
    def test_get_doi_from_paper_id(self, requests_mock):

        requests_mock.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{MOCK_PAPER_ID}?fields=externalIds,name",
            json=MOCK_PAPER_INFO
        )

        result = SemanticScholarInformationGatherer.get_doi_from_paper_id(MOCK_PAPER_ID)
        assert result == MOCK_PAPER_INFO["data"]["externalIds"]

class TestOpenAlexInformationGatherer:
    @patch('backend_server.gatherers.Works')
    def test_get_UK_authors_from_doi(self, mock_works):
        mock_work = Mock()
        mock_work.filter.return_value.get.return_value = [{
            'authorships': [
                {'author': {'id': 'author1', 'display_name': 'John Doe'}, 'institutions': [{'country_code': 'GB'}]},
                {'author': {'id': 'author2', 'display_name': 'Jane Smith'}, 'institutions': [{'country_code': 'US'}]},
                {'author': {'id': 'author3', 'display_name': 'Peter Pan'}, 'institutions': [{'country_code': 'GB'}, {'country_code': 'CA'}]}
            ]
        }]
        mock_works.return_value = mock_work
        
        result = OpenAlexInformationGatherer.get_UK_authors_from_doi(MOCK_REGULAR_DOI)
        expected = [
            {'authorId': 'author1', 'name': 'John Doe'},
            {'authorId': 'author3', 'name': 'Peter Pan'}
        ]
        assert result == expected

    @patch('backend_server.gatherers.Authors')
    @patch('backend_server.gatherers.Works')
    def test_get_author_info(self, mock_works, mock_authors):
        mock_author = {
            'display_name': 'John Doe',
            'last_known_institution': [{'display_name': 'University X'}],
            'works_count': 42,
            'cited_by_count': 1000,
            'h_index': 20,
            'ids': {'orcid': '0000-0000-0000-0000'}
        }
        mock_authors.return_value.__getitem__.return_value = mock_author
        
        mock_works.return_value.filter.return_value.get.return_value = [
            {'id': 'paper1'}, {'id': 'paper2'}
        ]
        
        result = OpenAlexInformationGatherer.get_author_info(MOCK_AUTHOR_ID)
        assert result['name'] == 'John Doe'
        assert result['works_count'] == 42
        assert result['citations'] == 1000

    @patch('backend_server.gatherers.Works')
    def test_get_details_from_paper_id(self, mock_works):
        expected_result = {
            "title": "Some Title",
            "publication_date": "2024-01-01",
            "abstract": "Some abstract"
        }

        mock_instance = mock_works.return_value
        mock_instance.__getitem__.return_value = expected_result  # Mock dictionary access

        result = OpenAlexInformationGatherer.get_details_from_paper_id(MOCK_PAPER_ID)

        assert result == expected_result

    def test_get_relevant_concepts_from_paper(self):
        mock_paper = {
            'concepts': [
                {'display_name': 'AI', 'level': 1, 'score': 0.8},
                {'display_name': 'ML', 'level': 2, 'score': 0.05},
                {'display_name': 'DL', 'level': 1, 'score': 0.3}
            ]
        }
        
        result = OpenAlexInformationGatherer.get_relevant_concepts_from_paper(mock_paper)
        assert len(result) == 2  # Only concepts with score > 0.1
        assert ('AI', 1, 0.8) in result
        assert ('DL', 1, 0.3) in result

class TestErrorCases:
    def test_semantic_scholar_invalid_doi(self, requests_mock):
        requests_mock.get(
            "https://api.semanticscholar.org/graph/v1/paper/DOI:invalid/authors?fields=authorId,name",
            status_code=404,
            json={
                "error": "Paper not found",
                "code": 404
            }
        )
        
        with pytest.raises(requests.exceptions.HTTPError):
            SemanticScholarInformationGatherer.get_authors_from_doi("doi.org/invalid")
    
    def test_semantic_scholar_missing_data(self, requests_mock):
        requests_mock.get(
            "https://api.semanticscholar.org/graph/v1/paper/DOI:invalid/authors?fields=authorId,name",
            status_code=200,
            json={
                "error": "Some other error"
            }
        )
        
        with pytest.raises(ValueError, match="Invalid API response format: 'data' field missing"):
            SemanticScholarInformationGatherer.get_authors_from_doi("doi.org/invalid")
            
    def test_semantic_scholar_invalid_json(self, requests_mock):
        requests_mock.get(
            "https://api.semanticscholar.org/graph/v1/paper/DOI:invalid/authors?fields=authorId,name",
            status_code=200,
            text="Invalid JSON"
        )
      
        with pytest.raises(requests.exceptions.JSONDecodeError):
            SemanticScholarInformationGatherer.get_authors_from_doi("doi.org/invalid")

    @patch('backend_server.gatherers.Works')
    def test_openalex_invalid_doi(self, mock_works):
        mock_works.return_value.filter.return_value.get.return_value = None
        
        result = OpenAlexInformationGatherer.get_UK_authors_from_doi("invalid_doi")
        assert result == []

    @patch('backend_server.gatherers.Works')
    def test_openalex_invalid_paper_id(self, mock_works):
        mock_instance = mock_works.return_value
        mock_instance.__getitem__.return_value = None

        result = OpenAlexInformationGatherer.get_details_from_paper_id("invalid_paper_id")
        assert result is None

    @patch('backend_server.gatherers.Authors')
    def test_openalex_invalid_author_id(self, mock_authors):
        mock_authors.return_value.__getitem__.return_value = None
        
        result = OpenAlexInformationGatherer.get_author_info("invalid_author_id")
        assert result == {}

    def test_semantic_scholar_network_error(self, requests_mock):
        requests_mock.get(
            f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{MOCK_ARXIV_DOI}/authors?fields=authorId,name",
            exc=requests.exceptions.ConnectionError
        )
        
        with pytest.raises(requests.exceptions.ConnectionError):
            SemanticScholarInformationGatherer.get_authors_from_doi(f"arXiv.{MOCK_ARXIV_DOI}")