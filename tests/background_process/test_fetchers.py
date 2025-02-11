import pytest
import os
import shutil
from unittest.mock import Mock, patch
from background_process.fetchers.report_fetcher import LocalPDFFetcher
from background_process.fetchers.paper_fetcher import PyAlexFetcher
from background_process.fetchers.topic_fetcher import TopicFetcher
from background_process.processors.base import ProcessingTask

# Fixtures
@pytest.fixture
def test_directory(tmp_path):
    """Create a temporary directory with some PDF files for testing"""
    test_dir = tmp_path / "test_pdfs"
    test_dir.mkdir()
    
    # Create some test PDF files
    (test_dir / "test1.pdf").write_text("dummy content")
    (test_dir / "test2.pdf").write_text("dummy content")
    (test_dir / "not_pdf.txt").write_text("dummy content")
    
    return str(test_dir)

@pytest.fixture
def pdf_fetcher(test_directory):
    """Create a LocalPDFFetcher instance"""
    return LocalPDFFetcher(test_directory)

# Tests for LocalPDFFetcher
def test_local_pdf_fetcher_init(test_directory):
    """Test LocalPDFFetcher initialization"""
    fetcher = LocalPDFFetcher(test_directory)
    
    assert fetcher.directory == test_directory
    assert os.path.exists(fetcher.temp_directory)
    assert "processing_temp" in fetcher.temp_directory

def test_local_pdf_fetcher_fetch(pdf_fetcher, test_directory):
    """Test fetching PDF files"""
    paths = list(pdf_fetcher.fetch())
    
    # Should only get PDF files
    assert len(paths) == 2
    
    # Check if files are in temp directory
    for path in paths:
        assert os.path.exists(path)
        assert "processing_temp" in path
        assert path.endswith('.pdf')

def test_local_pdf_fetcher_cleanup(test_directory):
    """Test cleanup of temporary directory"""
    temp_dir = None
    
    # Create a new scope for the fetcher
    with pytest.raises(Exception):
        with LocalPDFFetcher(test_directory) as fetcher:
            temp_dir = fetcher.temp_directory
            assert os.path.exists(temp_dir)
            raise Exception("Force cleanup")
    
    # Check if temp directory was cleaned up
    assert not os.path.exists(temp_dir)

@pytest.fixture
def mock_supabase_response():
    mock_response = Mock()
    mock_response.data = [{"id": 123, "cursor": "test_cursor", "current_cursor": "current_test_cursor"}]
    return mock_response

@pytest.fixture
def mock_supabase_client(mock_supabase_response):
    mock_client = Mock()
    # Setup the method chain to return our mock response
    mock_chain = Mock()
    mock_chain.execute.return_value = mock_supabase_response
    mock_chain.eq.return_value = mock_chain
    mock_chain.select.return_value = mock_chain
    mock_client.table.return_value = mock_chain
    return mock_client

@pytest.fixture
def mock_works():
    return Mock()

@pytest.fixture
def mock_topics():
    return Mock()

@pytest.fixture
def paper_fetcher(mock_supabase_client):
    return PyAlexFetcher(mock_supabase_client)

class TestPyAlexFetcher:
    def test_init(self, mock_supabase_client):
        # Setup basic mock responses
        mock_supabase_client.table().select().eq().execute.return_value = Mock(
            data=[{"id": 123}]
        )
        
        # Mock the other method calls to return specific values
        with patch.object(PyAlexFetcher, '_get_main_cursor', return_value='test_cursor'), \
            patch.object(PyAlexFetcher, '_get_current_cursor', return_value='current_test_cursor'), \
            patch.object(PyAlexFetcher, '_get_climate_relevant_topics', return_value=['topic1', 'topic2', 'topic3']):
            
            # Create the fetcher
            fetcher = PyAlexFetcher(mock_supabase_client)

            # Verify all initialized values
            assert fetcher.task_id == 123
            assert fetcher.cursor == 'test_cursor'
            assert fetcher.current_cursor == 'current_test_cursor'
            assert len(fetcher.climate_relevant_topics) == 3
            assert 'topic1' in fetcher.climate_relevant_topics
            assert 'topic2' in fetcher.climate_relevant_topics
            assert 'topic3' in fetcher.climate_relevant_topics
            assert isinstance(fetcher.climate_relevant_topics, set)
            
    def test_mark_batch_complete(self, mock_supabase_client, mock_supabase_response):
        # Setup the mock responses for initialization
        with patch.object(PyAlexFetcher, '_get_climate_relevant_topics', return_value=['topic1']):
            fetcher = PyAlexFetcher(mock_supabase_client)
            fetcher.mark_batch_complete()
        
        # Verify the update was called
        mock_supabase_client.table.assert_called_with('processor_progress')
        mock_supabase_client.table().update.assert_called_once()

    def test_get_paper_processing_task_id_existing(self, mock_supabase_client, mock_supabase_response):
        with patch.object(PyAlexFetcher, '_get_main_cursor', return_value='test_cursor'), \
             patch.object(PyAlexFetcher, '_get_current_cursor', return_value='current_test_cursor'), \
             patch.object(PyAlexFetcher, '_get_climate_relevant_topics', return_value=['topic1']):
            
            fetcher = PyAlexFetcher(mock_supabase_client)
            result = fetcher._get_paper_processing_task_id()
            
            assert result == 123

    def test_get_paper_processing_task_id_new(self, mock_supabase_client):
        # Mock empty response for first check
        empty_response = Mock()
        empty_response.data = []
        
        # Mock response for insert
        insert_response = Mock()
        insert_response.data = [{"id": 456}]
        
        # Setup the mock chain for both scenarios
        mock_chain = Mock()
        # Provide enough side effects for potential retries (3 empty responses + 3 insert responses)
        mock_chain.execute.side_effect = [empty_response, insert_response] * 3
        mock_chain.eq.return_value = mock_chain
        mock_chain.select.return_value = mock_chain
        mock_chain.insert.return_value = mock_chain
        mock_supabase_client.table.return_value = mock_chain

        with patch.object(PyAlexFetcher, '_get_main_cursor', return_value='test_cursor'), \
            patch.object(PyAlexFetcher, '_get_current_cursor', return_value='current_test_cursor'), \
            patch.object(PyAlexFetcher, '_get_climate_relevant_topics', return_value=['topic1']):
            
            fetcher = PyAlexFetcher(mock_supabase_client)
            result = fetcher._get_paper_processing_task_id()
            
            assert result == 456
            
            # Verify the correct sequence of calls
            mock_supabase_client.table.assert_called_with('processor_progress')
            mock_chain.select.assert_called_with("*")
            mock_chain.eq.assert_called_with('task', ProcessingTask.PAPER_PROCESSING.value)

            # Verify insert was called with correct arguments, regardless of number of retries
            mock_chain.insert.assert_called_with({
                "task": ProcessingTask.PAPER_PROCESSING.value
            })

    def test_get_main_cursor(self, mock_supabase_client):
        cursor_response = Mock()
        cursor_response.data = [{"cursor": "test_cursor"}]
        
        mock_chain = Mock()
        mock_chain.execute.return_value = cursor_response
        mock_chain.eq.return_value = mock_chain
        mock_chain.select.return_value = mock_chain
        mock_supabase_client.table.return_value = mock_chain

        with patch.object(PyAlexFetcher, '_get_paper_processing_task_id', return_value=123), \
             patch.object(PyAlexFetcher, '_get_current_cursor', return_value='current_test_cursor'), \
             patch.object(PyAlexFetcher, '_get_climate_relevant_topics', return_value=['topic1']):
            
            fetcher = PyAlexFetcher(mock_supabase_client)
            result = fetcher._get_main_cursor()
            
            assert result == "test_cursor"

class TestTopicFetcher:
    @patch('background_process.fetchers.Topics')
    @patch('background_process.fetchers.Works')
    def test_fetch(self, mock_works_class, mock_topics_class):
        # Arrange
        mock_topics = Mock()
        mock_topics_class.return_value = mock_topics
        mock_topics.get.return_value = (
            [
                {
                    'id': 'topic1',
                    'display_name': 'Topic 1',
                    'description': 'Description 1'
                }
            ],
            {'next_cursor': None}
        )

        mock_works = Mock()
        mock_works_class.return_value = mock_works
        mock_works.filter.return_value = mock_works
        mock_works.sort.return_value = mock_works
        mock_works.select.return_value = mock_works
        mock_works.paginate.return_value = [[
            {
                'title': 'Paper 1',
                'abstract_inverted_index_v3': {'test': [0]}
            }
        ]]

        fetcher = TopicFetcher()

        # Act
        results = list(fetcher.fetch())

        # Assert
        assert len(results) == 1
        assert results[0]['topic_id'] == 'topic1'
        assert len(results[0]['sample_works']) == 1
        assert results[0]['sample_works'][0]['abstract'] == 'test'