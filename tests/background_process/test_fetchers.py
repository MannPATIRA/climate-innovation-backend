import pytest
import os
import shutil
from unittest.mock import Mock, patch
from background_process.fetchers import LocalPDFFetcher, PyAlexFetcher

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
        fetcher = LocalPDFFetcher(test_directory)
        temp_dir = fetcher.temp_directory
        assert os.path.exists(temp_dir)
        raise Exception("Force cleanup")
    
    # Check if temp directory was cleaned up
    assert not os.path.exists(temp_dir)

# Tests for PyAlexFetcher
def test_pyalex_fetcher_init():
    """Test PyAlexFetcher initialization"""
    fetcher = PyAlexFetcher()
    assert isinstance(fetcher, PyAlexFetcher)

@patch('pyalex.Works')
def test_pyalex_fetcher_fetch(mock_works):
    """Test fetching papers from PyAlex"""
    # Mock the Works class and its methods
    mock_works_instance = Mock()
    mock_works.return_value = mock_works_instance
    
    # Mock the filter method to return self
    mock_works_instance.filter.return_value = mock_works_instance
    
    # Mock the get method to return a page with one paper
    mock_page = Mock()
    mock_page.get_next_cursor.return_value = None  # No more pages
    mock_paper = {
        'abstract': 'Test abstract',
        'id': 'test_id',
        'doi': 'test_doi',
        'title': 'Test Title'
    }
    mock_page.__iter__ = lambda self: iter([mock_paper])
    mock_works_instance.get.return_value = mock_page
    
    # Create fetcher and get results
    fetcher = PyAlexFetcher()
    results = list(fetcher.fetch(country='US'))
    
    # Verify results
    assert len(results) == 1
    abstract, metadata = results[0]
    assert abstract == 'Test abstract'
    assert metadata == {
        'id': 'test_id',
        'doi': 'test_doi',
        'title': 'Test Title'
    }
    
    # Verify that filter was called with correct arguments
    assert mock_works_instance.filter.called
    
    # Verify that get was called with correct arguments
    mock_works_instance.get.assert_called_with(per_page=100, cursor='*')

def test_pyalex_fetcher_fetch_no_abstract():
    """Test handling of papers without abstracts"""
    fetcher = PyAlexFetcher()
    # Mock the Works class to return a paper without an abstract
    with patch('pyalex.Works') as mock_works:
        mock_works_instance = Mock()
        mock_works.return_value = mock_works_instance
        mock_works_instance.filter.return_value = mock_works_instance
        
        mock_page = Mock()
        mock_page.get_next_cursor.return_value = None
        mock_paper = {
            'abstract': None,
            'id': 'test_id',
            'doi': 'test_doi',
            'title': 'Test Title'
        }
        mock_page.__iter__ = lambda self: iter([mock_paper])
        mock_works_instance.get.return_value = mock_page
        
        results = list(fetcher.fetch(country='US'))
        assert len(results) == 0  # Should skip papers without abstracts