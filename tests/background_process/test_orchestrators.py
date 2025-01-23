import pytest
from unittest.mock import Mock, patch, call
import os
from background_process.orchestrators import ReportOrchestrator, PaperOrchestrator
from background_process.fetchers import ReportFetcher, PaperFetcher
from background_process.processors import ReportProcessor, PaperProcessor
from background_process.summary_processors import SummaryProcessor

# Fixtures
@pytest.fixture
def mock_fetcher():
    return Mock(spec=ReportFetcher)

@pytest.fixture
def mock_processor():
    return Mock(spec=ReportProcessor)

@pytest.fixture
def mock_summarizer():
    return Mock(spec=SummaryProcessor)

@pytest.fixture
def report_orchestrator(mock_fetcher, mock_processor, mock_summarizer):
    return ReportOrchestrator(
        fetcher=mock_fetcher,
        processor=mock_processor,
        summarizer=mock_summarizer
    )

# Tests for ReportOrchestrator
def test_report_orchestrator_initialization(report_orchestrator, mock_fetcher, mock_processor, mock_summarizer):
    """Test proper initialization of ReportOrchestrator"""
    assert report_orchestrator.fetcher == mock_fetcher
    assert report_orchestrator.processor == mock_processor
    assert report_orchestrator.summarizer == mock_summarizer

def test_process_single_report_success(report_orchestrator, mock_processor):
    """Test successful processing of a single report"""
    # Setup
    test_path = "test.pdf"
    mock_document = Mock()
    mock_record = Mock()
    mock_processor.process.return_value = (mock_document, mock_record)

    # Create a temporary file
    with open(test_path, 'w') as f:
        f.write("dummy content")

    # Execute
    report_orchestrator.process_single_report(test_path)

    # Verify
    mock_processor.process.assert_called_once_with({'report_path': test_path})
    report_orchestrator.summarizer.summarize.assert_called_once_with(
        mock_document, mock_record, test_path
    )
    assert not os.path.exists(test_path)  # File should be deleted

def test_process_single_report_without_summarizer():
    """Test processing a report without a summarizer"""
    # Setup
    mock_fetcher = Mock(spec=ReportFetcher)
    mock_processor = Mock(spec=ReportProcessor)
    orchestrator = ReportOrchestrator(
        fetcher=mock_fetcher,
        processor=mock_processor,
        summarizer=None
    )
    
    test_path = "test.pdf"
    mock_document = Mock()
    mock_record = Mock()
    mock_processor.process.return_value = (mock_document, mock_record)

    # Create a temporary file
    with open(test_path, 'w') as f:
        f.write("dummy content")

    # Execute
    orchestrator.process_single_report(test_path)

    # Verify
    mock_processor.process.assert_called_once_with({'report_path': test_path})
    assert not os.path.exists(test_path)  # File should be deleted

def test_process_single_report_processor_error(report_orchestrator, mock_processor):
    """Test handling of processor errors"""
    # Setup
    test_path = "test.pdf"
    mock_processor.process.side_effect = Exception("Processing error")

    # Create a temporary file
    with open(test_path, 'w') as f:
        f.write("dummy content")

    # Execute
    report_orchestrator.process_single_report(test_path)

    # Verify
    mock_processor.process.assert_called_once()
    report_orchestrator.summarizer.summarize.assert_not_called()
    assert not os.path.exists(test_path)  # File should still be deleted

def test_process_single_report_summarizer_error(report_orchestrator, mock_processor):
    """Test handling of summarizer errors"""
    # Setup
    test_path = "test.pdf"
    mock_document = Mock()
    mock_record = Mock()
    mock_processor.process.return_value = (mock_document, mock_record)
    report_orchestrator.summarizer.summarize.side_effect = Exception("Summarizer error")

    # Create a temporary file
    with open(test_path, 'w') as f:
        f.write("dummy content")

    # Execute
    report_orchestrator.process_single_report(test_path)

    # Verify
    mock_processor.process.assert_called_once()
    report_orchestrator.summarizer.summarize.assert_called_once()
    assert not os.path.exists(test_path)  # File should still be deleted

def test_run_processes_all_reports(report_orchestrator, mock_fetcher):
    """Test that run processes all reports from fetcher"""
    # Setup
    test_paths = ["test1.pdf", "test2.pdf", "test3.pdf"]
    mock_fetcher.fetch.return_value = test_paths

    # Create mock process_single_report method
    report_orchestrator.process_single_report = Mock()

    # Execute
    report_orchestrator.run()

    # Verify
    assert report_orchestrator.process_single_report.call_count == len(test_paths)
    calls = [call(path) for path in test_paths]
    report_orchestrator.process_single_report.assert_has_calls(calls)

# Tests for PaperOrchestrator
@pytest.fixture
def paper_orchestrator():
    mock_fetcher = Mock(spec=PaperFetcher)
    mock_processor = Mock(spec=PaperProcessor)
    mock_summarizer = Mock(spec=SummaryProcessor)
    return PaperOrchestrator(
        fetcher=mock_fetcher,
        processor=mock_processor,
        summarizer=mock_summarizer
    )

def test_paper_orchestrator_initialization(paper_orchestrator):
    """Test proper initialization of PaperOrchestrator"""
    assert isinstance(paper_orchestrator.fetcher, Mock)
    assert isinstance(paper_orchestrator.processor, Mock)
    assert isinstance(paper_orchestrator.summarizer, Mock)

# Add more tests for PaperOrchestrator once its run method is implemented