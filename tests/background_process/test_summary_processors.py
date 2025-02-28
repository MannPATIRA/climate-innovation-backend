import pytest
from unittest.mock import Mock, patch
from background_process.processors.summary_processor import Summarizer, SummaryProcessor
from background_process.processors.report_processor import PDFDocument
from background_process.utils.process_log_manager import ProcessLogManager

# Mock Summarizer Implementation
class MockSummarizer(Summarizer):
    def generate_summary(self, text: str) -> str:
        return f"Summary of: {text[:50]}..."

# Fixtures
@pytest.fixture
def mock_supabase():
    mock = Mock()
    mock_table = Mock()
    mock.table.return_value = mock_table
    mock_table.insert.return_value = mock_table
    mock_table.select.return_value = mock_table
    mock_table.eq.return_value = mock_table

    # Simulate the structure of a Supabase response
    mock_execute = Mock()
    mock_execute.data = [{
        "id": 1,
        "content": "test content",
        "report_title": "Test Report",
        "report_id": 1
    }]
    mock_table.execute.return_value = mock_execute

    return mock

@pytest.fixture
def mock_pinecone_store():
    return Mock()

@pytest.fixture
def mock_summarizer():
    return MockSummarizer()

@pytest.fixture
def summary_processor(mock_summarizer, mock_supabase, mock_pinecone_store):
    return SummaryProcessor(
        summarizer=mock_summarizer,
        process_log_manager=ProcessLogManager(mock_supabase),
        supabase_client=mock_supabase,
        pinecone_store=mock_pinecone_store
    )

@pytest.fixture
def sample_document():
    return PDFDocument(
        content="This is a test document content for testing summary generation.",
        title="Test Document",
        content_hash="test_hash"
    )

@pytest.fixture
def sample_report_record():
    return {
        "id": 1,
        "content": "test content",
        "content_hash": "test_hash"
    }

# Tests for Summarizer
def test_summarizer_generate_content_hash(mock_summarizer):
    """Test hash generation in Summarizer"""
    summarizer = mock_summarizer
    content = "test content"
    hash1 = summarizer.generate_content_hash(content)
    hash2 = summarizer.generate_content_hash(content)
    
    assert isinstance(hash1, str)
    assert hash1 == hash2
    assert hash1 != summarizer.generate_content_hash("different content")

# Tests for SummaryProcessor
def test_summary_processor_initialization(summary_processor, mock_summarizer, mock_supabase, mock_pinecone_store):
    """Test SummaryProcessor initialization"""
    assert summary_processor.summarizer == mock_summarizer
    assert summary_processor.supabase == mock_supabase
    assert summary_processor.pinecone_store == mock_pinecone_store
    assert summary_processor.text_splitter is not None

def test_generate_summary(mock_summarizer):
    """Test summary generation"""
    text = "This is a test document that needs to be summarized."
    summary = mock_summarizer.generate_summary(text)
    assert isinstance(summary, str)
    assert summary.startswith("Summary of:")

def test_add_summary_to_db(summary_processor, mock_supabase, sample_document):
    """Test adding summary to database"""
    original_text = "Original text"
    summary = "Test summary"
    report_id = 1
    chunk_index = 0
    content_hash = "test_hash"
    
    mock_supabase.table().insert().execute.return_value.data = [{"id": 1}]
    
    result = summary_processor.add_summary_to_db(
        original_text=original_text,
        summary=summary,
        report_id=report_id,
        chunk_index=chunk_index,
        content_hash=content_hash
    )
    
    assert result == {"id": 1}
    mock_supabase.table.assert_called_with('summaries')

def test_get_summary(summary_processor, mock_supabase):
    """Test getting summary from database"""
    mock_summary = {"id": 1, "content": "test summary"}
    mock_supabase.table().select().eq().execute.return_value.data = [mock_summary]
    
    result = summary_processor.get_summary("test_hash")
    
    assert result == [mock_summary]
    mock_supabase.table.assert_called_with('summaries')
    mock_supabase.table().select.assert_called_with("*")

def test_chunk_and_embed(summary_processor, mock_pinecone_store):
    """Test chunking and embedding process"""
    summaries = ["summary1", "summary2"]
    original_chunks = ["chunk1", "chunk2"]
    metadata_base = {"report_id": 1, "report_title": "Test"}
    mock_pinecone_store.add_chunks.return_value = True
    
    result = summary_processor.chunk_and_embed(
        summaries=summaries,
        original_chunks=original_chunks,
        metadata_base=metadata_base
    )
    
    assert result is True
    mock_pinecone_store.add_chunks.assert_called_once()
    args = mock_pinecone_store.add_chunks.call_args[1]
    assert "chunks" in args
    assert "metadata" in args
    assert args["namespace"] == "report_summaries"

def test_process_new_document(summary_processor, mock_supabase, mock_pinecone_store):
    """Test processing a new document"""
    report_data = {
        "report_id": 1
    }
    mock_pinecone_store.add_chunks.return_value = True
    mock_supabase.table().select().eq().execute.return_value.data = [] # Mock no existing summaries

    with pytest.raises(Exception):
        summary_processor.process(report_data)

def test_process_existing_document(summary_processor, mock_supabase, mock_pinecone_store):
    """Test handling of already processed document"""
    report_data = {
        "report_id": 1
    }
    summaries, records = summary_processor.process(report_data)
    
    assert len(summaries) == 0
    assert len(records) == 0

def test_chunk_and_embed_error(summary_processor, mock_pinecone_store):
    """Test handling of chunking and embedding errors"""
    mock_pinecone_store.add_chunks.side_effect = Exception("Pinecone error")
    
    with pytest.raises(Exception) as exc_info:
        summary_processor.chunk_and_embed(
            summaries=["test"],
            original_chunks=["test"],
            metadata_base={"report_id": 1}
        )
    assert "Error adding summaries to Pinecone" in str(exc_info.value)

# Integration tests
def test_full_summary_workflow(mock_supabase, mock_pinecone_store, summary_processor):
    """Test the full summary workflow"""
    report_data = {
        "report_id": 1
    }
    mock_pinecone_store.add_chunks.return_value = True
    
    try:
        summaries, records = summary_processor.process(report_data)
        
        # Verify the sequence of operations
        mock_supabase.table.assert_called()
        
        assert isinstance(summaries, list)
        assert isinstance(records, list)
        
    except Exception as e:
        print(f"\nException occurred during test: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise