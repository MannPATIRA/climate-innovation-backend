import pytest
from unittest.mock import Mock, patch
from background_process.processors.summary_processor import Summarizer, SummaryProcessor
from background_process.processors.report_processor import PDFDocument

# Mock Summarizer Implementation
class MockSummarizer(Summarizer):
    def generate_summary(self, text: str) -> str:
        return f"Summary of: {text[:50]}..."

# Fixtures
@pytest.fixture
def mock_supabase():
    mock = Mock()
    # Setup mock table responses
    mock_table = Mock()
    mock.table.return_value = mock_table
    mock_table.insert.return_value = mock_table
    mock_table.select.return_value = mock_table
    mock_table.eq.return_value = mock_table
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

def test_generate_summary(summary_processor):
    """Test summary generation"""
    text = "This is a test document that needs to be summarized."
    summary = summary_processor.generate_summary(text)
    assert isinstance(summary, str)
    assert summary.startswith("Summary of:")

def test_add_summary_to_db(summary_processor, mock_supabase, sample_document):
    """Test adding summary to database"""
    summary = "Test summary"
    content_hash = "test_hash"
    report_id = 1
    
    mock_supabase.table().insert().execute.return_value.data = [{"id": 1}]
    
    result = summary_processor.add_summary_to_db(
        summary=summary,
        report_id=report_id,
        document=sample_document,
        content_hash=content_hash
    )
    
    assert result == {"id": 1}
    mock_supabase.table.assert_called_with('summaries')
    mock_supabase.table().insert.assert_called_with({
        "content": summary,
        "content_hash": content_hash,
        "report_id": report_id,
        "title": sample_document.title,
    })


def test_get_summary(summary_processor, mock_supabase):
    """Test getting summary from database"""
    mock_summary = {"id": 1, "content": "test summary"}
    mock_supabase.table().select().eq().execute.return_value.data = [mock_summary]
    
    result = summary_processor.get_summary("test_hash")
    
    assert result == [mock_summary]
    mock_supabase.table.assert_called_with('summaries')
    mock_supabase.table().select.assert_called_with("*")

def test_chunk_text(summary_processor):
    """Test text chunking"""
    long_text = "a" * 1000  # Text longer than chunk size
    chunks = summary_processor.chunk_text(long_text)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)

def test_chunk_and_embed(summary_processor, mock_pinecone_store):
    """Test chunking and embedding process"""
    summary = "test " * 200  # Long enough to create multiple chunks
    metadata = {"summary_id": 1}
    mock_pinecone_store.add_chunks.return_value = True
    
    result = summary_processor.chunk_and_embed(summary, metadata)
    
    assert result is True
    mock_pinecone_store.add_chunks.assert_called_once()
    args = mock_pinecone_store.add_chunks.call_args[1]
    assert "chunks" in args
    assert "metadata" in args
    assert args["namespace"] == "summaries"

@patch.object(SummaryProcessor, 'get_summary', return_value=[])
def test_summarize_new_summary(sample_document, sample_report_record, mock_supabase, mock_pinecone_store, summary_processor):
    """Test summarizing a new document"""
    # Setup
    mock_supabase.table().select().eq().execute.return_value.data = []  # No existing summary
    mock_supabase.table().insert().execute.return_value.data = [{"id": 1}]
    mock_pinecone_store.add_chunks.return_value = True
    
    # Execute
    summary_processor.summarize(sample_document, sample_report_record, "testpath.pdf")
    
    # Verify
    assert mock_supabase.table().insert.call_count == 2
    mock_pinecone_store.add_chunks.assert_called_once()

def test_summarize_existing_summary(summary_processor, sample_document, sample_report_record, mock_supabase, mock_pinecone_store):
    """Test handling of already existing summary"""
    # Setup
    mock_supabase.table().select().eq().execute.return_value.data = [{"id": 1}]  # Existing summary
    
    # Execute
    summary_processor.summarize(sample_document, sample_report_record, "testpath.pdf")
    
    # Verify
    mock_supabase.table().insert.assert_not_called()
    mock_pinecone_store.add_chunks.assert_not_called()

def test_chunk_and_embed_error(summary_processor, mock_pinecone_store):
    """Test handling of chunking and embedding errors"""
    mock_pinecone_store.add_chunks.side_effect = Exception("Pinecone error")
    
    with pytest.raises(Exception) as exc_info:
        summary_processor.chunk_and_embed("test summary", {})
    assert "Error adding summary to Pinecone" in str(exc_info.value)

# Integration tests
@patch.object(SummaryProcessor, 'get_summary', return_value=[])
def test_full_summary_workflow(sample_document, sample_report_record, mock_supabase, mock_pinecone_store, summary_processor):
    """Test the full summary workflow"""
    # Setup
    mock_supabase.table().select().eq().execute.return_value.data = []  # No existing summary
    mock_supabase.table().insert().execute.return_value.data = [{"id": 1}]
    mock_pinecone_store.add_chunks.return_value = True
    
    try:
        summary_processor.summarize(sample_document, sample_report_record, "testpath.pdf")
        
        # Verify the sequence of operations
        # Verify Supabase operations
        mock_supabase.table.assert_called()
        assert mock_supabase.table().insert.call_count == 2
        
        # Verify Pinecone operations
        mock_pinecone_store.add_chunks.assert_called_once()


        
        # Verify the call arguments to Pinecone
        actual_call = mock_pinecone_store.add_chunks.call_args
        assert actual_call is not None, "add_chunks was not called"
        
        # Print actual call details
        args, kwargs = actual_call
        
        # Verify the namespace
        assert kwargs["namespace"] == "summaries"
        assert isinstance(kwargs["chunks"], list)
        assert isinstance(kwargs["metadata"], list)
        
    except Exception as e:
        print(f"\nException occurred during test: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise