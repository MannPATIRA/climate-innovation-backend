import pytest
from unittest.mock import Mock, patch, mock_open
import os
from background_process.processors import ReportProcessor, PDFDocument, PaperProcessor
from PyPDF2 import PdfReader

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
def report_processor(mock_supabase, mock_pinecone_store):
    return ReportProcessor(
        supabase_client=mock_supabase,
        pinecone_store=mock_pinecone_store
    )

# Test PDFDocument
def test_pdf_document_creation():
    """Test PDFDocument dataclass creation"""
    doc = PDFDocument(
        content="test content",
        title="test title",
        content_hash="test hash"
    )
    assert doc.content == "test content"
    assert doc.title == "test title"
    assert doc.content_hash == "test hash"

# Tests for ReportProcessor
def test_generate_content_hash(report_processor):
    """Test hash generation"""
    content = "test content"
    hash1 = report_processor.generate_content_hash(content)
    hash2 = report_processor.generate_content_hash(content)
    
    assert isinstance(hash1, str)
    assert hash1 == hash2  # Same content should produce same hash
    assert hash1 != report_processor.generate_content_hash("different content")

@patch('PyPDF2.PdfReader')
def test_convert_pdf_to_text(mock_pdf_reader, report_processor):
    """Test PDF conversion to text"""
    # Setup mock PDF reader
    mock_page = Mock()
    mock_page.extract_text.return_value = "test page content"
    mock_pdf_reader.return_value.pages = [mock_page, mock_page]
    
    # Mock open file
    with patch("builtins.open", mock_open(read_data="dummy pdf content")):
        result = report_processor.convert_pdf_to_text("test.pdf")
    
    assert isinstance(result, PDFDocument)
    assert "test page content" in result.content
    assert result.title == "test"
    assert result.content_hash == report_processor.generate_content_hash(result.content)

def test_add_report_to_db(report_processor, mock_supabase):
    """Test adding report to database"""
    # Setup
    pdf_doc = PDFDocument(
        content="test content",
        title="test title",
        content_hash="test hash"
    )
    mock_supabase.table().insert().execute.return_value.data = [{"id": 1}]

    # Execute
    result = report_processor.add_report_to_db(pdf_doc)

    # Verify
    assert result == {"id": 1}
    mock_supabase.table.assert_called_with('reports')
    mock_supabase.table().insert.assert_called_once()

def test_get_report(report_processor, mock_supabase):
    """Test getting report from database"""
    # Setup
    mock_report = {"id": 1, "content": "test"}
    mock_supabase.table().select().eq().execute.return_value.data = [mock_report]

    # Execute
    result = report_processor.get_report("test_hash")

    # Verify
    assert result == [mock_report]
    mock_supabase.table.assert_called_with('reports')
    mock_supabase.table().select.assert_called_once()

def test_chunk_text(report_processor):
    """Test text chunking"""
    # Setup
    long_text = "a" * 1000  # Text longer than chunk size

    # Execute
    chunks = report_processor.chunk_text(long_text)

    # Verify
    assert isinstance(chunks, list)
    assert len(chunks) > 1  # Should be split into multiple chunks
    assert all(len(chunk) <= 500 for chunk in chunks)  # Each chunk should be <= chunk_size

def test_chunk_and_embed(report_processor, mock_pinecone_store):
    """Test chunking and embedding process"""
    # Setup
    pdf_doc = PDFDocument(
        content="test " * 200,  # Long enough to create multiple chunks
        title="test",
        content_hash="hash"
    )
    metadata = {"report_id": 1}
    mock_pinecone_store.add_chunks.return_value = True

    # Execute
    result = report_processor.chunk_and_embed(pdf_doc, metadata)

    # Verify
    assert result is True
    mock_pinecone_store.add_chunks.assert_called_once()
    args = mock_pinecone_store.add_chunks.call_args[1]
    assert "chunks" in args
    assert "metadata" in args
    assert args["namespace"] == "reports"

def test_process_new_report(report_processor, mock_supabase, mock_pinecone_store):
    """Test processing a new report"""
    # Setup
    mock_supabase.table().select().eq().execute.return_value.data = []  # No existing report
    mock_supabase.table().insert().execute.return_value.data = [{"id": 1}]
    mock_pinecone_store.add_chunks.return_value = True

    # Mock PDF conversion
    with patch.object(report_processor, 'convert_pdf_to_text') as mock_convert:
        mock_convert.return_value = PDFDocument(
            content="test content",
            title="test",
            content_hash="hash"
        )

        # Execute
        doc, record = report_processor.process({"report_path": "test.pdf"})

    # Verify
    assert isinstance(doc, PDFDocument)
    assert record["id"] == 1
    mock_pinecone_store.add_chunks.assert_called_once()

def test_process_existing_report(report_processor, mock_supabase, mock_pinecone_store):
    """Test processing an already existing report"""
    # Setup
    existing_report = {"id": 1, "content": "test"}
    mock_supabase.table().select().eq().execute.return_value.data = [existing_report]

    # Mock PDF conversion
    with patch.object(report_processor, 'convert_pdf_to_text') as mock_convert:
        mock_convert.return_value = PDFDocument(
            content="test content",
            title="test",
            content_hash="hash"
        )

        # Execute
        doc, record = report_processor.process({"report_path": "test.pdf"})

    # Verify
    assert isinstance(doc, PDFDocument)
    assert record == existing_report
    mock_pinecone_store.add_chunks.assert_not_called()  # Shouldn't add to Pinecone for existing report

# Error handling tests
def test_convert_pdf_to_text_error(report_processor):
    """Test handling of PDF conversion errors"""
    with pytest.raises(Exception) as exc_info:
        report_processor.convert_pdf_to_text("nonexistent.pdf")
    assert "Error converting PDF to text" in str(exc_info.value)

def test_chunk_and_embed_error(report_processor, mock_pinecone_store):
    """Test handling of chunking and embedding errors"""
    pdf_doc = PDFDocument(content="test", title="test", content_hash="hash")
    mock_pinecone_store.add_chunks.side_effect = Exception("Pinecone error")

    with pytest.raises(Exception) as exc_info:
        report_processor.chunk_and_embed(pdf_doc, {})
    assert "Error adding to Pinecone" in str(exc_info.value)

# Test PaperProcessor
def test_paper_processor_initialization(mock_supabase, mock_pinecone_store):
    """Test PaperProcessor initialization"""
    processor = PaperProcessor(mock_supabase, mock_pinecone_store)
    assert processor.supabase == mock_supabase
    assert processor.pinecone_store == mock_pinecone_store