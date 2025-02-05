import pytest
from unittest.mock import Mock, patch, mock_open, AsyncMock
from background_process.processors import ReportProcessor, PDFDocument, PaperProcessor, Paper, TopicProcessor, TopicAssessment

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

@pytest.fixture
def topic_processor(mock_supabase):
    # Patch both OpenAI and environment variable
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}), \
         patch('openai.OpenAI') as mock_openai:
        
        # Create mock client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Create processor
        processor = TopicProcessor(
            supabase_client=mock_supabase,
            model_name="test-model"
        )
        
        # Mock evaluator
        processor.evaluator = AsyncMock()
        processor.evaluator.ainvoke.return_value = TopicAssessment(
            is_climate_relevant=True,
            analysis="Test analysis"
        )
        
        yield processor

@pytest.fixture
def paper_processor(mock_supabase, mock_pinecone_store):
    return PaperProcessor(
        supabase_client=mock_supabase,
        pinecone_store=mock_pinecone_store
    )

class TestPDFDocument:
    def test_pdf_document_creation(self):
        """Test PDFDocument dataclass creation"""
        doc = PDFDocument(
            content="test content",
            title="test title",
            content_hash="test hash"
        )
        assert doc.content == "test content"
        assert doc.title == "test title"
        assert doc.content_hash == "test hash"

class TestReportProcessor:
    def test_generate_content_hash(self, report_processor):
        """Test hash generation"""
        content = "test content"
        hash1 = report_processor.generate_content_hash(content)
        hash2 = report_processor.generate_content_hash(content)
        
        assert isinstance(hash1, str)
        assert hash1 == hash2  # Same content should produce same hash
        assert hash1 != report_processor.generate_content_hash("different content")

    @patch('background_process.processors.PdfReader')
    def test_convert_pdf_to_text(self, mock_pdf_reader, report_processor):
        """Test PDF conversion to text"""
        # Setup mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "test page content"
        mock_pdf_reader.return_value.pages = [mock_page, mock_page]
        
        # Mock open file
        with patch("builtins.open", mock_open(read_data="dummy pdf content")):
            result = report_processor.convert_pdf_to_text("testpath.pdf")
        
        assert isinstance(result, PDFDocument)
        assert "test page content" in result.content
        assert result.title == "testpath"
        assert result.content_hash == report_processor.generate_content_hash(result.content)

    def test_convert_pdf_to_text_error(self, report_processor):
        """Test handling of PDF conversion errors"""
        with pytest.raises(Exception) as exc_info:
            report_processor.convert_pdf_to_text("nonexistent.pdf")
        assert "Error converting PDF to text" in str(exc_info.value)

    def test_add_report_to_db(self, report_processor, mock_supabase):
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
        mock_supabase.table().insert.assert_called_with({
                "content": pdf_doc.content,
                "content_hash": pdf_doc.content_hash,
                "report_title": pdf_doc.title
        })

    def test_get_report(self, report_processor, mock_supabase):
        """Test getting report from database"""
        # Setup
        mock_report = {"id": 1, "content": "test"}
        mock_supabase.table().select().eq().execute.return_value.data = [mock_report]

        # Execute
        result = report_processor.get_report("test_hash")

        # Verify
        assert result == [mock_report]
        mock_supabase.table.assert_called_with('reports')
        assert mock_supabase.table().select.call_count == 2

    def test_chunk_text(self, report_processor):
        """Test text chunking"""
        # Setup
        long_text = "a" * 1000  # Text longer than chunk size

        # Execute
        chunks = report_processor.chunk_text(long_text)

        # Verify
        assert isinstance(chunks, list)
        assert len(chunks) > 1  # Should be split into multiple chunks
        assert all(len(chunk) <= 500 for chunk in chunks)  # Each chunk should be <= chunk_size

    def test_chunk_and_embed(self, report_processor, mock_pinecone_store):
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

    def test_chunk_and_embed_error(self, report_processor, mock_pinecone_store):
        """Test handling of chunking and embedding errors"""
        pdf_doc = PDFDocument(content="test", title="test", content_hash="hash")
        mock_pinecone_store.add_chunks.side_effect = Exception("Pinecone error")

        with pytest.raises(Exception) as exc_info:
            report_processor.chunk_and_embed(pdf_doc, {})
        assert "Error adding to Pinecone" in str(exc_info.value)

    def test_process_new_report(self, report_processor, mock_supabase, mock_pinecone_store):
        """Test processing a new report"""
        # Setup
        mock_table = Mock()
        mock_supabase.table.return_value = mock_table
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
            doc, record = report_processor.process({"report_path": "testpath.pdf"})

        # Verify
        assert isinstance(doc, PDFDocument)
        assert record["id"] == 1
        mock_pinecone_store.add_chunks.assert_called_once()

    def test_process_existing_report(self, report_processor, mock_supabase, mock_pinecone_store):
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
            doc, record = report_processor.process({"report_path": "testpath.pdf"})

        # Verify
        assert isinstance(doc, PDFDocument)
        assert record == existing_report
        mock_pinecone_store.add_chunks.assert_not_called()  # Shouldn't add to Pinecone for existing report

class TestPaper:
    def test_paper_creation(self):
        """Test Paper dataclass creation"""
        paper = Paper(
            abstract="test abstract",
            openalex_id="test_id",
            doi="test_doi",
            title="test title",
            content_hash="test hash"
        )
        assert paper.abstract == "test abstract"
        assert paper.openalex_id == "test_id"
        assert paper.doi == "test_doi"
        assert paper.title == "test title"
        assert paper.content_hash == "test hash"

class TestPaperProcessor:
    def test_paper_processor_initialization(self, mock_supabase, mock_pinecone_store):
        """Test PaperProcessor initialization"""
        processor = PaperProcessor(mock_supabase, mock_pinecone_store)
        assert processor.supabase == mock_supabase
        assert processor.pinecone_store == mock_pinecone_store
        
    def test_add_paper_to_db(self, paper_processor, mock_supabase):
        """Test adding paper to database"""
        paper = Paper(
            abstract="test abstract",
            openalex_id="test_id",
            doi="test_doi",
            title="test title",
            content_hash="test hash"
        )
        mock_supabase.table().insert().execute.return_value.data = [{"id": 1}]

        result = paper_processor.add_paper_to_db(paper)

        assert result == {"id": 1}
        mock_supabase.table.assert_called_with('papers')
        mock_supabase.table().insert.assert_called_with({
            "openalex_id": paper.openalex_id,
            "doi": paper.doi,
            "abstract": paper.abstract,
            "content_hash": paper.content_hash,
            "title": paper.title
        })

    def test_get_paper(self, paper_processor, mock_supabase):
        """Test getting paper from database"""
        mock_paper = {"id": 1, "abstract": "test"}
        mock_supabase.table().select().eq().execute.return_value.data = [mock_paper]

        result = paper_processor.get_paper("test_id")

        assert result == [mock_paper]
        mock_supabase.table.assert_called_with('papers')
        mock_supabase.table().select.call_count == 2

    def test_paper_chunk_and_embed(self, paper_processor, mock_pinecone_store):
        """Test chunking and embedding process for papers"""
        paper = Paper(
            abstract="test " * 200,
            openalex_id="test_id",
            doi="test_doi",
            title="test title",
            content_hash="test hash"
        )
        metadata = {"paper_id": 1}
        mock_pinecone_store.add_chunks.return_value = True

        result = paper_processor.chunk_and_embed(paper, metadata)

        assert result is True
        mock_pinecone_store.add_chunks.assert_called_once()
        args = mock_pinecone_store.add_chunks.call_args[1]
        assert "chunks" in args
        assert "metadata" in args
        assert args["namespace"] == "papers"

    def test_process_new_paper(self, paper_processor):
        """Test processing a new paper"""

        # Mock get_paper to return empty list (no existing paper)
        paper_processor.get_paper = Mock(return_value=[])
        # Mock add_paper_to_db to return a new paper record
        paper_processor.add_paper_to_db = Mock(return_value={"id": 1})
        # Mock chunk_and_embed to simulate embedding process
        paper_processor.chunk_and_embed = Mock(return_value=True)

        test_data = {
            'abstract': 'test abstract',
            'metadata': {
                'id': 'test_id',
                'doi': 'test_doi',
                'title': 'test title'
            }
        }

        paper, record = paper_processor.process(test_data)

        assert isinstance(paper, Paper)
        assert record["id"] == 1
        paper_processor.chunk_and_embed.assert_called_once()
    
class TestTopicProcessor:
    def test_topic_processor_initialization(self, mock_supabase):
        """Test TopicProcessor initialization"""
        # Patch both OpenAI and environment variable
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}), \
            patch('openai.OpenAI') as mock_openai:

            processor = TopicProcessor(mock_supabase, "test-model")
            assert processor.supabase == mock_supabase
            assert processor._tasks == set()

    def test_format_sample_works(self, topic_processor):
        """Test formatting of sample works"""
        works = [
            {"title": "Test Title 1", "abstract": "Test Abstract 1"},
            {"title": "Test Title 2", "abstract": "Test Abstract 2"}
        ]
        
        result = topic_processor.format_sample_works(works)
        
        assert "Work 1:" in result
        assert "Test Title 1" in result
        assert "Test Abstract 1" in result
        assert "Work 2:" in result
        assert "Test Title 2" in result
        assert "Test Abstract 2" in result

    def test_get_topic_assessment(self, topic_processor, mock_supabase):
        """Test getting topic assessment from database"""
        mock_assessment = {"topic_id": "test_id", "is_climate_relevant": True}
        mock_supabase.table().select().eq().execute.return_value.data = [mock_assessment]

        result = topic_processor.get_topic_assessment("test_id")

        assert result == [mock_assessment]
        mock_supabase.table.assert_called_with('openalex_topic_assessments')

    def test_save_topic_assessment_to_db(self, topic_processor, mock_supabase):
        """Test saving topic assessment to database"""
        assessment = TopicAssessment(
            is_climate_relevant=True,
            analysis="Test analysis"
        )
        mock_supabase.table().insert().execute.return_value.data = [{"id": 1}]

        result = topic_processor.save_to_db(assessment, "test_topic_id")

        assert result == {"id": 1}
        mock_supabase.table.assert_called_with('openalex_topic_assessments')
        mock_supabase.table().insert.assert_called_with({
            "topic_id": "test_topic_id",
            "is_climate_relevant": True,
            "analysis": "Test analysis"
        })

    @pytest.mark.asyncio
    async def test_process_single_topic(self, topic_processor):
        """Test processing a single topic"""

        # Mock 'get_topic_assessment' to return empty list (no existing assessment)
        topic_processor.get_topic_assessment = Mock(return_value=[])

        # Mock 'save_to_db' to return a fixed record
        mock_record = {"id": 1, "topic_id": "test_id", "is_climate_relevant": True, "analysis": "Test analysis"}
        topic_processor.save_to_db = Mock(return_value=mock_record)

        # Mock the chain resulting from 'CLIMATE_RELEVANCE_PROMPT | self.evaluator'
        with patch('background_process.processors.CLIMATE_RELEVANCE_PROMPT') as mock_prompt:
            mock_chain = Mock()
            mock_prompt.__or__.return_value = mock_chain

            # Mock the 'ainvoke' method of the chain to return a TopicAssessment
            mock_chain.ainvoke = AsyncMock(return_value=TopicAssessment(
                is_climate_relevant=True,
                analysis="Test analysis"
            ))

            test_data = {
                'topic_id': 'test_id',
                'topic_name': 'Test Topic',
                'topic_description': 'Test Description',
                'sample_works': [
                    {'title': 'Test Paper', 'abstract': 'Test Abstract'}
                ]
            }

            assessment, record = await topic_processor.process_single_topic(test_data)

        # Assertions
        assert isinstance(assessment, TopicAssessment)
        assert assessment.is_climate_relevant is True
        assert assessment.analysis == "Test analysis"
        assert record["id"] == 1

        # Verify that 'save_to_db' was called with the correct arguments
        topic_processor.save_to_db.assert_called_once_with(assessment, 'test_id')

        # Ensure 'ainvoke' was called once
        assert mock_chain.ainvoke.call_count == 1

    @pytest.mark.asyncio
    @patch('background_process.processors.CLIMATE_RELEVANCE_PROMPT')
    async def test_process_batch_topics(self, mock_prompt, topic_processor):
        """Test processing multiple topics in batch"""

        # Mock 'get_topic_assessment' to return empty list
        topic_processor.get_topic_assessment = Mock(return_value=[])

        # Mock 'save_to_db' to return a fixed record
        mock_record = {"id": 1, "is_climate_relevant": True, "analysis": "Test analysis"}
        topic_processor.save_to_db = Mock(return_value=mock_record)

        # Mock the chain resulting from 'CLIMATE_RELEVANCE_PROMPT | self.evaluator'
        mock_chain = Mock()
        mock_prompt.__or__.return_value = mock_chain

        # Mock the 'ainvoke' method of the chain to return a TopicAssessment
        mock_chain.ainvoke = AsyncMock(return_value=TopicAssessment(
            is_climate_relevant=True,
            analysis="Test analysis"
        ))

        test_topics = [
            {
                'topic_id': 'test_id_1',
                'topic_name': 'Test Topic 1',
                'topic_description': 'Test Description 1',
                'sample_works': [{'title': 'Test Paper 1', 'abstract': 'Test Abstract 1'}]
            },
            {
                'topic_id': 'test_id_2',
                'topic_name': 'Test Topic 2',
                'topic_description': 'Test Description 2',
                'sample_works': [{'title': 'Test Paper 2', 'abstract': 'Test Abstract 2'}]
            }
        ]

        results = await topic_processor.process_batch(test_topics)

        assert len(results) == 2
        for assessment, record in results:
            assert isinstance(assessment, TopicAssessment)
            assert assessment.is_climate_relevant is True
            assert assessment.analysis == "Test analysis"
            assert record["id"] == 1

        # Ensure 'ainvoke' was called twice (once for each topic)
        assert mock_chain.ainvoke.call_count == 2