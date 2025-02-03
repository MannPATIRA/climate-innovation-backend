import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from backend_server.query_processors import MockQueryProcessor, QueryProcessor

# Test fixtures
@pytest.fixture
def mock_processor():
    return MockQueryProcessor()

@pytest.fixture
def query_processor():
    with patch('backend_server.query_processors.ChatOpenAI') as mock_chat:
        with patch('backend_server.query_processors.PineconeStore') as mock_pinecone:
            processor = QueryProcessor()
            # Configure the mocks properly
            mock_chat.return_value = AsyncMock()
            mock_pinecone.return_value = MagicMock()
            processor.chain = AsyncMock()
            yield processor

class TestMockQueryProcessor:
    @pytest.mark.asyncio
    async def test_mock_process_stream_yields_content(self, mock_processor):
        callback = AsyncMock()
        query = "What are the main climate challenges?"
        chat_history = []
        
        chunks = []
        async for chunk in mock_processor.process_stream(query, chat_history, callback):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        full_response = ''.join(chunks)
        assert "Climate Challenges" in full_response
        assert "Heat Waves" in full_response
        assert "Wildfires" in full_response
        assert "Floods" in full_response

    @pytest.mark.asyncio
    async def test_mock_process_stream_callback(self, mock_processor):
        callback = AsyncMock()
        query = "What are the main climate challenges?"
        chat_history = []
        
        async for _ in mock_processor.process_stream(query, chat_history, callback):
            pass
        
        # Wait for callback to be called
        await asyncio.sleep(0.1)
        assert callback.called

    @pytest.mark.asyncio
    async def test_mock_process_stream_json_format(self, mock_processor):
        callback = AsyncMock()
        query = "What are the main climate challenges?"
        chat_history = []
        
        full_response = ""
        async for chunk in mock_processor.process_stream(query, chat_history, callback):
            full_response += chunk
        
        try:
            # Find the JSON structure by looking for the last occurrence of "{'topics':"
            json_start = full_response.rindex("{'topics':")
            json_str = full_response[json_start:].strip()
            
            # Clean up the JSON string
            json_str = (
                json_str
                .replace("'", '"')  # Replace single quotes with double quotes
                .replace('\n', '')  # Remove newlines
                .strip()           # Remove any extra whitespace
            )
            
            # For debugging
            print(f"Extracted JSON string: {json_str}")
            
            data = json.loads(json_str)
            
            # Verify the structure
            assert "topics" in data, "Missing 'topics' key in JSON"
            assert isinstance(data["topics"], list), "'topics' should be a list"
            assert len(data["topics"]) > 0, "'topics' list is empty"
            
            # Verify each topic has required fields
            for topic in data["topics"]:
                assert "topic" in topic, "Missing 'topic' field in topic"
                assert "source" in topic, "Missing 'source' field in topic"
                assert isinstance(topic["topic"], str), "'topic' should be a string"
                assert isinstance(topic["source"], str), "'source' should be a string"
            
        except ValueError as e:
            pytest.fail(f"Could not find JSON structure: {str(e)}\nFull response: {full_response}")
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON format: {str(e)}\nJSON string: {json_str}\nFull response: {full_response}")

class TestQueryProcessor:
    @pytest.fixture
    def mock_pinecone_response(self):
        class MockMatch:
            def __init__(self, content, title):
                self.metadata = {
                    "content": content,
                    "report_title": title
                }
        
        return [
            MockMatch("Test content 1", "Report 1"),
            MockMatch("Test content 2", "Report 2"),
            MockMatch("Test content 3", "Report 3")
        ]

    @pytest.mark.asyncio
    async def test_get_relevant_chunks(self, query_processor, mock_pinecone_response):
        # Setup the mock to return the response directly
        query_processor.pinecone_store.query_chunk = MagicMock(return_value=mock_pinecone_response)
        
        chunks = query_processor._get_relevant_chunks("test query")
        
        assert "Test content 1" in chunks
        assert "Test content 2" in chunks
        assert "Test content 3" in chunks
        assert "Report 1" in chunks
        assert "Report 2" in chunks
        assert "Report 3" in chunks

    @pytest.mark.asyncio
    async def test_process_stream(self, query_processor):
        # Setup mock response as an async iterator
        mock_response = ["Test ", "response ", "with JSON ", "{'topics': [{'topic': 'Test Topic', 'source': 'Test Source'}]}"]
        
        async def mock_astream(*args, **kwargs):
            for chunk in mock_response:
                yield chunk

        # Configure the mock
        query_processor.chain.astream = mock_astream
        
        callback = AsyncMock()
        query = "test query"
        chat_history = []
        
        response = ""
        async for chunk in query_processor.process_stream(query, chat_history, callback):
            response += chunk
        
        assert "Test response" in response
        # Don't call the callback manually, it should be called by the process_stream method
        assert callback.call_count == 1
        callback.assert_called_once_with(response)

    def test_create_chain(self, query_processor):
        chain = query_processor._create_chain()
        assert chain is not None

# Helper function to create an async iterator
class AsyncIterator:
    def __init__(self, items):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item