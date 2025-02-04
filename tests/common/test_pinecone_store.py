import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from pinecone import ServerlessSpec
from common.pinecone_store import PineconeStore

# Test data
TEST_INDEX_NAME = "test-index"
TEST_MODEL = "multilingual-e5-large"
TEST_API_KEY = "test-api-key"
TEST_VECTORS = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
TEST_METADATA = [{"text": "test1"}, {"text": "test2"}]
TEST_IDS = ["id1", "id2"]
TEST_CHUNKS = ["test chunk 1", "test chunk 2"]

@pytest.fixture
def mock_pinecone():
    with patch('common.pinecone_store.Pinecone') as mock:
        # Mock index description
        mock.return_value.describe_index.return_value = {'host': 'test-host'}
        # Mock index instance
        mock_index = MagicMock()
        mock.return_value.Index.return_value = mock_index
        yield mock

@pytest.fixture
def pinecone_store(mock_pinecone):
    with patch.dict(os.environ, {'PINECONE_API_KEY': TEST_API_KEY}):
        store = PineconeStore(TEST_INDEX_NAME, TEST_MODEL)
        yield store

class TestPineconeStore:
    def test_init_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="PINECONE_API_KEY not found in environment variables"):
                PineconeStore(TEST_INDEX_NAME)

    def test_init_creates_index_if_not_exists(self, mock_pinecone):
        with patch.dict(os.environ, {'PINECONE_API_KEY': TEST_API_KEY}):
            mock_pinecone.return_value.has_index.return_value = False
            store = PineconeStore(TEST_INDEX_NAME)
            
            mock_pinecone.return_value.create_index.assert_called_once_with(
                name=TEST_INDEX_NAME,
                dimension=1024,
                metric='cosine',
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

    def test_add_embeddings_success(self, pinecone_store):
        result = pinecone_store.add_embeddings(TEST_VECTORS, TEST_METADATA, TEST_IDS)
        assert result is True
        pinecone_store.index.upsert.assert_called_once()

    def test_add_embeddings_failure(self, pinecone_store):
        pinecone_store.index.upsert.side_effect = Exception("Test error")
        result = pinecone_store.add_embeddings(TEST_VECTORS, TEST_METADATA, TEST_IDS)
        assert result is False

    def test_query_embeddings_success(self, pinecone_store):
        mock_results = MagicMock()
        mock_results.matches = [{"id": "test", "score": 0.9}]
        pinecone_store.index.query.return_value = mock_results
        
        results = pinecone_store.query_embeddings([0.1, 0.2, 0.3])
        assert results == mock_results.matches
        pinecone_store.index.query.assert_called_once()

    def test_query_embeddings_failure(self, pinecone_store):
        pinecone_store.index.query.side_effect = Exception("Test error")
        results = pinecone_store.query_embeddings([0.1, 0.2, 0.3])
        assert results == []

    def test_delete_embeddings_success(self, pinecone_store):
        result = pinecone_store.delete_embeddings(TEST_IDS)
        assert result is True
        pinecone_store.index.delete.assert_called_once_with(ids=TEST_IDS)

    def test_delete_embeddings_failure(self, pinecone_store):
        pinecone_store.index.delete.side_effect = Exception("Test error")
        result = pinecone_store.delete_embeddings(TEST_IDS)
        assert result is False

    @patch('common.pinecone_store.hashlib.sha256')
    def test_add_chunks_success(self, mock_hash, pinecone_store):
        # Mock hash function
        mock_hash.return_value.hexdigest.return_value = "test_hash"
        
        # Mock embedding generation
        mock_embeddings = [{'values': [0.1, 0.2, 0.3]} for _ in TEST_CHUNKS]
        pinecone_store.pc.inference.embed.return_value = mock_embeddings
        
        result = pinecone_store.add_chunks(TEST_CHUNKS, TEST_METADATA)
        assert result is True
        pinecone_store.index.upsert.assert_called_once()

    def test_add_chunks_failure(self, pinecone_store):
        pinecone_store.pc.inference.embed.side_effect = Exception("Test error")
        result = pinecone_store.add_chunks(TEST_CHUNKS)
        assert result is False

    def test_query_chunk_success(self, pinecone_store):
        # Mock embedding generation
        mock_embedding = [{'values': [0.1, 0.2, 0.3]}]
        pinecone_store.pc.inference.embed.return_value = mock_embedding
        
        # Mock query results
        mock_results = MagicMock()
        mock_results.matches = [{"id": "test", "score": 0.9}]
        pinecone_store.index.query.return_value = mock_results
        
        results = pinecone_store.query_chunk("test query")
        assert results == mock_results.matches

    def test_query_chunk_failure(self, pinecone_store):
        pinecone_store.pc.inference.embed.side_effect = Exception("Test error")
        results = pinecone_store.query_chunk("test query")
        assert results == []

    @patch('common.pinecone_store.Pinecone')
    def test_delete_index_success(self, mock_pinecone):
        with patch.dict(os.environ, {'PINECONE_API_KEY': TEST_API_KEY}):
            result = PineconeStore.delete_index(TEST_INDEX_NAME)
            assert result is True
            mock_pinecone.return_value.delete_index.assert_called_once_with(TEST_INDEX_NAME)

    @patch('common.pinecone_store.Pinecone')
    def test_delete_index_failure(self, mock_pinecone):
        with patch.dict(os.environ, {'PINECONE_API_KEY': TEST_API_KEY}):
            mock_pinecone.return_value.delete_index.side_effect = Exception("Test error")
            result = PineconeStore.delete_index(TEST_INDEX_NAME)
            assert result is False