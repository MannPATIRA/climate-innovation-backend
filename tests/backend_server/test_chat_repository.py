import pytest
from unittest.mock import Mock, patch
from backend_server.chat_repository import ChatRepository, ChatNotFoundError, InvalidSourceTypeError

@pytest.fixture
def mock_supabase():
    return Mock()

@pytest.fixture
def chat_repository(mock_supabase):
    return ChatRepository(mock_supabase)

def test_get_chat_success(chat_repository, mock_supabase):
    """Test successful chat retrieval"""
    # Setup
    mock_response = Mock()
    mock_response.data = {"id": 1, "type": "report"}
    mock_supabase.table().select().eq().single().execute.return_value = mock_response
    
    # Execute
    result = chat_repository.get_chat(1)
    
    # Verify
    assert result == {"id": 1, "type": "report"}
    mock_supabase.table.assert_called_with("chats")

def test_get_chat_not_found(chat_repository, mock_supabase):
    """Test chat retrieval when chat doesn't exist"""
    # Setup
    mock_response = Mock()
    mock_response.data = None
    mock_supabase.table().select().eq().single().execute.return_value = mock_response
    
    # Execute and verify
    with pytest.raises(ChatNotFoundError):
        chat_repository.get_chat(999)

def test_create_chat_success(chat_repository, mock_supabase):
    """Test successful chat creation"""
    # Setup
    mock_response = Mock()
    mock_response.data = [{"id": 1, "type": "report"}]
    mock_supabase.table().insert().execute.return_value = mock_response
    
    # Execute
    result = chat_repository.create_chat("reports")
    
    # Verify
    assert result == {"id": 1, "type": "report"}
    mock_supabase.table().insert.assert_called_with({"type": "report"})

def test_create_chat_invalid_type(chat_repository):
    """Test chat creation with invalid source type"""
    with pytest.raises(InvalidSourceTypeError):
        chat_repository.create_chat("invalid_type")

def test_get_chat_history(chat_repository, mock_supabase):
    """Test retrieving chat history"""
    # Setup
    expected_messages = [
        {"id": 1, "content": "Hello", "order": 1},
        {"id": 2, "content": "Hi", "order": 2}
    ]
    mock_response = Mock()
    mock_response.data = expected_messages
    mock_supabase.table().select().eq().order().execute.return_value = mock_response
    
    # Execute
    result = chat_repository.get_chat_history("1")
    
    # Verify
    assert result == expected_messages
    mock_supabase.table.assert_called_with("chat_messages")

def test_add_message(chat_repository, mock_supabase):
    """Test adding a message to chat"""
    # Setup
    mock_response = Mock()
    mock_response.data = [{"id": 1, "content": "Hello"}]
    mock_supabase.table().insert().execute.return_value = mock_response
    
    # Execute
    result = chat_repository.add_message("1", "Hello", 1, True)
    
    # Verify
    mock_supabase.table().insert.assert_called_with({
        "content": "Hello",
        "order": 1,
        "user_message": True,
        "chat_id": "1"
    })

def test_update_message_count(chat_repository, mock_supabase):
    """Test updating message count"""
    # Setup
    mock_response = Mock()
    mock_response.data = [{"id": 1, "message_count": 5}]
    mock_supabase.table().update().eq().execute.return_value = mock_response
    
    # Execute
    result = chat_repository.update_message_count("1", 5)
    
    # Verify
    mock_supabase.table().update.assert_called_with({"message_count": 5})
    mock_supabase.table().update().eq.assert_called_with("id", "1")