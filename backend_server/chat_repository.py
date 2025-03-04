from typing import List, Optional
from supabase import Client

class ChatNotFoundError(Exception):
    pass

class InvalidSourceTypeError(Exception):
    pass

class ChatRepository:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client

    def get_chat(self, chat_id: int):
        result = self.supabase.table("chats")\
            .select("*")\
            .eq("id", chat_id)\
            .single()\
            .execute()
        
        if not result.data:
            raise ChatNotFoundError(f"Chat with ID {chat_id} not found")
            
        return result.data

    def create_chat(self, source_type: str, user_email: str):
        if source_type not in ["reports", "papers"]:
            raise InvalidSourceTypeError("Invalid source type. Must be 'reports' or 'papers'")
        
        # Delete all chats with message_count = 0
        self.supabase.table("chats")\
            .delete()\
            .eq("message_count", 0)\
            .execute()
        
        data = self.supabase.table("chats").insert({
            "type": source_type.rstrip('s'),  # Convert 'reports' to 'report', 'papers' to 'paper'
            "user_email": user_email
        }).execute()
        
        return data.data[0]

    def get_chat_history(self, chat_id: str):
        result = self.supabase.table("chat_messages")\
            .select("*")\
            .eq("chat_id", chat_id)\
            .order("order")\
            .execute()
        
        return result.data

    def add_message(self, chat_id: str, content: str, order: int, is_user_message: bool):
        return self.supabase.table("chat_messages").insert({
            "content": content,
            "order": order,
            "user_message": is_user_message,
            "chat_id": chat_id
        }).execute()

    def update_message_count(self, chat_id: str, count: int):
        return self.supabase.table("chats")\
            .update({"message_count": count})\
            .eq("id", chat_id)\
            .execute()

    def get_all_chats(self, user_email: str):
        result = self.supabase.table("chats")\
            .select("*")\
            .eq("user_email", user_email)\
            .order("created_at", desc=True)\
            .execute()
        
        return result.data

    def update_chat_name(self, chat_id: str, name: str):
        """Update the name of a chat"""
        result = self.supabase.table("chats").update(
            {"name": name}
        ).eq("id", chat_id).execute()
        
        if not result.data:
            raise ChatNotFoundError(f"Chat with ID {chat_id} not found")
        
        return result.data[0]