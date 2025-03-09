from typing import List, Optional
from supabase import AsyncClient


class ChatNotFoundError(Exception):
    """ Exception to be thrown when we do not find a given chat """
    pass


class InvalidSourceTypeError(Exception):
    """ Exception to be thrown when we do not have a valid source type (i.e., Paper or Report) """
    pass


class ChatRepository:
    """
    Class to deal with Supabase chat representation, including creating, getting chats and messages within them

    Attributes
    ----------
    supabase : Client

    Methods
    ----------
    get_chat()
    create_chat()
    get_chat_history()
    add_message()
    update_message_count()
    get_all_chats()
    update_chat_name()
    """

    def __init__(self, supabase_client: AsyncClient):
        self.supabase = supabase_client

    async def get_chat(self, chat_id: int):
        """
        Returns a chat (represented as a Dict / JSON)
        Parameters
        ----------
        chat_id: Int

        Returns
        -------
        Dict - The data about the chat that was queried, including name, created_at and message_count
        """

        # Query supabase for chat
        result = await self.supabase.table("chats") \
            .select("*") \
            .eq("id", chat_id) \
            .single() \
            .execute()

        # Raise error if chat not found
        if not result.data:
            raise ChatNotFoundError(f"Chat with ID {chat_id} not found")

        return result.data

    async def create_chat(self, source_type: str, user_email: str):
        """
        Creates a chat given source and email
        Parameters
        ----------
        source_type: str, one of ["reports", "papers"] to signify the type of chat
        user_email: str, email id of user who created the chat

        Returns
        -------
        bool - Success value
        """

        # Raise error if invalid source_type
        if source_type not in ["reports", "papers"]:
            raise InvalidSourceTypeError("Invalid source type. Must be 'reports' or 'papers'")

        # Delete all chats with message_count = 0
        await self.supabase.table("chats") \
            .delete() \
            .eq("message_count", 0) \
            .execute()

        # Create new chat in the table
        data = await self.supabase.table("chats").insert({
            "type": source_type.rstrip('s'),  # Convert 'reports' to 'report', 'papers' to 'paper'
            "user_email": user_email
        }).execute()

        return data.data[0]

    async def get_chat_history(self, chat_id: str):
        """
        Returns a given chat's message history
        Parameters
        ----------
        chat_id: str, the id of the chat whose history we want

        Returns
        -------
        Dict - ordered messages
        """
        result = await self.supabase.table("chat_messages") \
            .select("*") \
            .eq("chat_id", chat_id) \
            .order("order") \
            .execute()

        return result.data

    async def add_message(self, chat_id: str, content: str, order: int, is_user_message: bool):
        """
        Adds a message to a chat
        Parameters
        ----------
        chat_id: str, id of the chat to add a message to
        content: str, message content
        order: int, the ordinal number of the message within the chat
        is_user_message: bool, whether message was sent by user or bot

        Returns
        -------
        Bool - success value
        """
        return await self.supabase.table("chat_messages").insert({
            "content": content,
            "order": order,
            "user_message": is_user_message,
            "chat_id": chat_id
        }).execute()

    async def update_message_count(self, chat_id: str, count: int):
        """
        Change value of message count for a given chat
        Parameters
        ----------
        chat_id: str, chat whose message count to update
        count: int, new message count

        Returns
        -------
        Bool - success value
        """
        return await self.supabase.table("chats") \
            .update({"message_count": count}) \
            .eq("id", chat_id) \
            .execute()

    async def get_all_chats(self, user_email: str):
        """
        Returns all the chats for a given user
        Parameters
        ----------
        user_email: str, email id of user whose chats we want

        Returns
        -------
        List[Dict] - list of all the chats for this user
        """
        result = await self.supabase.table("chats") \
            .select("*") \
            .eq("user_email", user_email) \
            .order("created_at", desc=True) \
            .execute()

        return result.data

    async def update_chat_name(self, chat_id: str, name: str):
        """
        Updates the name of a given chat
        Parameters
        ----------
        chat_id: str, id of chat whose name we want to update
        name: str, new name for the chat

        Returns
        -------
        Bool - success value
        """

        # Update the chat name
        result = await self.supabase.table("chats").update(
            {"name": name}
        ).eq("id", chat_id).execute()

        # If chat not found raise error
        if not result.data:
            raise ChatNotFoundError(f"Chat with ID {chat_id} not found")

        return result.data[0]
