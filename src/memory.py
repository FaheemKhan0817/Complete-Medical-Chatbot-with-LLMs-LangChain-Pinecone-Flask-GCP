from langchain_community.chat_message_histories import ChatMessageHistory

# Initialize a global chat history object.
# For a real-world application, you would replace this with a more robust
# session-based or database-backed history management system to handle
# multiple users simultaneously.
chat_history = ChatMessageHistory()

def get_chat_history():
    """Returns the current chat history."""
    return chat_history

def add_to_history(user_message: str, ai_message: str):
    """Adds a user message and an AI response to the history."""
    chat_history.add_user_message(user_message)
    chat_history.add_ai_message(ai_message)

def clear_history():
    """Clears the chat history for a new session."""
    chat_history.clear()