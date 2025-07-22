import os
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

# Import Google and Pinecone modules
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Import LangChain modules for building the RAG chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import your custom system prompt and memory functions
from src.prompt import system_prompt
from src.memory import get_chat_history, add_to_history, clear_history

# Initialize Flask App
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Configuration and Initialization ---

# Set API Keys from environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize the embedding model for querying
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type="RETRIEVAL_QUERY",
    google_api_key=GOOGLE_API_KEY
)

# Connect to the existing Pinecone index and create the base retriever
index_name = "medical-chatbot"
vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Initialize the Google Gemini LLM
chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

# --- Create History-Aware RAG Chain ---

# 1. Contextualizer Prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    chat_model, base_retriever, contextualize_q_prompt
)

# 2. Main Answering Prompt
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)

# 3. Combine them into the final RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- Flask Routes ---

@app.route("/")
def index():
    # Clear the history for a new session when the page is reloaded
    clear_history()
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    user_input = msg
    print(f"User Input: {user_input}")

    # Get current chat history
    current_chat_history = get_chat_history()

    # Invoke the RAG chain with the input AND the history
    response = rag_chain.invoke({
        "input": user_input,
        "chat_history": current_chat_history.messages
    })
    
    # Update the history with the latest turn
    add_to_history(user_input, response["answer"])
    
    print(f"Response: {response['answer']}")
    return str(response["answer"])

# --- Main Execution ---

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
