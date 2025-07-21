import os
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

# Import the embeddings object directly from your helper file
from src.helper import embeddings

# Import Google and Pinecone modules
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

# Import LangChain modules for building the RAG chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Import your custom system prompt
from src.prompt import system_prompt

# Initialize Flask App
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Configuration and Initialization ---

# Set API Keys from environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Connect to the existing Pinecone index using the imported embeddings
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings  # Using the imported embeddings object
)

# Create the retriever
retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# Initialize the Google Gemini LLM
chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

# Create the prompt from your prompt file
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the RAG chain
Youtube_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, Youtube_chain)

# --- Flask Routes ---

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    user_input = msg
    print(f"User Input: {user_input}")

    # Invoke the RAG chain with the user's input
    response = rag_chain.invoke({"input": user_input})
    
    print(f"Response: {response['answer']}")
    return str(response["answer"])

# --- Main Execution ---

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)