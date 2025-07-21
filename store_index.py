from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore
from tqdm import tqdm

load_dotenv()


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY



extracted_data=load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks=text_split(filter_data)

embeddings = embeddings

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)



index_name = "medical-chatbot"  # change if desired


# Create a Pinecone client instance
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"

if index_name not in pc.list_indexes().names():
    print(f"Creating index {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=768,     
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",         
            region="us-east-1"    
        )
    )

pinecone_index = pc.Index(index_name)



batch_size = 100  # You can adjust this number if needed

for i in tqdm(range(0, len(text_chunks), batch_size)):
    # Get the batch of documents
    i_end = min(i + batch_size, len(text_chunks))
    batch = text_chunks[i:i_end]
    
    # Prepare texts and metadata
    texts = [chunk.page_content for chunk in batch]
    metadatas = [chunk.metadata for chunk in batch]
    
    # Create unique IDs for each chunk
    ids = [f"{index_name}-{i+j}" for j in range(len(batch))]
    
    # Get embeddings for the batch
    embeds = embeddings.embed_documents(texts)
    
    # Prepare the vectors for upsert
    vectors_to_upsert = list(zip(ids, embeds, metadatas))
    
    # Upsert the batch to Pinecone
    pinecone_index.upsert(vectors=vectors_to_upsert)