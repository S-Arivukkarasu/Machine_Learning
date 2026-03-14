import os
import glob
from pathlib import Path
from huggingface_hub import login
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings


DB_NAME = str(Path("/home/alexender/Desktop/Projects/My_projects/Data/prod-ready-rag"))
collection_name = "MD_Docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
KNOWLEDGE_BASE_PATH = str(Path("/home/alexender/Desktop/Projects/My_projects/Data/knowledge-base"))
AVERAGE_CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# For using Hugging Face
hf_token = os.getenv("HUGGING_FACE_WRITE_TOKEN")
login(hf_token)


def fetch_documents():
    """A homemade version of the LangChain DirectoryLoader"""
    
    folders = glob.glob(str(Path(KNOWLEDGE_BASE_PATH) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=AVERAGE_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings, collection_name=collection_name).delete_collection()
        print(f"{collection_name} already exists and deleting")

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME, collection_name=collection_name
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")