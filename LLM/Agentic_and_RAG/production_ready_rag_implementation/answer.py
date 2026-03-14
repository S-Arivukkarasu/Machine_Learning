from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

OLLAMA_MODEL = "llama3.1"
DB_NAME = Path("/home/alexender/Desktop/Projects/My_projects/Data/prod-ready-rag")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
collection_name = "MD_Docs"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
RETRIEVAL_K = 10

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings, collection_name=collection_name)
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
llm = ChatOllama(temperature=0, model=OLLAMA_MODEL)


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question)

def combined_question(question: str, history: list[dict]=[]) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" +question

def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs