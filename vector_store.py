from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document

def build_vectorstore(documents: List[Document], persist_directory: str = "chroma_db"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No document chunks were generated. Check your input documents.")

    print(f"Total Chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()

    print("Vectorstore persisted at:", persist_directory)

    return vectordb.as_retriever()
