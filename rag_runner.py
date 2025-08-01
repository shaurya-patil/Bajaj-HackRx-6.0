
from typing import List
from langchain_core.documents import Document

from document_loader import load_documents
from vector_store import build_vectorstore
from rag_pipeline import build_rag_chain

from dotenv import load_dotenv
load_dotenv()


def run_rag_pipeline(document_paths: List[str], questions: List[str]) -> List[dict]:
    print("Loading documents")
    documents: List[Document] = load_documents(document_paths)

    if not documents:
        raise ValueError("No documents were loaded. Check the paths.")

    print("Building vectorstore")
    retriever = build_vectorstore(documents)

    print("Initializing RAG chain")
    rag_chain = build_rag_chain(retriever)

    print("Running queries\n")
    results = []
    for q in questions:
        print(f"Q: {q}")
        answer = rag_chain.invoke(q)
        print(f"A: {answer}\n")
        results.append({
            "question": q,
            "answer": answer.strip()
        })

    return results
