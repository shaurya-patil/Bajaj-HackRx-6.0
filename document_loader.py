import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader, UnstructuredEmailLoader

def load_documents(file_paths: List[str]) -> List[Document]:

    all_docs = []

    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(path)
        elif ext == ".txt":
            loader = TextLoader(path, encoding="utf-8")
        elif ext == ".eml":
            loader = UnstructuredEmailLoader(path)
        else:
            raise ValueError(f"Unsupported file type: {ext} for {path}")

        docs = loader.load()
        all_docs.extend(docs)

    return all_docs
