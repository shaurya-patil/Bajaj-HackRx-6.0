import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(retriever: VectorStoreRetriever):

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant. Use the context below to answer the question:

Context:
{context}

Question:
{question}

Answer:"""
    )

    rag_chain = (
        RunnableMap({
            "context": RunnableLambda(lambda q: retriever.get_relevant_documents(q)) | RunnableLambda(format_docs),
            "question": RunnableLambda(lambda x: x)
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

