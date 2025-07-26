# main.py

import os
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from dotenv import load_dotenv
from rag_core import RAGProcessor

load_dotenv()

app = FastAPI(
    title="HackRx 6.0 RAG API",
    description="Processes documents to answer questions according to HackRx 6.0 specs."
)
rag_processor = RAGProcessor()

# Define the security scheme and the required token
auth_scheme = HTTPBearer()
REQUIRED_BEARER_TOKEN = "222d8c43lcoba5ea6f20b9ad022368586c5e6d78a7d106dcbd58a56ceb2e591"

# Pydantic models for request and the simplified response
class RequestModel(BaseModel):
    documents: HttpUrl
    questions: List[str]

class ResponseModel(BaseModel):
    answers: List[str]

# Dependency to verify the token
def verify_token(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != REQUIRED_BEARER_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Invalid or missing bearer token"
        )
    return credentials.credentials

@app.post("/api/v1/hackrx/run", response_model=ResponseModel)
async def handle_hackrx_run(
    request: RequestModel,
    token: str = Security(verify_token)
):
    try:
        answers_data = rag_processor.process_request(str(request.documents), request.questions)
        return ResponseModel(answers=answers_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "HackRx 6.0 RAG API is running. POST to /api/v1/hackrx/run to submit."}