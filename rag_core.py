# rag_core.py

import os
import fitz  # PyMuPDF
import docx
import email
import io
import numpy as np
import google.generativeai as genai
import requests
from sentence_transformers import SentenceTransformer, util
from typing import List
from urllib.parse import urlparse


class RAGProcessor:
    def __init__(self):
        self._setup_gemini_api()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            top_p=0.9,
            max_output_tokens=500
        )

    def _setup_gemini_api(self):
        api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=api_key)

    def _smart_chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
        text = text.replace('\n\n\n', '\n\n').strip()
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                words = current_chunk.split()
                overlap_text = ' '.join(words[-30:])
                current_chunk = overlap_text + "\n\n" + paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 100]

    def _extract_text_from_pdf_content(self, content: bytes) -> str:
        doc = fitz.open(stream=content, filetype="pdf")
        return "".join(page.get_text() for page in doc)

    def _extract_text_from_docx_content(self, content: bytes) -> str:
        doc = docx.Document(io.BytesIO(content))
        return "\n".join([para.text for para in doc.paragraphs])

    def _extract_text_from_eml_content(self, content: bytes) -> str:
        msg = email.message_from_bytes(content)
        text = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    text += part.get_payload(decode=True).decode(errors='ignore')
        else:
            if msg.get_content_type() == "text/plain":
                text = msg.get_payload(decode=True).decode(errors='ignore')
        return text

    # This function now IGNORES the document context.
    def generate_answer_for_question(self, question: str) -> str:
        prompt = (
            f"Your job is to only return the direct answer to the question. "
            f"Do not include any explanation or formatting. Return a plain sentence.\n\n"
            f"QUESTION: {question}\nANSWER:"
        )
        try:
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            return response.text.strip()
        except Exception as e:
            return f"Error generating answer: {e}"

    def process_request(self, document_url: str, questions: List[str]) -> List[str]:
        # This function still downloads and processes the document to meet the API requirements,
        # but the extracted text is NOT used by generate_answer_for_question.
        try:
            response = requests.get(document_url)
            response.raise_for_status()
            content = response.content

            parsed_path = urlparse(document_url).path
            file_extension = os.path.splitext(parsed_path)[1].lower()

            full_text = ""
            if file_extension == '.pdf':
                full_text = self._extract_text_from_pdf_content(content)
            elif file_extension == '.docx':
                full_text = self._extract_text_from_docx_content(content)
            elif file_extension == '.eml':
                full_text = self._extract_text_from_eml_content(content)
            else:
                raise ValueError(f"Unsupported or unrecognized file type from URL path: {parsed_path}")

        except requests.exceptions.RequestException as e:
            return [f"Failed to download document: {e}"] * len(questions)
        except Exception as e:
            return [f"Failed to process document: {e}"] * len(questions)

        document_chunks = self._smart_chunk_text(full_text)
        if not document_chunks:
            return ["Failed to extract text from the document."] * len(questions)

        # The document_chunks are ignored here as requested.
        answers = [self.generate_answer_for_question(q) for q in questions]
        return answers