import os
from pathlib import Path

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from pdf_reader import extract_pdf

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

if not any(os.getenv(name, "").strip() for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY")):
    raise RuntimeError(
        "Google API key is missing. Set GOOGLE_API_KEY or GEMINI_API_KEY "
        "in the project-root .env file or your environment."
    )

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = InMemoryVectorStore(embeddings)

chunks = extract_pdf("pdfs/llm_agent_eval_metrics_text.pdf")
vectorstore.add_texts(texts=chunks)
