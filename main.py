import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from google import genai


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medical-index"

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable")
if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY environment variable")

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

client = genai.Client(api_key=GEMINI_API_KEY)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

embed_model: Optional[SentenceTransformer] = None
pinecone_index = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, pinecone_index

    log.info("Loading embedding model...")
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    log.info("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)

    log.info("System ready.")
    yield
    log.info("Shutdown complete.")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    text: str


class MatchResult(BaseModel):
    symptom: str
    reply: str
    category: str
    confidence: float


class AskResponse(BaseModel):
    query: str
    gemini_reply: str
    model_used: str
    matches: List[MatchResult]
    low_confidence: bool


def search_pinecone(query: str) -> List[MatchResult]:
    vector = embed_model.encode(query).tolist()
    result = pinecone_index.query(vector=vector, top_k=3, include_metadata=True)

    matches = []
    for m in result.matches:
        meta = m.metadata or {}
        matches.append(MatchResult(
            symptom=meta.get("symptom", ""),
            reply=meta.get("reply", ""),
            category=meta.get("category", ""),
            confidence=float(m.score),
        ))
    return matches


def build_prompt(query: str, matches: List[MatchResult]) -> str:
    context = "\n".join([
        f"Source {i+1}\nSymptom: {m.symptom}\nReply: {m.reply}\nCategory: {m.category}\nConfidence: {m.confidence:.2f}"
        for i, m in enumerate(matches)
    ])
    return (
        "You are a medical assistant.\n"
        "Use ONLY the context below to answer. "
        "If the context is insufficient, advise the user to consult a doctor.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )


def ask_gemini(prompt: str) -> tuple[str, str]:
    last_error = None
    for model in GEMINI_MODELS:
        try:
            log.info(f"Trying Gemini model: {model}")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            return response.text.strip(), model
        except Exception as e:
            log.warning(f"Model {model} failed: {e}")
            last_error = e

    return f"All Gemini models failed. Last error: {last_error}", "none"


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Empty text")

    matches = search_pinecone(req.text)
    prompt = build_prompt(req.text, matches)
    reply, model_used = ask_gemini(prompt)

    return AskResponse(
        query=req.text,
        gemini_reply=reply,
        model_used=model_used,
        matches=matches,
        low_confidence=not matches,
    )


@app.get("/health")
def health():
    return {"status": "ok"}