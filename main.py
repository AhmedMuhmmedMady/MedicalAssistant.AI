import os
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────
# Config (Railway Environment Variables only)
# ─────────────────────────────────────────────

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX", "medical-index")
MODEL_NAME       = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
TOP_K            = int(os.getenv("TOP_K", "3"))
MIN_CONFIDENCE   = float(os.getenv("MIN_CONFIDENCE", "0.3"))
PORT             = int(os.getenv("PORT", "8000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────

embed_model: Optional[SentenceTransformer] = None
pinecone_index = None
gemini_model = None


# ─────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, pinecone_index, gemini_model

    if not GEMINI_API_KEY:
        raise Exception("GEMINI_API_KEY is missing in environment variables")

    if not PINECONE_API_KEY:
        raise Exception("PINECONE_API_KEY is missing in environment variables")

    log.info("Loading embedding model...")
    embed_model = SentenceTransformer(MODEL_NAME)

    log.info("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)

    log.info("Configuring Gemini...")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

    log.info("All services initialized successfully.")

    yield

    log.info("Shutting down...")


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Medical AI Service",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

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
    matches: List[MatchResult]
    low_confidence: bool


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def search_pinecone(query: str) -> List[MatchResult]:
    vector = embed_model.encode(query).tolist()
    result = pinecone_index.query(
        vector=vector,
        top_k=TOP_K,
        include_metadata=True
    )

    matches = []
    for m in result.matches:
        meta = m.metadata or {}
        matches.append(MatchResult(
            symptom=meta.get("symptom", ""),
            reply=meta.get("reply", ""),
            category=meta.get("category", ""),
            confidence=round(float(m.score), 4)
        ))

    return matches


def build_gemini_prompt(query: str, matches: List[MatchResult]) -> str:
    context = "\n".join([
        f"[Source {i+1}]\n"
        f"Category: {m.category}\n"
        f"Symptom: {m.symptom}\n"
        f"Reply: {m.reply}\n"
        for i, m in enumerate(matches)
    ])

    return f"""
You are a medical assistant.
Use ONLY the context below.

Context:
{context}

User Question:
{query}

Answer clearly and safely.
"""


def ask_gemini(prompt: str) -> str:
    try:
        response = gemini_model.generate_content(prompt)

        if not response or not response.text:
            raise ValueError("Empty response from Gemini")

        return response.text.strip()

    except Exception as e:
        log.error(f"Gemini error: {str(e)}")
        return f"Gemini failed: {str(e)}"


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Empty question")

    try:
        matches = search_pinecone(req.text)
    except Exception as e:
        log.error(f"Pinecone error: {e}")
        raise HTTPException(status_code=503, detail="Pinecone unavailable")

    low_confidence = not matches or matches[0].confidence < MIN_CONFIDENCE

    if low_confidence:
        gemini_reply = (
            "I couldn't find strong matches in the medical database. "
            "Please consult a doctor."
        )
    else:
        prompt = build_gemini_prompt(req.text, matches)
        gemini_reply = ask_gemini(prompt)

    return AskResponse(
        query=req.text,
        gemini_reply=gemini_reply,
        matches=matches,
        low_confidence=low_confidence
    )