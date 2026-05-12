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
# Config (set these as Railway environment variables)
# ─────────────────────────────────────────────
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY",  "pcsk_3Bxu6E_HjF5cNUBvb5aQJ3qYmBmcGtfinhJuc1Gd1Kj5oJcxdQR4FtJjjJHFcMvzwxtPow")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY",    "AIzaSyANo6d9z_nu_fHOccstqyDvSTbSrRxCZbo")
INDEX_NAME        = os.getenv("PINECONE_INDEX",    "medical-index")
MODEL_NAME        = os.getenv("EMBED_MODEL",       "paraphrase-multilingual-MiniLM-L12-v2")
TOP_K             = int(os.getenv("TOP_K",         "3"))
MIN_CONFIDENCE    = float(os.getenv("MIN_CONFIDENCE", "0.3"))
PORT              = int(os.getenv("PORT",          "8000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Global singletons
# ─────────────────────────────────────────────
embed_model: Optional[SentenceTransformer] = None
pinecone_index = None
gemini_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, pinecone_index, gemini_model

    log.info("Loading embedding model…")
    embed_model = SentenceTransformer(MODEL_NAME)
    log.info("Embedding model loaded.")

    log.info("Connecting to Pinecone…")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)
    log.info("Pinecone connected.")

    log.info("Configuring Gemini…")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    log.info("Gemini ready.")

    yield  # app runs here

    log.info("Shutting down…")


app = FastAPI(
    title="Medical AI Service",
    description="Pinecone semantic search + Gemini response generation",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
    result = pinecone_index.query(vector=vector, top_k=TOP_K, include_metadata=True)

    matches = []
    for m in result.matches:
        meta = m.metadata or {}
        matches.append(MatchResult(
            symptom=meta.get("symptom", ""),
            reply=meta.get("reply", ""),
            category=meta.get("category", ""),
            confidence=round(float(m.score), 4),
        ))
    return matches


def build_gemini_prompt(query: str, matches: List[MatchResult]) -> str:
    context_parts = []
    for i, m in enumerate(matches, 1):
        context_parts.append(
            f"[Source {i}]\n"
            f"Category: {m.category}\n"
            f"Symptom: {m.symptom}\n"
            f"Medical Reply: {m.reply}\n"
        )
    context = "\n".join(context_parts)

    return (
        "You are a helpful medical assistant. "
        "Using ONLY the medical information provided below, answer the user's question clearly and concisely. "
        "If the information is insufficient, say so politely. "
        "Do NOT invent information. Always recommend consulting a doctor for serious concerns.\n\n"
        f"--- Medical Context ---\n{context}\n"
        f"--- User Question ---\n{query}\n\n"
        "Answer:"
    )


def ask_gemini(prompt: str) -> str:
    for attempt in range(3):
        try:
            response = gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            log.warning(f"Gemini attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return "Sorry, the AI service is temporarily unavailable. Please try again later."


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "index": INDEX_NAME}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=422, detail="Question text cannot be empty.")

    try:
        matches = search_pinecone(req.text.strip())
    except Exception as e:
        log.error(f"Pinecone error: {e}")
        raise HTTPException(status_code=503, detail="Search service unavailable.")

    low_confidence = not matches or matches[0].confidence < MIN_CONFIDENCE

    if low_confidence:
        gemini_reply = (
            "I couldn't find relevant medical information for your question in our database. "
            "Please consult a qualified healthcare professional."
        )
    else:
        prompt = build_gemini_prompt(req.text.strip(), matches)
        gemini_reply = ask_gemini(prompt)

    return AskResponse(
        query=req.text.strip(),
        gemini_reply=gemini_reply,
        matches=matches,
        low_confidence=low_confidence,
    )
