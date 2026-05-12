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


# ─────────────────────────────
# 🔴 ضع مفاتيحك هنا مباشرة للتجربة
# ─────────────────────────────

GEMINI_API_KEY = os.getenv("AIzaSyANo6d9z_nu_fHOccstqyDvSTbSrRxCZbo")
PINECONE_API_KEY = os.getenv("pcsk_3Bxu6E_HjF5cNUBvb5aQJ3qYmBmcGtfinhJuc1Gd1Kj5oJcxdQR4FtJjjJHFcMvzwxtPow")
INDEX_NAME = "medical-index"

if not GEMINI_API_KEY:
    raise Exception("Missing GEMINI_API_KEY")


# ─────────────────────────────
# Gemini Client (NEW SDK)
# ─────────────────────────────

client = genai.Client(api_key=GEMINI_API_KEY)


# ─────────────────────────────
# Logging
# ─────────────────────────────

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ─────────────────────────────
# Globals
# ─────────────────────────────

embed_model: Optional[SentenceTransformer] = None
pinecone_index = None


# ─────────────────────────────
# Lifespan
# ─────────────────────────────

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
    log.info("Shutdown")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────
# Models
# ─────────────────────────────

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


# ─────────────────────────────
# Pinecone Search
# ─────────────────────────────

def search_pinecone(query: str) -> List[MatchResult]:
    vector = embed_model.encode(query).tolist()

    result = pinecone_index.query(
        vector=vector,
        top_k=3,
        include_metadata=True
    )

    matches = []
    for m in result.matches:
        meta = m.metadata or {}
        matches.append(MatchResult(
            symptom=meta.get("symptom", ""),
            reply=meta.get("reply", ""),
            category=meta.get("category", ""),
            confidence=float(m.score)
        ))

    return matches


# ─────────────────────────────
# Prompt
# ─────────────────────────────

def build_prompt(query: str, matches: List[MatchResult]) -> str:
    context = "\n".join([
        f"""
Source {i+1}
Symptom: {m.symptom}
Reply: {m.reply}
Category: {m.category}
Confidence: {m.confidence}
"""
        for i, m in enumerate(matches)
    ])

    return f"""
You are a medical assistant.

Use ONLY the context below.
If not enough info, say consult a doctor.

Context:
{context}

Question:
{query}

Answer:
"""


# ─────────────────────────────
# Gemini Call
# ─────────────────────────────

def ask_gemini(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        return response.text.strip()

    except Exception as e:
        log.error(f"Gemini error: {e}")
        return f"Gemini failed: {str(e)}"


# ─────────────────────────────
# API Endpoint
# ─────────────────────────────

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Empty text")

    matches = search_pinecone(req.text)

    prompt = build_prompt(req.text, matches)
    reply = ask_gemini(prompt)

    return AskResponse(
        query=req.text,
        gemini_reply=reply,
        matches=matches,
        low_confidence=not matches
    )


@app.get("/health")
def health():
    return {"status": "ok"}