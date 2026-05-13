from __future__ import annotations

import os
import re
import json
import asyncio
import logging
import hashlib
from enum import Enum
from typing import List, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from google import genai
from pinecone import Pinecone

# ═════════════════ LOGGING ═════════════════

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("medical-ai")

# ═════════════════ SAFE ENV (NO CRASH) ═════════════════

def env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if not value:
        log.warning(f"Missing env var: {name}")
    return value


GEMINI_API_KEY = env("GEMINI_API_KEY")
PINECONE_API_KEY = env("PINECONE_API_KEY")

# 🔥 FIX: match Railway variable name
INDEX_PRIMARY = env("PINECONE_INDEX")

# ═════════════════ CLIENTS (SAFE INIT) ═════════════════

gemini_client = None
pc = None
index_primary = None

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)

if pc and INDEX_PRIMARY:
    index_primary = pc.Index(INDEX_PRIMARY)

# ═════════════════ APP ═════════════════

app = FastAPI(title="Medical AI Safe Deploy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═════════════════ MODELS ═════════════════

class AskPayload(BaseModel):
    question: str


class QueryIntent(str, Enum):
    GREETING = "greeting"
    GRATITUDE = "gratitude"
    FAREWELL = "farewell"
    MEDICAL = "medical"

# ═════════════════ HELPERS ═════════════════

def normalize(p: AskPayload) -> str:
    return p.question.strip()

# ═════════════════ CACHE ═════════════════

EMBED_CACHE: Dict[str, List[float]] = {}

def cache_key(t: str) -> str:
    return hashlib.md5(t.encode()).hexdigest()

# ═════════════════ EMBEDDING ═════════════════

async def embed(text: str):
    if not gemini_client:
        return []

    key = cache_key(text)
    if key in EMBED_CACHE:
        return EMBED_CACHE[key]

    result = await asyncio.to_thread(
        lambda: gemini_client.models.embed_content(
            model="text-embedding-004",
            contents=text,
        )
    )

    vec = result.embeddings[0].values
    EMBED_CACHE[key] = vec
    return vec

# ═════════════════ INTENT ROUTER ═════════════════

class IntentClassifier:
    @staticmethod
    def classify(text: str, score: float = 0.0) -> QueryIntent:
        t = text.lower()

        if any(x in t for x in ["hi", "hello", "ازيك", "مرحبا"]):
            return QueryIntent.GREETING

        if any(x in t for x in ["thanks", "شكرا"]):
            return QueryIntent.GRATITUDE

        if any(x in t for x in ["bye", "سلام"]):
            return QueryIntent.FAREWELL

        return QueryIntent.MEDICAL

# ═════════════════ SEARCH ═════════════════

def search(vector):
    if not index_primary:
        return [], 0.0

    try:
        res = index_primary.query(
            vector=vector,
            top_k=7,
            include_metadata=True
        )

        docs = [
            {
                "text": m.metadata.get("text", ""),
                "score": m.score
            }
            for m in res.matches
            if m.score > 0.6
        ]

        return docs, (docs[0]["score"] if docs else 0.0)

    except Exception as e:
        log.error(f"Search error: {e}")
        return [], 0.0

# ═════════════════ RAG ═════════════════

async def rag_answer(question: str, docs: List[Dict]):
    if not gemini_client:
        return "AI service not configured."

    context = "\n\n".join(d["text"][:300] for d in docs)

    prompt = f"""
السياق:
{context}

السؤال:
{question}

أجب فقط من السياق.
"""

    result = await asyncio.to_thread(
        lambda: gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
    )

    return result.text

# ═════════════════ SOCIAL ═════════════════

async def social_reply():
    return "أهلاً 👋 ازاي أقدر أساعدك طبيًا؟"

# ═════════════════ /ASK ═════════════════

@app.post("/ask")
async def ask(payload: AskPayload):

    q = normalize(payload)
    if not q:
        return JSONResponse({"error": "empty question"}, status_code=400)

    intent = IntentClassifier.classify(q)

    # 🔥 FAST PATH (no embedding)
    if intent in [QueryIntent.GREETING, QueryIntent.GRATITUDE, QueryIntent.FAREWELL]:
        return {
            "query": q,
            "reply": await social_reply(),
            "intent": intent.value,
            "is_medical": False
        }

    # 🔥 EMBEDDING (safe)
    vector = await embed(q)

    # 🔥 SEARCH
    docs, score = await asyncio.to_thread(search, vector)

    if not docs:
        return {
            "query": q,
            "reply": "مفيش بيانات كافية في قاعدة المعرفة.",
            "is_medical": True
        }

    # 🔥 RAG
    answer = await rag_answer(q, docs)

    return {
        "query": q,
        "reply": answer,
        "matches": docs,
        "intent": intent.value,
        "low_confidence": score < 0.75,
        "is_medical": True
    }

# ═════════════════ HEALTH (ALWAYS SAFE) ═════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "gemini": bool(GEMINI_API_KEY),
        "pinecone": bool(PINECONE_API_KEY),
        "index": bool(INDEX_PRIMARY)
    }