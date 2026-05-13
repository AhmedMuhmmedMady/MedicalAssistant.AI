from __future__ import annotations

import os
import re
import json
import base64
import hashlib
import logging
import asyncio
from enum import Enum
from typing import List, Dict, Optional, Tuple

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from google import genai
from pinecone import Pinecone

# ═════════════════ LOGGING ═════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("medical-ai")

# ═════════════════ ENV ═════════════════

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_PRIMARY = os.getenv("PINECONE_INDEX_PRIMARY", "")
INDEX_LEGACY = os.getenv("PINECONE_INDEX_LEGACY", "")

TOP_K = int(os.getenv("TOP_K", "7"))
MAX_DOCS = int(os.getenv("MAX_DOCS", "10"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.6"))

if not all([GEMINI_API_KEY, PINECONE_API_KEY, INDEX_PRIMARY]):
    raise ValueError("Missing env vars")

# ═════════════════ CLIENTS ═════════════════

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_primary = pc.Index(INDEX_PRIMARY)
index_legacy = pc.Index(INDEX_LEGACY) if INDEX_LEGACY else None

# ═════════════════ SIMPLE CACHE ═════════════════

EMBED_CACHE: Dict[str, List[float]] = {}

def cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

async def get_embedding(text: str):
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

# ═════════════════ MODELS ═════════════════

class AskPayload(BaseModel):
    question: Optional[str] = None
    text: Optional[str] = None


class QueryIntent(str, Enum):
    GREETING = "greeting"
    GRATITUDE = "gratitude"
    FAREWELL = "farewell"
    MEDICAL = "medical"

# ═════════════════ HELPERS ═════════════════

def normalize(payload: AskPayload) -> str:
    return (payload.question or payload.text or "").strip()

# ═════════════════ INTENT ROUTER (FAST PATH) ═════════════════

class IntentClassifier:

    @staticmethod
    def classify(text: str, score: float = 0.0) -> QueryIntent:
        t = text.lower()

        if any(x in t for x in ["hi", "hello", "مرحبا", "ازيك"]):
            return QueryIntent.GREETING

        if any(x in t for x in ["thanks", "شكرا"]):
            return QueryIntent.GRATITUDE

        if any(x in t for x in ["bye", "سلام"]):
            return QueryIntent.FAREWELL

        if score > SCORE_THRESHOLD and len(t.split()) > 3:
            return QueryIntent.MEDICAL

        return QueryIntent.MEDICAL

# ═════════════════ HYBRID SEARCH ═════════════════

def query_index(index, vector):
    try:
        res = index.query(
            vector=vector,
            top_k=TOP_K,
            include_metadata=True
        )
        return [
            {"text": m.metadata.get("text", ""), "score": m.score}
            for m in res.matches
            if m.score >= SCORE_THRESHOLD
        ]
    except:
        return []


def hybrid_search(vector):
    seen = {}

    for idx in [index_primary, index_legacy]:
        if not idx:
            continue

        for d in query_index(idx, vector):
            txt = d["text"]
            if txt not in seen or d["score"] > seen[txt]["score"]:
                seen[txt] = d

    docs = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return docs[:MAX_DOCS], (docs[0]["score"] if docs else 0.0)

# ═════════════════ RERANKER ═════════════════

def rerank(query: str, docs: List[Dict]) -> List[Dict]:
    q = query.lower()
    scored = []

    for d in docs:
        bonus = 0

        if any(w in d["text"].lower() for w in q.split()):
            bonus += 0.1

        bonus += min(len(d["text"]) / 2000, 0.2)

        d["score"] += bonus
        scored.append(d)

    return sorted(scored, key=lambda x: x["score"], reverse=True)

# ═════════════════ LLM ═════════════════

async def generate_social():
    return "أهلاً 👋 ازاي أقدر أساعدك؟"

async def generate_rag(question: str, docs: List[Dict]):
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

    return result.text + "\n\n⚠️ تنبيه طبي"

# ═════════════════ FASTAPI ═════════════════

app = FastAPI(title="Medical AI vNext")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═════════════════ /ask (UPGRADED PIPELINE) ═════════════════

@app.post("/ask")
async def ask(payload: AskPayload):

    q = normalize(payload)
    if not q:
        return JSONResponse(status_code=400, content={"error": "empty"})

    # 1. FAST INTENT CHECK (NO EMBEDDING)
    intent = IntentClassifier.classify(q)

    if intent in [
        QueryIntent.GREETING,
        QueryIntent.GRATITUDE,
        QueryIntent.FAREWELL,
    ]:
        return {
            "query": q,
            "gemini_reply": await generate_social(),
            "intent": intent.value,
            "is_medical": False,
        }

    # 2. EMBEDDING (cached)
    vector = await get_embedding(q)

    # 3. SEARCH
    docs, score = await asyncio.to_thread(hybrid_search, vector)

    # 4. RERANK
    docs = rerank(q, docs)

    # 5. NO DATA
    if not docs:
        return {
            "query": q,
            "gemini_reply": "مفيش بيانات كافية.",
            "is_medical": True,
        }

    # 6. RAG RESPONSE
    answer = await generate_rag(q, docs)

    return {
        "query": q,
        "gemini_reply": answer,
        "matches": docs,
        "intent": intent.value,
        "low_confidence": score < 0.75,
        "is_medical": True,
    }

# ═════════════════ HEALTH ═════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cache_size": len(EMBED_CACHE)
    }