# ============================================================
# main.py
# Medical AI Assistant v11
# Strict Medical RAG + Social AI + Query Rewriting
# + Conversation Memory + Retry Logic + Pydantic Validation
# ============================================================

from __future__ import annotations

import os
import json
import base64
import logging
import asyncio
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from google import genai
from pinecone import Pinecone

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

log = logging.getLogger("medical-ai")

# ============================================================
# ENV
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY")
if not PINECONE_INDEX:
    raise ValueError("Missing PINECONE_INDEX")

# ============================================================
# Clients
# ============================================================

client = genai.Client(api_key=GEMINI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# ============================================================
# FastAPI
# ============================================================

app = FastAPI(title="Medical AI Assistant", version="11.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Constants
# ============================================================

DISCLAIMER = (
    "⚠️ تنبيه: هذه المعلومات للتوجيه العام فقط "
    "ولا تُغني عن استشارة طبيب متخصص."
)

MAX_RETRIES = 3
RETRY_DELAY = 1.5  # seconds
SCORE_THRESHOLD = 0.65
TOP_K = 7
MAX_DOCS = 10

# ============================================================
# Pydantic Models
# ============================================================


class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=4000)


class AskPayload(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: Optional[List[Message]] = Field(default_factory=list)


class AskResponse(BaseModel):
    status: str
    intent: str
    reply: str
    source: str


# ============================================================
# SYSTEM PROMPTS
# ============================================================

INTENT_PROMPT = """
You are an intent classifier.

Classify ONLY into:
- social
- medical

Rules:
- Greetings, thanks, casual chat => social
- Symptoms, diseases, medicine, pain, diagnosis => medical

Return ONLY the single word:
social
OR
medical
""".strip()

SOCIAL_PROMPT = """
You are a warm Egyptian medical assistant chatbot.

Rules:
- Speak naturally in Egyptian Arabic (عامية مصرية).
- Be short, friendly, and human.
- No robotic or generic replies.
- Never fabricate diagnoses.
- If user mentions any symptom, gently ask for more details.
- Max 3 sentences.
""".strip()

QUERY_REWRITE_PROMPT = """
You are a medical query normalizer for a RAG system.

Convert the user message into search-optimized forms.

Return ONLY valid JSON with no markdown:
{
  "normalized_ar": "الاستعلام بالعربي الطبي الفصيح",
  "medical_keywords": "English medical keywords for search",
  "symptom_query": "symptom-focused Arabic search query"
}
""".strip()

RAG_PROMPT = """
You are a professional Arabic medical AI assistant.

STRICT RULES:
1. Answer ONLY using the provided medical context.
2. Never hallucinate or invent information.
3. If context is insufficient, say so clearly.
4. Always respond in clear Arabic.
5. Be medically safe — never recommend stopping prescribed medication.
6. Structure your answer clearly with short paragraphs.
7. If the user's history shows follow-up questions, maintain continuity.
""".strip()

VISION_PROMPT = """
You are a medical imaging analysis assistant.

Analyze the provided medical image carefully.

Return ONLY valid JSON with no markdown:
{
  "status": "success",
  "analysis_ar": "التحليل باللغة العربية بشكل مفصل",
  "technical_details": "Technical imaging details in English"
}
""".strip()

# ============================================================
# Retry Wrapper
# ============================================================


async def call_with_retry(fn, *args, retries=MAX_RETRIES, **kwargs):
    """Call an async or sync function with exponential backoff retry."""
    last_error = None
    for attempt in range(retries):
        try:
            if asyncio.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            else:
                return fn(*args, **kwargs)
        except Exception as ex:
            last_error = ex
            wait = RETRY_DELAY * (2 ** attempt)
            log.warning(f"Attempt {attempt + 1} failed: {ex}. Retrying in {wait:.1f}s...")
            await asyncio.sleep(wait)
    raise last_error


# ============================================================
# Intent Detection
# ============================================================


async def detect_intent(message: str) -> str:
    try:
        def _call():
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=message,
                config={
                    "temperature": 0,
                    "max_output_tokens": 5,
                    "system_instruction": INTENT_PROMPT,
                },
            )

        response = await call_with_retry(_call)
        result = response.text.strip().lower()
        return "medical" if "medical" in result else "social"

    except Exception as ex:
        log.exception(f"Intent detection failed: {ex}")
        return "medical"  # safe default


# ============================================================
# Social Chat
# ============================================================


async def social_chat(message: str, history: List[Message]) -> str:
    try:
        # Build conversation turns for context
        history_text = ""
        if history:
            for msg in history[-6:]:  # last 3 turns
                prefix = "المستخدم" if msg.role == "user" else "المساعد"
                history_text += f"{prefix}: {msg.content}\n"

        contents = f"{history_text}المستخدم: {message}" if history_text else message

        def _call():
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config={
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "max_output_tokens": 150,
                    "system_instruction": SOCIAL_PROMPT,
                },
            )

        response = await call_with_retry(_call)
        return response.text.strip()

    except Exception as ex:
        log.exception(f"Social chat failed: {ex}")
        return "أهلاً 🌹 تحت أمرك، كيف أقدر أساعدك؟"


# ============================================================
# Query Rewriter
# ============================================================


async def rewrite_query(question: str) -> Dict[str, str]:
    try:
        def _call():
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=question,
                config={
                    "temperature": 0.2,
                    "max_output_tokens": 300,
                    "system_instruction": QUERY_REWRITE_PROMPT,
                },
            )

        response = await call_with_retry(_call)
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)

        return {
            "normalized_ar": parsed.get("normalized_ar", question),
            "medical_keywords": parsed.get("medical_keywords", question),
            "symptom_query": parsed.get("symptom_query", question),
        }

    except Exception as ex:
        log.exception(f"Query rewrite failed: {ex}")
        return {
            "normalized_ar": question,
            "medical_keywords": question,
            "symptom_query": question,
        }


# ============================================================
# Embedding (single call, batched internally)
# ============================================================


def create_embedding(text: str) -> List[float]:
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text,
    )
    return response.embeddings[0].values


def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Embed multiple texts. The google-genai SDK embed_content
    accepts a list — we use that to avoid 4 sequential HTTP calls.
    Falls back to sequential on error.
    """
    try:
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=texts,
        )
        return [e.values for e in response.embeddings]
    except Exception as ex:
        log.warning(f"Batch embedding failed, falling back to sequential: {ex}")
        return [create_embedding(t) for t in texts]


# ============================================================
# Pinecone Search
# ============================================================


def pinecone_search(query_embedding: List[float], top_k: int = TOP_K) -> List[Dict]:
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    documents = []
    for match in results.matches:
        metadata = match.metadata or {}
        text = metadata.get("text")
        score = match.score

        if not text or score < SCORE_THRESHOLD:
            continue

        documents.append({"score": score, "text": text})

    return documents


# ============================================================
# Multi-Query Retrieval (batched embeddings)
# ============================================================


async def retrieve_documents(question: str) -> List[Dict]:
    rewritten = await rewrite_query(question)

    queries = [
        question,
        rewritten["normalized_ar"],
        rewritten["medical_keywords"],
        rewritten["symptom_query"],
    ]

    # Deduplicate queries before embedding
    unique_queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))

    # Batch embed all queries in ONE API call
    embeddings = create_embeddings_batch(unique_queries)

    merged: Dict[str, Dict] = {}

    for embedding in embeddings:
        docs = pinecone_search(embedding)
        for doc in docs:
            text = doc["text"]
            # Keep highest score if duplicate
            if text not in merged or doc["score"] > merged[text]["score"]:
                merged[text] = doc

    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:MAX_DOCS]


# ============================================================
# Generate Medical Answer (with history)
# ============================================================


async def generate_medical_answer(
    question: str,
    docs: List[Dict],
    history: List[Message],
) -> str:
    context = "\n\n---\n\n".join(doc["text"] for doc in docs)

    # Build history string for continuity
    history_text = ""
    if history:
        for msg in history[-6:]:
            prefix = "المستخدم" if msg.role == "user" else "المساعد"
            history_text += f"{prefix}: {msg.content}\n"

    prompt = f"""
السياق الطبي:
{context}

{'سجل المحادثة السابقة:' + chr(10) + history_text if history_text else ''}
سؤال المستخدم الحالي:
{question}

أجب فقط باستخدام السياق الطبي المقدم أعلاه.
""".strip()

    try:
        def _call():
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "max_output_tokens": 700,
                    "system_instruction": RAG_PROMPT,
                },
            )

        response = await call_with_retry(_call)
        answer = response.text.strip()
        return f"{answer}\n\n{DISCLAIMER}"

    except Exception as ex:
        log.exception(f"Medical answer generation failed: {ex}")
        return f"حدث خطأ أثناء إنشاء الرد الطبي.\n\n{DISCLAIMER}"


# ============================================================
# Vision Parser
# ============================================================


def parse_vision_response(raw_text: str):
    clean = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(clean)
        return (
            parsed.get("status", "success"),
            parsed.get("analysis_ar", ""),
            parsed.get("technical_details", ""),
        )
    except Exception:
        return "success", raw_text, "Raw text response."


# ============================================================
# Routes
# ============================================================


@app.get("/")
async def root():
    return {"status": "running", "version": "11.0.0"}


@app.get("/health")
async def health():
    """Health check — verifies Pinecone connection."""
    try:
        stats = index.describe_index_stats()
        return {
            "status": "ok",
            "pinecone_vectors": stats.total_vector_count,
        }
    except Exception as ex:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": str(ex)},
        )


# ============================================================
# /ask  — main chat endpoint
# ============================================================


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskPayload):
    question = payload.question.strip()
    history = payload.history or []

    intent = await detect_intent(question)

    if intent == "social":
        reply = await social_chat(question, history)
        return AskResponse(
            status="success",
            intent="social",
            reply=reply,
            source="social-ai",
        )

    # Medical path
    docs = await retrieve_documents(question)

    if not docs:
        return AskResponse(
            status="success",
            intent="medical",
            source="fallback",
            reply=(
                "مش لاقي معلومات كافية في قاعدة البيانات عشان أجاوب بدقة. "
                "ممكن توضّح الأعراض أكتر؟\n\n" + DISCLAIMER
            ),
        )

    answer = await generate_medical_answer(
        question=question,
        docs=docs,
        history=history,
    )

    return AskResponse(
        status="success",
        intent="medical",
        reply=answer,
        source="rag",
    )


# ============================================================
# /analyze-image
# ============================================================


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WebP.",
        )

    try:
        image_bytes = await file.read()

        if len(image_bytes) > 10 * 1024 * 1024:  # 10 MB limit
            raise HTTPException(status_code=413, detail="Image too large. Max 10MB.")

        # Encode to base64 — required by Gemini vision API
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        def _call():
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    {
                        "inline_data": {
                            "mime_type": file.content_type,
                            "data": image_b64,
                        }
                    },
                    {"text": "Analyze this medical image."},
                ],
                config={
                    "temperature": 0.2,
                    "max_output_tokens": 800,
                    "system_instruction": VISION_PROMPT,
                },
            )

        response = await call_with_retry(_call)
        raw_text = response.text.strip()
        status, analysis, technical = parse_vision_response(raw_text)

        return JSONResponse(content={
            "status": status,
            "analysis_ar": analysis,
            "technical_details": technical,
            "model_used": "gemini-2.5-flash",
            "disclaimer": DISCLAIMER,
        })

    except HTTPException:
        raise
    except Exception as ex:
        log.exception(f"Image analysis failed: {ex}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "analysis_ar": "حدث خطأ أثناء تحليل الصورة.",
                "technical_details": str(ex),
                "model_used": "gemini-2.5-flash",
                "disclaimer": DISCLAIMER,
            },
        )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)