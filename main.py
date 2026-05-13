# ============================================================
# Medical AI Assistant — Strict RAG + Social AI Edition v9
# ============================================================

from __future__ import annotations

import os
import json
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from google import genai
from pinecone import Pinecone

# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("medical-ai")

# ============================================================
# Environment Variables
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
# Gemini Client
# ============================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# Pinecone
# ============================================================

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Medical AI Assistant",
    version="9.0.0",
)

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

MEDICAL_DISCLAIMER = (
    "⚠️ تنبيه: هذه المعلومات للتوجيه العام فقط "
    "ولا تُغني عن استشارة طبيب متخصص."
)

# ============================================================
# System Prompts
# ============================================================

INTENT_SYSTEM_PROMPT = """
You are an intent classifier.

Classify the user message into ONLY one category:

social
medical

RULES:
- Greetings => social
- Thanks => social
- Casual conversation => social
- Emotional talk => social
- Symptoms => medical
- Diseases => medical
- Medicines => medical
- Pain => medical
- Medical questions => medical

Return ONLY:
social

OR

medical
"""

SOCIAL_SYSTEM_PROMPT = """
You are a friendly Egyptian medical AI assistant.

RULES:
- Speak naturally.
- Use Egyptian Arabic naturally.
- Be warm and conversational.
- Keep responses short.
- Don't act robotic.
- Don't diagnose in social mode.
- If symptoms appear, ask the user to explain more.
"""

RAG_SYSTEM_PROMPT = """
You are a professional medical AI assistant.

STRICT RULES:
- Answer ONLY from the provided medical context.
- If context is insufficient say so clearly.
- Never hallucinate medical information.
- Use Arabic.
- Be medically safe.
- Keep answers clear and useful.
"""

VISION_SYSTEM_PROMPT = """
You are a medical vision AI assistant.

Analyze the medical image carefully.

IMPORTANT:
- Respond ONLY with valid JSON.
- No markdown.
- No extra text.

Required JSON format:

{
  "status": "success",
  "analysis_ar": "<Arabic explanation>",
  "technical_details": "<English technical details>"
}

If image is not medical:

{
  "status": "error",
  "analysis_ar": "الصورة المرفوعة ليست صورة طبية.",
  "technical_details": "Non-medical image."
}
"""

# ============================================================
# Helper Functions
# ============================================================

async def detect_intent(message: str) -> str:

    try:

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=message,
            config={
                "temperature": 0,
                "max_output_tokens": 5,
                "system_instruction": INTENT_SYSTEM_PROMPT,
            },
        )

        result = response.text.strip().lower()

        if "medical" in result:
            return "medical"

        return "social"

    except Exception as ex:

        log.exception("Intent detection failed: %s", ex)

        return "medical"


# ============================================================
# Pinecone Search
# ============================================================

def search_medical_context(question: str) -> List[str]:

    try:

        embedding_response = gemini_client.models.embed_content(
            model="text-embedding-004",
            contents=question,
        )

        embedding = embedding_response.embeddings[0].values

        results = index.query(
            vector=embedding,
            top_k=5,
            include_metadata=True,
        )

        documents = []

        for match in results.matches:

            metadata = match.metadata or {}

            text = metadata.get("text")

            if text:
                documents.append(text)

        return documents

    except Exception as ex:

        log.exception("Pinecone search failed: %s", ex)

        return []


# ============================================================
# Generate Medical RAG Answer
# ============================================================

async def generate_rag_answer(
    question: str,
    documents: List[str],
) -> str:

    context = "\n\n".join(documents)

    prompt = f"""
Medical Context:
{context}

User Question:
{question}

Answer ONLY using the medical context above.
"""

    try:

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "temperature": 0.2,
                "top_p": 0.9,
                "max_output_tokens": 500,
                "system_instruction": RAG_SYSTEM_PROMPT,
            },
        )

        answer = response.text.strip()

        return f"{answer}\n\n{MEDICAL_DISCLAIMER}"

    except Exception as ex:

        log.exception("RAG generation failed: %s", ex)

        return (
            "حدث خطأ أثناء إنشاء الرد الطبي.\n\n"
            + MEDICAL_DISCLAIMER
        )


# ============================================================
# Generate Social Response
# ============================================================

async def generate_social_response(
    message: str,
) -> str:

    try:

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=message,
            config={
                "temperature": 0.9,
                "top_p": 0.95,
                "max_output_tokens": 120,
                "system_instruction": SOCIAL_SYSTEM_PROMPT,
            },
        )

        return response.text.strip()

    except Exception as ex:

        log.exception("Social generation failed: %s", ex)

        return "أهلاً 🌹 تحت أمرك."


# ============================================================
# Parse Vision Response
# ============================================================

def parse_vision_response(raw_text: str):

    clean = (
        raw_text
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:

        parsed = json.loads(clean)

    except Exception:

        return (
            "success",
            raw_text,
            "Raw text response."
        )

    status = parsed.get("status", "success")

    analysis = (
        parsed.get("analysis_ar")
        or parsed.get("analysis")
        or "لم يتمكن النظام من إنشاء تحليل."
    )

    technical = (
        parsed.get("technical_details")
        or "No technical details."
    )

    return (
        status,
        analysis,
        technical,
    )


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "running",
        "service": "Medical AI Assistant v9"
    }


# ============================================================
# Ask Endpoint
# ============================================================

@app.post("/ask")
async def ask(payload: Dict[str, Any]):

    question = payload.get("question")

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question is required."
        )

    # ========================================================
    # Intent Detection
    # ========================================================

    intent = await detect_intent(question)

    # ========================================================
    # SOCIAL MODE
    # ========================================================

    if intent == "social":

        reply = await generate_social_response(question)

        return JSONResponse(content={
            "status": "success",
            "intent": "social",
            "reply": reply,
            "source": "gemini-social",
        })

    # ========================================================
    # MEDICAL MODE
    # ========================================================

    documents = search_medical_context(question)

    if not documents:

        return JSONResponse(content={
            "status": "success",
            "intent": "medical",
            "source": "fallback",
            "reply": (
                "مش لاقي معلومات كافية في قاعدة البيانات "
                "عشان أجاوب بدقة. "
                "ممكن توضّح سؤالك أو الأعراض أكتر؟\n\n"
                + MEDICAL_DISCLAIMER
            )
        })

    answer = await generate_rag_answer(
        question=question,
        documents=documents,
    )

    return JSONResponse(content={
        "status": "success",
        "intent": "medical",
        "reply": answer,
        "source": "rag",
    })


# ============================================================
# Analyze Medical Image Endpoint
# ============================================================

@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...)
):

    try:

        image_bytes = await file.read()

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "mime_type": file.content_type,
                    "data": image_bytes,
                },
                "Analyze this medical image."
            ],
            config={
                "temperature": 0.2,
                "max_output_tokens": 700,
                "system_instruction": VISION_SYSTEM_PROMPT,
            },
        )

        raw_text = response.text.strip()

        (
            status,
            analysis,
            technical,
        ) = parse_vision_response(raw_text)

        return JSONResponse(content={
            "status": status,
            "analysis_ar": analysis,
            "technical_details": technical,
            "model_used": "gemini-2.5-flash",
            "disclaimer": MEDICAL_DISCLAIMER,
        })

    except Exception as ex:

        log.exception("Image analysis failed: %s", ex)

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "analysis_ar": (
                    "حدث خطأ أثناء تحليل الصورة الطبية."
                ),
                "technical_details": str(ex),
                "model_used": "gemini-2.5-flash",
                "disclaimer": MEDICAL_DISCLAIMER,
            }
        )


# ============================================================
# Run Local
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )