import asyncio
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import JSONResponse

from core.config import MAX_IMAGE_BYTES, MAX_IMAGE_MB, EXTERNAL_CALL_TIMEOUT
from core.constants import ALLOWED_IMAGE_TYPES, MEDICAL_DISCLAIMER
from models.schemas import ImageAnalysisResponse
from services.gemini_service import GeminiService

router = APIRouter()

@router.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    gemini_service: GeminiService = request.app.state.gemini

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        return JSONResponse(status_code=400, content={"status":"error","analysis":f"نوع الملف '{file.content_type}' غير مدعوم.","model_used":"none","disclaimer":MEDICAL_DISCLAIMER})
    try: image_bytes = await file.read()
    except Exception: return JSONResponse(status_code=400, content={"status":"error","analysis":"فشل في قراءة الملف.","model_used":"none","disclaimer":MEDICAL_DISCLAIMER})
    if not image_bytes: return JSONResponse(status_code=400, content={"status":"error","analysis":"الملف المرفوع فارغ.","model_used":"none","disclaimer":MEDICAL_DISCLAIMER})
    if len(image_bytes) > MAX_IMAGE_BYTES: return JSONResponse(status_code=413, content={"status":"error","analysis":f"حجم الصورة يتجاوز {MAX_IMAGE_MB}MB.","model_used":"none","disclaimer":MEDICAL_DISCLAIMER})
    try:
        status, analysis, model = await asyncio.wait_for(gemini_service.analyze_image(image_bytes, file.content_type), timeout=EXTERNAL_CALL_TIMEOUT)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=200, content={"status":"fallback","analysis":"انتهت مهلة تحليل الصورة، يرجى المحاولة مرة أخرى أو استشارة طبيب.","model_used":"timeout","disclaimer":MEDICAL_DISCLAIMER})
        
    if status == "error":
        status = "fallback"
        
    return JSONResponse(status_code=200, content={"status":status,"analysis":analysis,"model_used":model,"disclaimer":MEDICAL_DISCLAIMER})
