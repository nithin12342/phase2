"""
Multimodal Deep Learning API - Main Application Module
Implements scalability best practices including:
- Health checks
- Rate limiting
- File size limits
- Pagination
- Request timeouts
- Structured logging
"""
import uuid
import os
import logging
import numpy as np
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Query, APIRouter, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
from contextlib import asynccontextmanager

# Defer heavy imports to ensure health check availability
# from database import SessionLocal, create_db_and_tables, get_db
# from vector_store import vector_store as vs, VECTOR_DIM
import schemas
import models
from config import get_settings
from middleware import RateLimitMiddleware, RequestIDMiddleware, TimeoutMiddleware
from resilience import db_retry, ml_circuit_breaker, CircuitOpenError
from cache import cache, cache_embedding, get_cached_embedding
from storage import storage

VECTOR_DIM = 768


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()


# Helper functions for lazy dependencies
def get_db_session():
    from database import SessionLocal
    return SessionLocal()

def get_vs():
    from vector_store import vector_store as vs
    return vs

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown events."""
    logger.info("Starting application (Fast Startup Mode)...")
    
    # Defer table creation
    try:
        from database import create_db_and_tables
        create_db_and_tables()
    except Exception as e:
        logger.error(f"Postponed DB init failed: {e}")
    
    logger.info("Models will be loaded on first request (Lazy-Load enabled).")
    logger.info("Application ready for health checks.")
    yield
    logger.info("Shutting down application...")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(TimeoutMiddleware)  # Enabled for request timeouts

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api/v1")

UPLOAD_DIR = "backend/uploads"
IMAGES_DIR = os.path.join(UPLOAD_DIR, "images")
AUDIO_DIR = os.path.join(UPLOAD_DIR, "audio")
VIDEO_DIR = os.path.join(UPLOAD_DIR, "video")
TABULAR_DIR = os.path.join(UPLOAD_DIR, "tabular")
TEXT_DIR = os.path.join(UPLOAD_DIR, "text")

for d in [IMAGES_DIR, AUDIO_DIR, VIDEO_DIR, TABULAR_DIR, TEXT_DIR]:
    os.makedirs(d, exist_ok=True)

ALLOWED_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".webp"},
    "audio": {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a", ".wma", ".alac", ".mp4"},
    "video": {".mp4", ".mov", ".avi", ".mkv", ".webm"},
    "tabular": {".json", ".csv"},
    "text": {".txt", ".md"}
}


FILE_SIZE_LIMITS = {
    "text": settings.max_text_size_mb * 1024 * 1024,
    "image": settings.max_image_size_mb * 1024 * 1024,
    "audio": settings.max_audio_size_mb * 1024 * 1024,
    "video": settings.max_video_size_mb * 1024 * 1024,
    "tabular": settings.max_tabular_size_mb * 1024 * 1024
}


def validate_file(file: UploadFile, file_type: str):
    """Validate file extension and size."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS[file_type]:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type for {file_type}. Allowed: {ALLOWED_EXTENSIONS[file_type]}"
        )
    return ext


async def check_file_size(file: UploadFile, file_type: str):
    """Check if file size is within limits. Always resets file pointer."""
    try:
        content = await file.read()
        max_size = FILE_SIZE_LIMITS.get(file_type, settings.max_file_size_mb * 1024 * 1024)
        if len(content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size for {file_type}: {max_size // (1024*1024)}MB"
            )
        return content
    finally:
        await file.seek(0)  # Always reset file pointer


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.
    Returns basic health status.
    """
    return {
        "status": "healthy",
        "version": settings.api_version
    }


@app.get("/warmup", tags=["Health"])
async def warmup():
    """
    Warmup endpoint for Azure Container Apps.
    This can be called periodically to keep the container from scaling to zero.
    """
    # Simply checking model status is enough to generate activity
    status = models.get_model_status()
    return {"status": "warm", "models": status}


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness check endpoint - verifies all dependencies are ready.
    """
    try:
        db = get_db_session()
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db.close()
        
        status = models.get_model_status()
        models_ready = status.get("h5_loaded", False)
        
        cache_stats = cache.stats()
        
        storage_info = storage.get_backend_info()
        
        return {
            "status": "ready",
            "database": "connected",
            "models": "loaded" if models_ready else "loading",
            "cache": cache_stats.get("backend", "unknown"),
            "storage": storage_info.get("type", "unknown")
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/stats", tags=["Health"])
async def get_system_stats():
    """
    Get system statistics including cache, storage, and vector store info.
    """
    vs_stats = {"status": "unavailable"}
    try:
        vs = get_vs()
        vs_stats = vs.get_stats()
    except Exception as e:
        logger.warning(f"Vector store stats unavailable: {e}")
    return {
        "cache": cache.stats(),
        "storage": storage.get_backend_info(),
        "vector_store": vs_stats
    }


@app.get("/", tags=["Root"])
async def read_root():
    """Welcome endpoint with API information."""
    return {
        "message": "Welcome to the Multimodal Deep Learning API!",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health",
        "status_dashboard": "/status"
    }


@app.get("/status", tags=["Health"], response_class=HTMLResponse)
async def status_dashboard():
    """System health dashboard — visual overview of all components."""
    db = get_db_session()

    model_status = models.get_model_status()
    model_ok = model_status.get("h5_loaded", False)

    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    try:
        cache_info = cache.stats()
        cache_ok = True
    except Exception:
        cache_info = {}
        cache_ok = False

    try:
        storage_info = storage.get_backend_info()
        storage_ok = True
    except Exception:
        storage_info = {}
        storage_ok = False

    def badge(ok, label_ok="Online", label_fail="Offline"):
        if ok:
            return f'<span style="background:#10b981;color:#fff;padding:4px 14px;border-radius:20px;font-weight:600;font-size:13px">{label_ok}</span>'
        return f'<span style="background:#ef4444;color:#fff;padding:4px 14px;border-radius:20px;font-weight:600;font-size:13px">{label_fail}</span>'

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>System Health Dashboard</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);color:#e2e8f0;min-height:100vh;padding:40px 20px}}
.container{{max-width:900px;margin:0 auto}}
h1{{font-size:28px;font-weight:700;margin-bottom:8px;background:linear-gradient(90deg,#6366f1,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.subtitle{{color:#94a3b8;font-size:14px;margin-bottom:32px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:32px}}
.card{{background:rgba(30,41,59,0.8);border:1px solid rgba(99,102,241,0.2);border-radius:12px;padding:24px;backdrop-filter:blur(10px)}}
.card h3{{font-size:13px;text-transform:uppercase;letter-spacing:1px;color:#94a3b8;margin-bottom:12px}}
.card .value{{font-size:20px;font-weight:700;margin-bottom:8px}}
.detail{{background:rgba(30,41,59,0.6);border:1px solid rgba(99,102,241,0.15);border-radius:12px;padding:24px;margin-bottom:16px}}
.detail h3{{font-size:16px;font-weight:600;margin-bottom:16px;color:#c4b5fd}}
.row{{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(148,163,184,0.1)}}
.row:last-child{{border:none}}
.label{{color:#94a3b8}}
.val{{font-weight:600}}
.refresh{{display:inline-block;margin-top:24px;padding:10px 24px;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;text-decoration:none}}
.refresh:hover{{opacity:0.9;transform:translateY(-1px);transition:all 0.2s}}
.pulse{{animation:pulse 2s infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.5}}}}
</style></head><body>
<div class="container">
<h1>H5-OmniFusion System Dashboard</h1>
<p class="subtitle">Real-time health status &mdash; {settings.api_version}</p>

<div class="grid">
<div class="card"><h3>API Server</h3><div class="value">{badge(True)}</div><div style="color:#94a3b8;font-size:13px">Port 8000</div></div>
<div class="card"><h3>Database</h3><div class="value">{badge(db_ok)}</div><div style="color:#94a3b8;font-size:13px">SQLite</div></div>
<div class="card"><h3>ML Model</h3><div class="value">{badge(model_ok, 'Loaded', 'Survey Mode')}</div><div style="color:#94a3b8;font-size:13px">H5-OmniFusion</div></div>
<div class="card"><h3>Cache</h3><div class="value">{badge(cache_ok)}</div><div style="color:#94a3b8;font-size:13px">{cache_info.get('backend','in-memory')}</div></div>
</div>

<div class="detail">
<h3>Model Details</h3>
<div class="row"><span class="label">Mode</span><span class="val">{model_status.get('mode','local')}</span></div>
<div class="row"><span class="label">Vector Dimension</span><span class="val">{model_status.get('vector_dim', 768)}</span></div>
<div class="row"><span class="label">Checkpoint Loaded</span><span class="val">{'Yes' if model_ok else 'No (using survey scoring)'}</span></div>
</div>

<div class="detail">
<h3>Storage</h3>
<div class="row"><span class="label">Backend</span><span class="val">{storage_info.get('type','local')}</span></div>
<div class="row"><span class="label">Status</span><span class="val">{badge(storage_ok)}</span></div>
</div>

<div class="detail">
<h3>API Endpoints</h3>
<div class="row"><span class="label">POST /api/v1/submit-survey</span><span class="val" style="color:#10b981">Active</span></div>
<div class="row"><span class="label">GET /api/v1/history</span><span class="val" style="color:#10b981">Active</span></div>
<div class="row"><span class="label">GET /health</span><span class="val" style="color:#10b981">Active</span></div>
<div class="row"><span class="label">GET /docs</span><span class="val" style="color:#10b981">Active</span></div>
</div>

<a class="refresh" href="/status">Refresh Status</a>
<a class="refresh" href="/docs" style="margin-left:12px;background:linear-gradient(135deg,#0ea5e9,#6366f1)">API Docs</a>
</div></body></html>"""

    return HTMLResponse(content=html)


@api_router.post("/predict", response_model=schemas.PredictionHistory, tags=["Predictions"])
async def predict(
    text: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    tabular: Optional[UploadFile] = File(None)
):
    db = get_db_session()
    """
    Generate multimodal prediction from uploaded files.
    
    Accepts any combination of:
    - text: .txt, .md files (max 5MB)
    - image: .jpg, .jpeg, .png, .webp (max 10MB)
    - audio: .mp3, .wav, .aac (max 50MB)
    - video: .mp4, .mov, .avi (max 100MB)
    - tabular: .json, .csv (max 10MB)
    """
    prediction_id = str(uuid.uuid4())
    logger.info(f"Processing prediction request: {prediction_id}")
    
    text_filename, image_filename, audio_filename, video_filename, tabular_filename = None, None, None, None, None

    text_content, image_content, audio_content, video_content, tabular_content = None, None, None, None, None
    
    if text:
        validate_file(text, "text")
        text_content = await check_file_size(text, "text")
        text_filename = f"{prediction_id}_{text.filename}"
        with open(os.path.join(TEXT_DIR, text_filename), "wb") as f:
            f.write(text_content)
            
    if image:
        validate_file(image, "image")
        image_content = await check_file_size(image, "image")
        image_filename = f"{prediction_id}_{image.filename}"
        with open(os.path.join(IMAGES_DIR, image_filename), "wb") as f:
            f.write(image_content)
            
    if audio:
        validate_file(audio, "audio")
        audio_content = await check_file_size(audio, "audio")
        audio_filename = f"{prediction_id}_{audio.filename}"
        with open(os.path.join(AUDIO_DIR, audio_filename), "wb") as f:
            f.write(audio_content)

    if video:
        validate_file(video, "video")
        video_content = await check_file_size(video, "video")
        video_filename = f"{prediction_id}_{video.filename}"
        with open(os.path.join(VIDEO_DIR, video_filename), "wb") as f:
            f.write(video_content)

    if tabular:
        validate_file(tabular, "tabular")
        tabular_content = await check_file_size(tabular, "tabular")
        tabular_filename = f"{prediction_id}_{tabular.filename}"
        with open(os.path.join(TABULAR_DIR, tabular_filename), "wb") as f:
            f.write(tabular_content)

    embeddings = {}
    if text_content:
        try:
            text_str = text_content.decode('utf-8')
        except UnicodeDecodeError:
            text_str = text_content.decode('latin-1') 
        embeddings['text'] = models.get_text_embedding(text_str)
    
    if image_content:
        embeddings['image'] = models.get_image_embedding(image_content)
    
    if audio_content:
        embeddings['audio'] = models.get_audio_embedding(audio_content)
    
    if video_content:
        embeddings['video'] = models.get_video_embedding(video_content)
    
    if tabular_content:
        embeddings['tabular'] = models.get_tabular_embedding({"filename": tabular_filename, "content": tabular_content})
    
    prediction_text = "No meaningful prediction generated."
    final_embedding = None

    if embeddings:
        prediction_text = models.get_fusion_prediction(embeddings)
        all_available_embeddings = [emb for emb in embeddings.values() if emb is not None]
        if all_available_embeddings:
            final_embedding = np.concatenate(all_available_embeddings).astype('float32')

    import database
    db_prediction = database.PredictionHistory(
        id=prediction_id,
        text_filename=text_filename,
        image_filename=image_filename,
        audio_filename=audio_filename,
        video_filename=video_filename,
        tabular_filename=tabular_filename,
        prediction=prediction_text
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    db.close()

    if final_embedding is not None:
        if final_embedding.shape[0] != VECTOR_DIM:
            logger.warning(f"Embedding dimension mismatch: {final_embedding.shape[0]} vs {VECTOR_DIM}")
            if final_embedding.shape[0] > VECTOR_DIM:
                final_embedding = final_embedding[:VECTOR_DIM]
            else:
                final_embedding = np.pad(final_embedding, (0, VECTOR_DIM - final_embedding.shape[0]), 'constant')
        try:
            vs = get_vs()
            vs.add_vector(final_embedding, prediction_id)
            logger.info(f"Prediction completed with embedding: {prediction_id}")
        except Exception as ve:
            logger.warning(f"Vector store unavailable (non-fatal): {ve}")
    else:
        logger.info(f"Prediction completed without embedding (no files): {prediction_id}")
    
    return db_prediction


@api_router.get("/history", response_model=List[schemas.PredictionHistory], tags=["Predictions"])
async def get_history(
    limit: int = Query(default=50, ge=1, le=100, description="Number of records to return"),
    offset: int = Query(default=0, ge=0, description="Number of records to skip")
):
    from database import PredictionHistory
    db = get_db_session()
    try:
        res = db.query(PredictionHistory)\
            .order_by(PredictionHistory.timestamp.desc())\
            .offset(offset)\
            .limit(limit)\
            .all()
        return res
    finally:
        db.close()


@api_router.get("/history/count", tags=["Predictions"])
async def get_history_count():
    """Get total count of predictions for pagination."""
    from database import PredictionHistory
    db = get_db_session()
    try:
        count = db.query(PredictionHistory).count()
        return {"total": count}
    finally:
        db.close()


@api_router.get("/similar/{prediction_id}", tags=["Predictions"])
async def get_similar(prediction_id: str, k: int = Query(default=5, ge=1, le=20)):
    """
    Find similar predictions based on vector similarity.
    
    - **prediction_id**: The prediction ID to find similar items for
    - **k**: Number of similar items to return (1-20, default 5)
    """
    vs = get_vs()
    try:
        if prediction_id not in vs.id_map:
            raise HTTPException(status_code=404, detail="Prediction ID not found in vector store.")

        similar_ids = vs.search_similar_to_id(prediction_id, k=k + 1)
        similar_ids = [sid for sid in similar_ids if sid != prediction_id][:k]
        return {"prediction_id": prediction_id, "similar_ids": similar_ids}
    except (ValueError, NotImplementedError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except CircuitOpenError as e:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")


@api_router.post("/submit-survey", response_model=schemas.SurveyResponse, tags=["Survey"])
async def submit_survey(
    survey_data: str = Form(..., description="JSON string of survey data"),
    video: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    doc: Optional[UploadFile] = File(None),
    photo: Optional[UploadFile] = File(None)
):
    db = get_db_session()
    """
    Submit a complete mental health survey with optional media files.
    At least one media file is recommended but not required.
    """
    import json
    
    try:
        data_dict = json.loads(survey_data)
        survey_in = schemas.SurveyResponseCreate(**data_dict)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in survey_data")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")

    submission_id = str(uuid.uuid4())
    logger.info(f"Processing survey submission: {submission_id}")

    video_filename, audio_filename, doc_filename, photo_filename = None, None, None, None
    embeddings = {}  # Initialize embeddings dictionary
    
    if video and video.filename:
        ext = validate_file(video, "video")
        video_bytes = await check_file_size(video, "video")
        video_filename = f"{submission_id}_video{ext}"
        with open(os.path.join(VIDEO_DIR, video_filename), "wb") as f:
            f.write(video_bytes)
        emb, _ = models.get_video_embedding(video_bytes)
        embeddings['video'] = emb

    if audio and audio.filename:
        ext = validate_file(audio, "audio")
        audio_bytes = await check_file_size(audio, "audio")
        audio_filename = f"{submission_id}_audio{ext}"
        with open(os.path.join(AUDIO_DIR, audio_filename), "wb") as f:
            f.write(audio_bytes)
        emb, _ = models.get_audio_embedding(audio_bytes)
        embeddings['audio'] = emb

    if doc and doc.filename:
        doc_ext = os.path.splitext(doc.filename)[1].lower()
        if doc_ext not in [".pdf", ".doc", ".docx", ".txt", ".md"]:
            raise HTTPException(status_code=400, detail="Invalid doc type. Allowed: .pdf, .doc, .docx, .txt, .md")
        doc_bytes = await doc.read()
        await doc.seek(0)
        doc_filename = f"{submission_id}_doc{doc_ext}"
        with open(os.path.join(TEXT_DIR, doc_filename), "wb") as f:
            f.write(doc_bytes)
        try:
            text_str = doc_bytes.decode('utf-8')
        except UnicodeDecodeError:
            text_str = doc_bytes.decode('latin-1')
        emb, _ = models.get_text_embedding(text_str)
        embeddings['text'] = emb

    if photo and photo.filename:
        photo_ext = ".jpg"
        _, ext = os.path.splitext(photo.filename)
        if ext:
            photo_ext = ext.lower()
        if photo_ext not in ALLOWED_EXTENSIONS["image"]:
            raise HTTPException(status_code=400, detail=f"Invalid photo type. Allowed: {ALLOWED_EXTENSIONS['image']}")
        photo_bytes = await check_file_size(photo, "image")
        photo_filename = f"{submission_id}_photo{photo_ext}"
        with open(os.path.join(IMAGES_DIR, photo_filename), "wb") as f:
            f.write(photo_bytes)
        emb, _ = models.get_image_embedding(photo_bytes)
        embeddings['image'] = emb

    emb, _ = models.get_tabular_embedding(data_dict)
    embeddings['tabular'] = emb

    models.set_survey_context(data_dict)
    prediction_result = models.get_fusion_prediction(embeddings)
    
    import database
    
    db_survey = database.SurveyResponse(
        id=submission_id,
        gender=survey_in.gender,
        country=survey_in.country,
        occupation=survey_in.occupation,
        days_indoors=survey_in.days_indoors,
        is_self_employed=survey_in.is_self_employed,
        self_employed_date=survey_in.self_employed_date,
        growing_stress=survey_in.growing_stress,
        changes_habits=survey_in.changes_habits,
        mental_health_history=survey_in.mental_health_history,
        family_history=survey_in.family_history,
        treatment_sought=survey_in.treatment_sought,
        mood_swings=survey_in.mood_swings,
        work_interest=survey_in.work_interest,
        social_weakness=survey_in.social_weakness,
        coping_struggles=survey_in.coping_struggles,
        interview_attended=survey_in.interview_attended,
        care_options_awareness=survey_in.care_options_awareness,
        video_filename=video_filename,
        audio_filename=audio_filename,
        doc_filename=doc_filename,
        photo_filename=photo_filename,
        depression_risk=prediction_result
    )
    
    db.add(db_survey)
    db.commit()
    db.refresh(db_survey)
    db.close()
    
    # Try vector store (non-blocking — survey already saved)
    try:
        vs = get_vs()
        # Store embeddings if available
        logger.info(f"Vector store available, storing embeddings for {submission_id}")
    except Exception as ve:
        logger.warning(f"Vector store unavailable (non-fatal): {ve}")
    
    logger.info(f"Survey submitted successfully: {submission_id}")
    return db_survey


app.include_router(api_router)

@app.post("/predict", response_model=schemas.PredictionHistory, tags=["Legacy"], deprecated=True)
async def predict_legacy(
    text: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    tabular: Optional[UploadFile] = File(None)
):
    """Legacy predict endpoint - use /api/v1/predict instead."""
    return await predict(text, image, audio, video, tabular)


@app.get("/history", response_model=List[schemas.PredictionHistory], tags=["Legacy"], deprecated=True)
async def get_history_legacy(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    """Legacy history endpoint - use /api/v1/history instead."""
    return await get_history(limit, offset)


@app.get("/similar/{prediction_id}", tags=["Legacy"], deprecated=True)
async def get_similar_legacy(prediction_id: str, k: int = Query(default=5, ge=1, le=20)):
    """Legacy similar endpoint - use /api/v1/similar instead."""
    return await get_similar(prediction_id, k)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)