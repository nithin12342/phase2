"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone


class PredictionHistoryBase(BaseModel):
    """Base schema for prediction history."""
    text_filename: Optional[str] = Field(None, description="Uploaded text file name")
    image_filename: Optional[str] = Field(None, description="Uploaded image file name")
    audio_filename: Optional[str] = Field(None, description="Uploaded audio file name")
    video_filename: Optional[str] = Field(None, description="Uploaded video file name")
    tabular_filename: Optional[str] = Field(None, description="Uploaded tabular file name")
    prediction: str = Field(..., description="Model prediction output")


class PredictionHistoryCreate(PredictionHistoryBase):
    """Schema for creating a prediction."""
    pass


class PredictionHistory(PredictionHistoryBase):
    """Schema for prediction history response."""
    id: str = Field(..., description="Unique prediction ID")
    timestamp: datetime = Field(..., description="Prediction timestamp")

    class Config:
        from_attributes = True  # Updated from orm_mode for Pydantic v2


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    version: Optional[str] = None


class ReadinessResponse(BaseModel):
    """Readiness check response schema."""
    status: str
    database: str
    models: str


class PaginationParams(BaseModel):
    """Pagination parameters."""
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class SimilarPredictionsResponse(BaseModel):
    """Response for similar predictions."""
    prediction_id: str
    similar_ids: list[str]


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str


class SurveyResponseBase(BaseModel):
    """Base schema for survey response."""
    gender: Optional[str] = None
    country: Optional[str] = None
    occupation: Optional[str] = None
    days_indoors: Optional[str] = None
    is_self_employed: Optional[str] = None
    self_employed_date: Optional[str] = None
    growing_stress: Optional[str] = None
    changes_habits: Optional[str] = None
    mental_health_history: Optional[str] = None
    family_history: Optional[str] = None
    treatment_sought: Optional[str] = None
    mood_swings: Optional[str] = None
    work_interest: Optional[str] = None
    social_weakness: Optional[str] = None
    coping_struggles: Optional[str] = None
    interview_attended: Optional[str] = None
    care_options_awareness: Optional[str] = None


class SurveyResponseCreate(SurveyResponseBase):
    """Schema for creating a survey response (from JSON string parse)."""
    pass


class SurveyResponse(SurveyResponseBase):
    """Schema for survey response output."""
    id: str
    timestamp: datetime
    video_filename: Optional[str] = None
    audio_filename: Optional[str] = None
    doc_filename: Optional[str] = None
    photo_filename: Optional[str] = None
    depression_risk: Optional[str] = None

    class Config:
        from_attributes = True


class ChatMessage(BaseModel):
    """Schema for a single chat message."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatResponse(BaseModel):
    """Schema for chat response (non-streaming, if needed)."""
    response: str

