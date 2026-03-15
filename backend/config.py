"""
Configuration module for the Multimodal Deep Learning API.
Uses Pydantic Settings for environment variable support.
"""
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    api_title: str = "Multimodal Deep Learning API"
    api_version: str = "1.0.0"
    debug: bool = False
    
    database_url: str = "sqlite:///backend/database/predictions_v2.db"
    db_pool_size: int = 5
    db_max_overflow: int = 10
    
    max_file_size_mb: int = 10  # Maximum file size in MB
    max_text_size_mb: int = 5
    max_image_size_mb: int = 10
    max_audio_size_mb: int = 50
    max_video_size_mb: int = 100
    max_tabular_size_mb: int = 10
    
    rate_limit_requests: int = 100  # requests per window
    rate_limit_window_seconds: int = 60  # window in seconds
    
    request_timeout_seconds: int = 300
    
    cors_origins: str = "http://localhost,http://localhost:3000,http://localhost:8080"
    
    @property
    def cors_origins_list(self) -> list:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
    
    vector_dimension: int = 768
    faiss_nlist: int = 100  # Number of clusters for IVF index
    
    redis_url: Optional[str] = None
    cache_ttl_seconds: int = 3600  # Default cache TTL
    
    storage_backend: str = "local"  # Options: local, azure
    
    azure_storage_connection_string: Optional[str] = None
    azure_container_name: Optional[str] = None
    
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 30
    retry_max_attempts: int = 3
    retry_base_delay: float = 0.5
    
    use_inference_api: bool = False
    custom_model_repo: Optional[str] = "nithin12342/h5-omnifusion-v3"
    custom_model_filename: str = "h5_omnifusion.pt"
    huggingface_token: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore unknown env vars


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
