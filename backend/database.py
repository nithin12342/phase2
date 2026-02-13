"""
Database module with support for SQLite (dev) and PostgreSQL (prod).
Includes connection pooling configuration.
"""
from sqlalchemy import create_engine, Column, String, Text, DateTime, event, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool, StaticPool
from datetime import datetime, timezone
import os
import logging

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

DATABASE_DIR = "backend/database"

os.makedirs(DATABASE_DIR, exist_ok=True)


def get_database_url():
    """Get database URL from settings or environment."""
    db_url = settings.database_url
    
    if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:////"):
        return db_url
    
    return db_url


def create_engine_with_config():
    """Create SQLAlchemy engine with appropriate configuration."""
    db_url = get_database_url()
    
    if db_url.startswith("sqlite"):
        logger.info("Using SQLite database")
        engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,  # Better for SQLite with threads
            echo=settings.debug
        )
        
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.close()
            
    elif db_url.startswith("postgresql"):
        logger.info("Using PostgreSQL database")
        engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,   # Recycle connections after 1 hour
            echo=settings.debug
        )
    else:
        logger.info(f"Using database: {db_url.split('@')[0] if '@' in db_url else db_url}")
        engine = create_engine(
            db_url,
            pool_pre_ping=True,
            echo=settings.debug
        )
    
    return engine


engine = create_engine_with_config()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class PredictionHistory(Base):
    """Model for storing prediction history."""
    __tablename__ = "prediction_history"

    id = Column(String, primary_key=True, index=True)
    text_filename = Column(String, nullable=True)
    image_filename = Column(String, nullable=True)
    audio_filename = Column(String, nullable=True)
    video_filename = Column(String, nullable=True)
    tabular_filename = Column(String, nullable=True)
    prediction = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    def __repr__(self):
        return f"<PredictionHistory(id={self.id}, prediction={self.prediction[:50]}...)>"


class SurveyResponse(Base):
    """Model for storing detailed mental health survey responses."""
    __tablename__ = "survey_responses"

    id = Column(String, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    
    gender = Column(String, nullable=True)
    country = Column(String, nullable=True)
    occupation = Column(String, nullable=True)
    days_indoors = Column(String, nullable=True) # Stored as string to handle range/values flexibly
    
    is_self_employed = Column(String, nullable=True)
    self_employed_date = Column(String, nullable=True) # ISO format date string
    
    growing_stress = Column(String, nullable=True)
    changes_habits = Column(String, nullable=True)
    mental_health_history = Column(String, nullable=True)
    family_history = Column(String, nullable=True)
    treatment_sought = Column(String, nullable=True)
    mood_swings = Column(String, nullable=True)
    work_interest = Column(String, nullable=True)
    social_weakness = Column(String, nullable=True)
    coping_struggles = Column(String, nullable=True)
    
    interview_attended = Column(String, nullable=True)
    care_options_awareness = Column(String, nullable=True)
    
    video_filename = Column(String, nullable=True)
    audio_filename = Column(String, nullable=True)
    doc_filename = Column(String, nullable=True)
    photo_filename = Column(String, nullable=True)
    
    depression_risk = Column(String, nullable=True)

    def __repr__(self):
        return f"<SurveyResponse(id={self.id}, country={self.country})>"


def get_db():
    """Dependency to get the database session with proper cleanup."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_db_and_tables():
    """Create all database tables."""
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully.")


def check_database_connection() -> bool:
    """Check if database is accessible."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
