import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Use the environment variable if available (from Docker), otherwise use a local SQLite file for direct running
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nvh_diagnostics.db")

# SQLite needs check_same_thread=False, PostgreSQL doesn't care but we handle kwargs
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class DiagnosticRecord(Base):
    __tablename__ = "diagnostic_records"

    id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    module_used = Column(String)
    fault_detected = Column(Boolean)
    fault_type = Column(String)
    severity_score = Column(Float)
    rul_cycles = Column(Float, nullable=True)
    maintenance_alert = Column(String)
    confidence = Column(Float)
    processing_time_ms = Column(Float)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
