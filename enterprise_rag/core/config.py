"""Configuration settings for Enterprise RAG Knowledge Assistant."""

import os
from pathlib import Path
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
PROCESSED_CHUNKS_DIR = DATA_DIR / "processed_chunks"


class ChunkingConfig(BaseModel):
    """Configuration parameters for Parent-Child Hierarchical Chunking."""
    parent_chunk_size: int = Field(default=1200, description="Target character length for parent context chunks")
    parent_chunk_overlap: int = Field(default=150, description="Overlap between consecutive parent chunks")
    child_chunk_size: int = Field(default=300, description="Target character length for child search chunks")
    child_chunk_overlap: int = Field(default=50, description="Overlap between consecutive child chunks")
    min_chunk_length: int = Field(default=50, description="Discard any chunk smaller than this length")


class Settings(BaseModel):
    """Global system configuration."""
    app_name: str = "Enterprise RAG Knowledge Assistant"
    env: str = os.getenv("APP_ENV", "development")
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    raw_docs_dir: Path = RAW_DOCS_DIR
    processed_chunks_dir: Path = PROCESSED_CHUNKS_DIR


settings = Settings()
