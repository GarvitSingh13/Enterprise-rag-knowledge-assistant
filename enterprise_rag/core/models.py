"""Core data models and schemas for Enterprise RAG Knowledge Assistant.

Defines Pydantic models for documents, metadata, security clearance levels,
departments, and hierarchical (parent-child) chunks.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import uuid
from datetime import datetime, timezone


class Department(str, Enum):
    """Enterprise organizational departments."""
    ENGINEERING = "engineering"
    HR = "hr"
    FINANCE = "finance"
    LEGAL = "legal"
    EXECUTIVE = "executive"
    GENERAL = "general"


class ClearanceLevel(str, Enum):
    """Security clearance levels for Role-Based Access Control (RBAC)."""
    PUBLIC = "public"            # Visible to all employees and contractors
    INTERNAL = "internal"        # Visible to all authenticated employees
    CONFIDENTIAL = "confidential"# Restricted to specific departments (e.g. HR or Finance)
    RESTRICTED = "restricted"    # Restricted to C-suite and executives only


class DocumentMetadata(BaseModel):
    """Structured enterprise metadata attached to each document and chunk."""
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(default="Untitled Document")
    department: Department = Field(default=Department.GENERAL)
    clearance_level: ClearanceLevel = Field(default=ClearanceLevel.INTERNAL)
    document_type: str = Field(default="documentation")  # policy, sop, handbook, memo, faq
    author: Optional[str] = Field(default="Enterprise Admin")
    version: str = Field(default="1.0.0")
    created_date: Optional[str] = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    tags: List[str] = Field(default_factory=list)
    source_file: str = Field(default="")
    allowed_roles: List[str] = Field(default_factory=lambda: ["employee"])
    extra_attributes: Dict[str, Any] = Field(default_factory=dict)


class RawDocument(BaseModel):
    """Represents an ingested raw document before chunking."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: DocumentMetadata

    @property
    def character_count(self) -> int:
        return len(self.content)


class ChunkType(str, Enum):
    """Type of chunk in hierarchical chunking."""
    PARENT = "parent"  # Larger chunk (1000-1500 chars) retaining broad contextual nuance
    CHILD = "child"    # Smaller chunk (250-400 chars) optimized for high-precision retrieval


class Chunk(BaseModel):
    """Represents a text chunk produced by the chunking engine."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None  # Populated for child chunks to reference their parent
    content: str
    chunk_type: ChunkType
    chunk_index: int
    metadata: DocumentMetadata
    character_count: int = 0
    estimated_token_count: int = 0

    def model_post_init(self, __context: Any) -> None:
        if self.character_count == 0:
            self.character_count = len(self.content)
        if self.estimated_token_count == 0:
            # Rule of thumb for English text: ~4 characters per token
            self.estimated_token_count = max(1, len(self.content) // 4)
