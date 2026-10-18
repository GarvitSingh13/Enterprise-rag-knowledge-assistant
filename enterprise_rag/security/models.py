"""Security Data Models and User Schemas for Enterprise RBAC."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid
from enterprise_rag.core.models import Department, ClearanceLevel


class UserRole(str, Enum):
    """Enterprise user role categories."""
    EMPLOYEE = "employee"
    ENGINEER = "engineer"
    HR_SPECIALIST = "hr_specialist"
    FINANCE_ANALYST = "finance_analyst"
    MANAGER = "manager"
    C_SUITE = "c_suite"
    ADMIN = "admin"


class User(BaseModel):
    """Internal user record stored in authentication registry."""
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    full_name: str
    department: Department
    roles: List[UserRole]
    hashed_password: str
    salt: str


class TokenPayload(BaseModel):
    """Decoded JWT claims payload."""
    sub: str = Field(description="Subject identifier (user_id)")
    username: str
    full_name: str
    department: str
    roles: List[str]
    exp: int
    iat: int


class UserPublicProfile(BaseModel):
    """Sanitized public user profile."""
    user_id: str
    username: str
    full_name: str
    department: Department
    roles: List[UserRole]
