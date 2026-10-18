"""Role-Based Access Control (RBAC) Permission Evaluator.

Defines clearance policies and generates pre-retrieval search constraints
based on authenticated user roles and organizational boundaries.
"""

from typing import List, Set, Dict, Any, Optional
from enterprise_rag.core.models import ClearanceLevel, Department, Chunk
from enterprise_rag.security.models import TokenPayload, UserRole


class RBACGuard:
    """Evaluates security permissions and enforces clearance boundaries."""

    # Clearance Level -> Set of Roles authorized to access it
    CLEARANCE_ROLE_PERMISSIONS: Dict[ClearanceLevel, Set[str]] = {
        ClearanceLevel.PUBLIC: {"employee", "contractor", "public", "guest"},
        ClearanceLevel.INTERNAL: {
            "employee",
            "engineer",
            "hr_specialist",
            "finance_analyst",
            "manager",
            "c_suite",
            "admin",
        },
        ClearanceLevel.CONFIDENTIAL: {
            "hr_specialist",
            "finance_analyst",
            "engineer",
            "manager",
            "c_suite",
            "admin",
        },
        ClearanceLevel.RESTRICTED: {"c_suite", "admin"},
    }

    @classmethod
    def can_access_clearance(cls, user_roles: List[str], clearance: ClearanceLevel) -> bool:
        """Verify if any user role grants access to the specified clearance tier."""
        allowed = cls.CLEARANCE_ROLE_PERMISSIONS.get(clearance, set())
        return any(role.lower() in allowed for role in user_roles)

    @classmethod
    def can_access_chunk(cls, token_payload: TokenPayload, chunk: Chunk) -> bool:
        """
        Verify if an authenticated user is permitted to view a specific document chunk.
        """
        # 1. Clearance Level Check
        if not cls.can_access_clearance(token_payload.roles, chunk.metadata.clearance_level):
            return False

        # 2. Allowed Roles Intersection Check
        if chunk.metadata.allowed_roles:
            user_roles_set = set(r.lower() for r in token_payload.roles)
            chunk_roles_set = set(r.lower() for r in chunk.metadata.allowed_roles)
            if not user_roles_set.intersection(chunk_roles_set):
                return False

        return True

    @classmethod
    def get_search_filters(cls, token_payload: TokenPayload) -> Dict[str, Any]:
        """
        Derive search constraints to inject directly into the Hybrid Retriever.
        Enforces security at the retrieval level (Pre-Retrieval Pre-Filtering).
        """
        return {
            "allowed_roles": [r.lower() for r in token_payload.roles],
        }
