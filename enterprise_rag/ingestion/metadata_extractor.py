"""Metadata Extractor for enterprise documents.

Extracts structured metadata (department, clearance level, doc type, tags)
from file paths, frontmatter, and content heuristics.
"""

from pathlib import Path
from typing import Dict, Any, List
from enterprise_rag.core.models import Department, ClearanceLevel, DocumentMetadata


class MetadataExtractor:
    """Extracts and infers metadata from file paths and content."""

    DEPARTMENT_MAP = {
        "engineering": Department.ENGINEERING,
        "dev": Department.ENGINEERING,
        "tech": Department.ENGINEERING,
        "hr": Department.HR,
        "people": Department.HR,
        "talent": Department.HR,
        "finance": Department.FINANCE,
        "accounting": Department.FINANCE,
        "legal": Department.LEGAL,
        "compliance": Department.LEGAL,
        "executive": Department.EXECUTIVE,
        "c_suite": Department.EXECUTIVE,
        "leadership": Department.EXECUTIVE,
    }

    CLEARANCE_MAP = {
        "restricted": ClearanceLevel.RESTRICTED,
        "secret": ClearanceLevel.RESTRICTED,
        "executive": ClearanceLevel.RESTRICTED,
        "confidential": ClearanceLevel.CONFIDENTIAL,
        "private": ClearanceLevel.CONFIDENTIAL,
        "internal": ClearanceLevel.INTERNAL,
        "public": ClearanceLevel.PUBLIC,
    }

    ROLE_MAP = {
        ClearanceLevel.PUBLIC: ["employee", "contractor", "public"],
        ClearanceLevel.INTERNAL: ["employee", "manager", "admin", "executive"],
        ClearanceLevel.CONFIDENTIAL: ["manager", "admin", "executive", "hr_specialist", "finance_analyst"],
        ClearanceLevel.RESTRICTED: ["c_suite", "executive", "super_admin"],
    }

    @classmethod
    def extract_from_path(cls, file_path: Path, parsed_frontmatter: Dict[str, Any] = None) -> DocumentMetadata:
        """Infer metadata from path segments and merge with optional frontmatter."""
        path_str = str(file_path).lower().replace("\\", "/")
        path_parts = [p.lower() for p in file_path.parts]

        # 1. Infer department
        department = Department.GENERAL
        for key, dept in cls.DEPARTMENT_MAP.items():
            if any(key in part for part in path_parts):
                department = dept
                break

        # 2. Infer clearance level
        clearance = ClearanceLevel.INTERNAL
        for key, cl in cls.CLEARANCE_MAP.items():
            if key in path_str:
                clearance = cl
                break

        # 3. Infer document type
        doc_type = "documentation"
        filename = file_path.stem.lower()
        if "sop" in filename or "procedure" in filename:
            doc_type = "sop"
        elif "policy" in filename:
            doc_type = "policy"
        elif "handbook" in filename or "guide" in filename:
            doc_type = "guide"
        elif "memo" in filename:
            doc_type = "memo"

        # 4. Generate human-readable title
        title = file_path.stem.replace("_", " ").replace("-", " ").title()

        # 5. Extract tags
        tags: List[str] = [department.value, doc_type, clearance.value]
        if "error" in filename or "incident" in filename:
            tags.append("troubleshooting")
        if "travel" in filename or "expense" in filename:
            tags.append("expenses")
        if "leave" in filename or "remote" in filename:
            tags.append("workplace")

        # 6. Build base metadata
        meta = DocumentMetadata(
            title=title,
            department=department,
            clearance_level=clearance,
            document_type=doc_type,
            source_file=file_path.name,
            tags=list(set(tags)),
            allowed_roles=cls.ROLE_MAP.get(clearance, ["employee"]),
        )

        # 7. Merge frontmatter overrides if provided
        if parsed_frontmatter:
            if "title" in parsed_frontmatter:
                meta.title = str(parsed_frontmatter["title"])
            if "department" in parsed_frontmatter:
                dept_val = str(parsed_frontmatter["department"]).lower()
                meta.department = cls.DEPARTMENT_MAP.get(dept_val, Department.GENERAL)
            if "clearance" in parsed_frontmatter or "clearance_level" in parsed_frontmatter:
                cl_val = str(parsed_frontmatter.get("clearance") or parsed_frontmatter.get("clearance_level")).lower()
                meta.clearance_level = cls.CLEARANCE_MAP.get(cl_val, ClearanceLevel.INTERNAL)
                meta.allowed_roles = cls.ROLE_MAP.get(meta.clearance_level, ["employee"])
            if "document_type" in parsed_frontmatter:
                meta.document_type = str(parsed_frontmatter["document_type"])
            if "author" in parsed_frontmatter:
                meta.author = str(parsed_frontmatter["author"])
            if "tags" in parsed_frontmatter and isinstance(parsed_frontmatter["tags"], list):
                meta.tags = list(set(meta.tags + [str(t) for t in parsed_frontmatter["tags"]]))

        return meta
