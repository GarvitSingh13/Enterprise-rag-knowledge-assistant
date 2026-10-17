"""Prompt Versioning & Management Engine.

Loads, validates, and manages version-controlled system prompt templates (YAML).
Enables dynamic version switching, rollbacks, and parameter substitution.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from pydantic import BaseModel, Field

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


class PromptTemplate(BaseModel):
    """Schema for a versioned enterprise prompt template."""
    version: str
    name: str
    author: str
    created_date: str
    description: str
    system_instructions: str
    template: str

    def format(self, **kwargs: Any) -> str:
        """Safely format template substituting provided variables."""
        # Provide defaults for missing optional variables
        defaults = {
            "context": "[No context documents provided]",
            "chat_history": "[No previous conversation]",
            "query": "",
        }
        params = {**defaults, **kwargs}
        return self.template.format(**params)


class PromptManager:
    """Manages versioned prompts, template loading, and version switching."""

    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or TEMPLATES_DIR
        self.templates: Dict[str, PromptTemplate] = {}
        self.active_version: str = "1.0.0"
        self._load_all_templates()

    def _load_all_templates(self) -> None:
        """Scan directory and load all YAML templates."""
        if not self.templates_dir.exists():
            return

        for yml_file in self.templates_dir.glob("*.yaml"):
            try:
                with open(yml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                template = PromptTemplate(**data)
                self.templates[template.version] = template
            except Exception as e:
                print(f"[Warning] Failed to load prompt template {yml_file.name}: {e}")

        # Set default active version to highest version
        if self.templates:
            sorted_versions = sorted(self.templates.keys())
            self.active_version = sorted_versions[-1]

    def get_template(self, version: Optional[str] = None) -> PromptTemplate:
        """Retrieve a specific prompt version or the active version."""
        target_v = version or self.active_version
        if target_v not in self.templates:
            # Fallback to any available template
            if self.templates:
                return next(iter(self.templates.values()))
            raise ValueError(f"No prompt template found for version '{target_v}'")
        return self.templates[target_v]

    def set_active_version(self, version: str) -> None:
        """Switch active prompt version."""
        if version not in self.templates:
            raise ValueError(f"Cannot switch to unknown version: '{version}'. Available: {list(self.templates.keys())}")
        self.active_version = version

    def list_versions(self) -> List[Dict[str, str]]:
        """List all available prompt versions with metadata."""
        return [
            {
                "version": t.version,
                "name": t.name,
                "author": t.author,
                "description": t.description,
                "is_active": t.version == self.active_version,
            }
            for t in self.templates.values()
        ]
