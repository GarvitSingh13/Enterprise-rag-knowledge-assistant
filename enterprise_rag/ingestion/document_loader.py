"""Enterprise Document Loader.

Loads files across common enterprise formats (.md, .txt, .json, .pdf),
extracts YAML frontmatter if present, and returns RawDocument instances.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import json
from enterprise_rag.core.models import RawDocument, DocumentMetadata
from enterprise_rag.ingestion.metadata_extractor import MetadataExtractor


class DocumentLoader:
    """Loads enterprise documents from filesystem."""

    SUPPORTED_EXTENSIONS = {".md", ".txt", ".json", ".pdf"}

    @staticmethod
    def _parse_frontmatter(raw_text: str) -> Tuple[Dict[str, Any], str]:
        """Extract simple key: value frontmatter enclosed between '---' markers."""
        frontmatter = {}
        content = raw_text.strip()

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                header = parts[1].strip()
                content = parts[2].strip()
                for line in header.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        k = k.strip()
                        v = v.strip()
                        # Simple list parsing: [a, b, c]
                        if v.startswith("[") and v.endswith("]"):
                            items = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
                            frontmatter[k] = items
                        else:
                            frontmatter[k] = v.strip("'\"")

        return frontmatter, content

    @classmethod
    def load_file(cls, file_path: Path) -> Optional[RawDocument]:
        """Load a single file into a RawDocument instance."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            return None

        content = ""
        frontmatter: Dict[str, Any] = {}

        if ext in {".md", ".txt"}:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
            frontmatter, content = cls._parse_frontmatter(raw_text)

        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                content = data.get("content", str(data))
                frontmatter = {k: v for k, v in data.items() if k != "content"}
            else:
                content = str(data)

        elif ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(path))
                pages_text = []
                for p in reader.pages:
                    extracted = p.extract_text()
                    if extracted:
                        pages_text.append(extracted)
                content = "\n\n".join(pages_text)
            except ImportError:
                # Graceful fallback if pypdf is not yet installed
                content = f"[PDF file: {path.name} - Install pypdf to extract full text]"

        if not content.strip():
            return None

        metadata = MetadataExtractor.extract_from_path(path, frontmatter)
        return RawDocument(content=content, metadata=metadata)

    @classmethod
    def load_directory(cls, dir_path: Path) -> List[RawDocument]:
        """Recursively scan directory and load all supported documents."""
        directory = Path(dir_path)
        if not directory.exists():
            return []

        documents: List[RawDocument] = []
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS:
                doc = cls.load_file(file_path)
                if doc:
                    documents.append(doc)

        return documents
