"""Registry of section titles discovered from uploaded documents."""

import json
import re
from pathlib import Path

from src.utils.config import VECTOR_DB_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

REGISTRY_FILE: Path = VECTOR_DB_DIR / ".section_registry.json"

# Fallback titles when registry has not been built yet (matches Direkt manual)
DEFAULT_SECTION_TITLES: tuple[str, ...] = (
    "Introduction",
    "About Direkt",
    "Key Features",
    "Sales module",
    "Inventory module",
    "Finance module",
    "Tax module",
    "AI module",
    "Getting Started",
    "Step 1: Create Your Company",
    "Step 2: Create Warehouses",
    "Step 3: Add Products",
    "Product Catalogue",
    "Order Processing",
    "Omnichannel Point of Sale(POS)",
    "Creating a Sale",
    "Payments",
    "Inventory Management",
    "Purchase Orders",
    "Accounting",
    "GST",
    "Analytics Dashboard",
    "Security",
    "Frequently Asked Questions",
)


class SectionRegistry:
    """Stores section titles extracted during indexing for retrieval and parsing."""

    _titles: list[str] = []
    _loaded: bool = False

    @classmethod
    def get_titles(cls) -> tuple[str, ...]:
        """Return registered section titles, loading from disk when needed."""
        if not cls._loaded:
            cls._load_from_disk()
        if cls._titles:
            return tuple(cls._titles)
        return DEFAULT_SECTION_TITLES

    @classmethod
    def clear_and_set(cls, titles: list[str]) -> None:
        """Replace registry with titles discovered during indexing."""
        unique: list[str] = []
        seen: set[str] = set()
        for title in titles:
            cleaned = title.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                unique.append(cleaned)

        cls._titles = unique
        cls._loaded = True
        cls._save_to_disk()
        logger.info("Registered %d section title(s) from uploaded documents.", len(unique))

    @classmethod
    def build_title_synonyms(cls) -> dict[str, str]:
        """Build keyword-to-section mappings from discovered titles."""
        synonyms: dict[str, str] = {}
        skip_words = {
            "and", "the", "for", "your", "a", "of", "to", "in", "on",
            "module", "modules", "step", "create", "add", "sale", "point",
            "management", "processing", "orders", "order", "features",
            "feature", "about", "with", "from", "into", "through",
        }

        for title in cls.get_titles():
            words = re.findall(r"\b[a-zA-Z]{3,}\b", title.lower())
            for word in words:
                if word in skip_words:
                    continue
                if word not in synonyms:
                    synonyms[word] = title

            # Map module names directly
            module_match = re.match(r"^(\w+)\s+module$", title, re.IGNORECASE)
            if module_match:
                synonyms[module_match.group(1).lower()] = title

            # Map step titles by step number keyword
            step_match = re.match(r"^step\s+(\d+):", title, re.IGNORECASE)
            if step_match:
                synonyms[f"step{step_match.group(1)}"] = title

            first_word = words[0] if words else ""
            if (
                first_word
                and first_word not in skip_words
                and first_word not in synonyms
            ):
                synonyms[first_word] = title

        return synonyms

    @classmethod
    def _load_from_disk(cls) -> None:
        cls._loaded = True
        if not REGISTRY_FILE.exists():
            return

        try:
            data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
            cls._titles = list(data.get("section_titles", []))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load section registry: %s", exc)
            cls._titles = []

    @classmethod
    def _save_to_disk(cls) -> None:
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        REGISTRY_FILE.write_text(
            json.dumps({"section_titles": cls._titles}, indent=2),
            encoding="utf-8",
        )
