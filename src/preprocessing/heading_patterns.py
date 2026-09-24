"""Shared heading detection patterns for Direkt manual documents."""

import re

from src.preprocessing.section_registry import SectionRegistry

MAJOR_HEADING_RE = re.compile(
    r"^(?:CHAPTER|SECTION|Part|Article|Appendix)\s+[\dIVXLC]+",
    re.IGNORECASE,
)

MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+")

NUMBERED_SECTION_RE = re.compile(r"^\d+(?:\.\d+)*\s+[A-Za-z]")

# Onboarding steps: "Step 1: Create Your Company"
STEP_HEADING_RE = re.compile(
    r"^Step\s+\d+:\s+[A-Za-z].+$",
    re.IGNORECASE,
)

# Feature modules: "Sales module", "Inventory module"
MODULE_HEADING_RE = re.compile(
    r"^[A-Za-z]+(?:\s+[A-Za-z]+){0,3}\s+module$",
    re.IGNORECASE,
)

# Headings with parenthetical abbreviations: "Omnichannel Point of Sale(POS)"
PAREN_HEADING_RE = re.compile(
    r"^[A-Z][A-Za-z\s]+(?:\([A-Z]+\))$",
)

POLICY_HEADING_RE = re.compile(
    r"^[A-Z][A-Za-z]*(?:\s+(?:and|&)\s+[A-Z][A-Za-z]*)?"
    r"(?:\s+[A-Z][A-Za-z]*){0,4}\s+(?:Policy|Management)$",
)

DASHBOARD_HEADING_RE = re.compile(
    r"^[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3}\s+Dashboard$",
)

# Supports acronym headings such as GST and title-case section names
TITLE_CASE_HEADING_RE = re.compile(
    r"^(?:[A-Z]{2,}|[A-Z][a-z]+)"
    r"(?:\s+(?:[A-Z]{2,}|[A-Z][a-z]+|and|&|of|a|the|[a-z]+)){0,8}$",
)

CATALOGUE_HEADING_RE = re.compile(
    r"^Product\s+Catalogue$", re.IGNORECASE
)

ORDER_PROCESSING_HEADING_RE = re.compile(
    r"^Order\s+Processing$", re.IGNORECASE
)

GETTING_STARTED_HEADING_RE = re.compile(
    r"^Getting\s+Started$", re.IGNORECASE
)

KEY_FEATURES_HEADING_RE = re.compile(
    r"^Key\s+Features$", re.IGNORECASE
)

CREATING_SALE_HEADING_RE = re.compile(
    r"^Creating\s+a\s+Sale$", re.IGNORECASE
)

PURCHASE_ORDERS_HEADING_RE = re.compile(
    r"^Purchase\s+Orders$", re.IGNORECASE
)

FAQ_HEADING_RE = re.compile(r"^Frequently Asked Questions$", re.IGNORECASE)

LIST_ITEM_RE = re.compile(r"^\s*(?:[-•*▪▸]|\d+[.)]|[a-zA-Z][.)])\s+")

TABLE_ROW_RE = re.compile(r"^\s*\|?.+\|.+\|?\s*$")

TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s\-:|]+\|?\s*$")

SUBSECTION_LABEL_RE = re.compile(
    r"^(?:Sample|Example|Note|Important)\s+[A-Za-z\s]+$",
    re.IGNORECASE,
)

# Prose FAQ sentence splitter (used when Q:/A: format is absent)
PROSE_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9₹$])")

# Targeted fact extraction patterns for Direkt manual content
PAYMENT_METHODS_RE = re.compile(
    r"\b(?:supports multiple payment methods, including|payment methods, including)\s+"
    r"([A-Za-z,\s]+?)(?:\.|Dynamic QR)",
    re.IGNORECASE,
)
GST_TYPES_RE = re.compile(
    r"\b(?:calculate[s]?|calculations for)\s+(CGST,\s*SGST,\s*and\s*IGST|"
    r"CGST,\s*SGST,\s*IGST)\b",
    re.IGNORECASE,
)
GSTR_REPORTS_RE = re.compile(
    r"\b(?:generates?|generate)\s+(GSTR-1,\s*GSTR-3B[^.]*|"
    r"GSTR-1,\s*GSTR-3B,\s*Sales Register[^.]*)\b",
    re.IGNORECASE,
)
NAV_PATH_RE = re.compile(
    r"\bgo to\s+([A-Za-z\s]+(?:→|->)\s*[A-Za-z\s]+)",
    re.IGNORECASE,
)
MODULE_FEATURES_RE = re.compile(
    r"\b(Sales|Inventory|Finance|Tax|AI)\s+(?:module\s+)?(?:enables|helps)\s+"
    r"(?:businesses to\s+)?(.+?)(?:\.|$)",
    re.IGNORECASE,
)
DIREKT_MODULES_RE = re.compile(
    r"\b(Sales,\s*Inventory,\s*Finance,\s*Tax,\s*and\s*AI)\b",
    re.IGNORECASE,
)
COMPANY_FIELDS_RE = re.compile(
    r"\b(?:company name|business name|GSTIN|PAN|address|contact information)\b",
    re.IGNORECASE,
)
PRODUCT_FIELDS_RE = re.compile(
    r"\b(?:product name|SKU|barcode|unit|GST rate|selling price|"
    r"purchase price|opening stock|product images?)\b",
    re.IGNORECASE,
)
WAREHOUSE_FIELDS_RE = re.compile(
    r"\b(?:warehouse name|address|manager|contact number)\b",
    re.IGNORECASE,
)
DASHBOARD_METRICS_RE = re.compile(
    r"\b(?:revenue|profit|total orders|top-selling products|top customers|"
    r"stock value|outstanding payments|GST summary|inventory health|purchase trends)\b",
    re.IGNORECASE,
)
SECURITY_FEATURES_RE = re.compile(
    r"\b(?:secure authentication|encrypted communication|role-based access control|"
    r"audit logs|automatic backups|permission management)\b",
    re.IGNORECASE,
)
SHARE_CHANNELS_RE = re.compile(
    r"\b(?:WhatsApp|SMS|QR codes?)\b",
    re.IGNORECASE,
)

_section_titles_pattern: re.Pattern[str] | None = None
_section_titles_key: str = ""


def _build_titles_pattern(titles: tuple[str, ...]) -> re.Pattern[str]:
    if not titles:
        return re.compile(r"(?!x)x")
    alternatives = "|".join(
        re.escape(title) for title in sorted(titles, key=len, reverse=True)
    )
    return re.compile(rf"^(?:{alternatives})$", re.IGNORECASE)


def get_official_section_titles_re() -> re.Pattern[str]:
    """Return a cached regex built from section titles in uploaded documents."""
    global _section_titles_pattern, _section_titles_key

    titles = SectionRegistry.get_titles()
    key = "|".join(titles)
    if _section_titles_pattern is None or _section_titles_key != key:
        _section_titles_key = key
        _section_titles_pattern = _build_titles_pattern(titles)

    return _section_titles_pattern


def is_section_heading(line: str) -> bool:
    """Return True when a line is a document section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 100 or stripped.endswith("."):
        return False
    if MAJOR_HEADING_RE.match(stripped):
        return True
    if MARKDOWN_HEADING_RE.match(stripped):
        return True
    if get_official_section_titles_re().match(stripped):
        return True
    if STEP_HEADING_RE.match(stripped):
        return True
    if MODULE_HEADING_RE.match(stripped):
        return True
    if PAREN_HEADING_RE.match(stripped):
        return True
    if NUMBERED_SECTION_RE.match(stripped):
        return True
    if POLICY_HEADING_RE.match(stripped):
        return True
    if DASHBOARD_HEADING_RE.match(stripped):
        return True
    if CATALOGUE_HEADING_RE.match(stripped):
        return True
    if ORDER_PROCESSING_HEADING_RE.match(stripped):
        return True
    if GETTING_STARTED_HEADING_RE.match(stripped):
        return True
    if KEY_FEATURES_HEADING_RE.match(stripped):
        return True
    if CREATING_SALE_HEADING_RE.match(stripped):
        return True
    if PURCHASE_ORDERS_HEADING_RE.match(stripped):
        return True
    if FAQ_HEADING_RE.match(stripped):
        return True
    if SUBSECTION_LABEL_RE.match(stripped):
        return True
    if TITLE_CASE_HEADING_RE.match(stripped):
        return True
    return False


def is_list_item(line: str) -> bool:
    """Return True when a line is a bullet or numbered list item."""
    return bool(LIST_ITEM_RE.match(line))


def is_table_row(line: str) -> bool:
    """Return True when a line looks like a markdown or pipe-delimited table row."""
    if TABLE_SEPARATOR_RE.match(line):
        return True
    return bool(TABLE_ROW_RE.match(line))
