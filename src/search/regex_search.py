"""Regex search engine tuned for Direkt manual documents."""

import re
from dataclasses import dataclass
from typing import Any, Pattern

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class RegexPattern:
    """A named regex pattern for structured data extraction."""

    name: str
    pattern: str
    description: str = ""


class RegexSearchEngine:
    """Regex search optimized for Direkt business platform content."""

    DEFAULT_PATTERNS: list[RegexPattern] = [
        # GST and tax identifiers
        RegexPattern("gstin", r"\bGSTIN\b", "GST Identification Number"),
        RegexPattern("pan", r"\bPAN\b", "Permanent Account Number"),
        RegexPattern(
            "gst_type",
            r"\b(?:CGST|SGST|IGST)\b",
            "GST Tax Components",
        ),
        RegexPattern(
            "gstr_report",
            r"\bGSTR-1\b|\bGSTR-3B\b",
            "GSTR Reports",
        ),
        RegexPattern(
            "gst_register",
            r"\b(?:Sales Register|Purchase Register|ITC Register)\b",
            "GST Registers",
        ),
        # Product and inventory identifiers
        RegexPattern("sku", r"\bSKU\b", "Stock Keeping Unit"),
        RegexPattern("barcode", r"\bbarcode\b", "Product Barcode"),
        RegexPattern(
            "product_field",
            r"\b(?:selling price|purchase price|opening stock|GST rate)\b",
            "Product Fields",
        ),
        # Navigation paths in the manual
        RegexPattern(
            "nav_path",
            r"\b(?:Settings|Inventory|Catalogue|POS)\s*(?:→|->)\s*[A-Za-z\s]+",
            "Navigation Path",
        ),
        # Payment methods
        RegexPattern(
            "payment_method",
            r"\b(?:Cash|UPI|Bank Transfer|Credit|Cheque|Mixed Payments?)\b",
            "Payment Method",
        ),
        RegexPattern(
            "qr_code",
            r"\b(?:QR code|Dynamic QR)\b",
            "QR Code Payment",
        ),
        # Business modules
        RegexPattern(
            "direkt_module",
            r"\b(?:Sales|Inventory|Finance|Tax|AI)\s+module\b",
            "Direkt Feature Module",
        ),
        RegexPattern(
            "module_list",
            r"\bSales,\s*Inventory,\s*Finance,\s*Tax,\s*and\s*AI\b",
            "All Direkt Modules",
        ),
        # Setup steps
        RegexPattern(
            "setup_step",
            r"\bStep\s+\d+:\s+[A-Za-z\s]+",
            "Setup Step",
        ),
        # Sharing channels
        RegexPattern(
            "share_channel",
            r"\b(?:WhatsApp|SMS|QR code)\b",
            "Catalogue Share Channel",
        ),
        # Security features
        RegexPattern(
            "security_feature",
            r"\b(?:role-based access control|audit logs|automatic backups|"
            r"encrypted communication|secure authentication|permission management)\b",
            "Security Feature",
        ),
        # Analytics metrics
        RegexPattern(
            "dashboard_metric",
            r"\b(?:revenue|profit|top-selling products|top customers|"
            r"stock value|outstanding payments|inventory health|purchase trends)\b",
            "Dashboard Metric",
        ),
        # Accounting
        RegexPattern(
            "accounting",
            r"\b(?:double-entry accounting|journal entry|cash and bank book|"
            r"payment reconciliation)\b",
            "Accounting Feature",
        ),
        # Inventory operations
        RegexPattern(
            "inventory_op",
            r"\b(?:stock transfer|stock adjustment|low stock alert|"
            r"batch tracking|barcode scanning)\b",
            "Inventory Operation",
        ),
        # POS and sales
        RegexPattern(
            "pos_feature",
            r"\b(?:POS billing|quotations?|sales orders?|credit sales?|"
            r"customer ledger|omnichannel)\b",
            "POS/Sales Feature",
        ),
        # AI features
        RegexPattern(
            "ai_feature",
            r"\b(?:sales forecasts?|inventory predictions?|"
            r"purchase recommendations?|AI-based recommendations?|business insights)\b",
            "AI Feature",
        ),
        # FAQ
        RegexPattern(
            "faq_question",
            r"Q:\s*.+?(?=\s+A:)",
            "FAQ Question",
        ),
        RegexPattern(
            "faq_answer",
            r"A:\s*.+?(?=\s+Q:|$)",
            "FAQ Answer",
        ),
        RegexPattern(
            "list_item",
            r"(?:^|\n)\s*(?:[-•*]|\d+[.)])\s+.+\S",
            "List Item",
        ),
    ]

    def __init__(self) -> None:
        self._patterns: dict[str, Pattern[str]] = {}
        self._register_default_patterns()

    def add_pattern(self, regex_pattern: RegexPattern) -> None:
        try:
            self._patterns[regex_pattern.name] = re.compile(
                regex_pattern.pattern, re.IGNORECASE
            )
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern: {regex_pattern.name}") from exc

    def search(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        query_patterns = self._extract_patterns_from_text(query)
        query_tokens = set(re.findall(r"\b[A-Za-z0-9@._%-]+\b", query))
        query_lower = query.lower()

        if not query_patterns and not query_tokens:
            return []

        results: list[dict[str, Any]] = []

        for doc in documents:
            text = doc.get("text", "")
            if not text:
                continue

            text_lower = doc.get("text_lower") or text.lower()

            score = self._score_document(
                text, text_lower, query_patterns, query_tokens, query_lower
            )

            if score > 0:
                results.append(
                    {
                        "chunk_id": doc["chunk_id"],
                        "text": doc["text"],
                        "metadata": doc.get("metadata", {}),
                        "score": score,
                        "search_type": "regex",
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _register_default_patterns(self) -> None:
        for pattern in self.DEFAULT_PATTERNS:
            try:
                self._patterns[pattern.name] = re.compile(pattern.pattern, re.IGNORECASE)
            except re.error as exc:
                logger.warning("Skipping invalid pattern '%s': %s", pattern.name, exc)

    def _extract_patterns_from_text(self, text: str) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for name, compiled in self._patterns.items():
            matches = compiled.findall(text)
            if matches:
                found[name] = matches if isinstance(matches[0], str) else [m[0] for m in matches]
        return found

    def _score_document(
        self,
        text: str,
        text_lower: str,
        query_patterns: dict[str, list[str]],
        query_tokens: set[str],
        query_lower: str,
    ) -> float:
        score = 0.0

        for pattern_name, query_matches in query_patterns.items():
            if pattern_name in self._patterns:
                doc_matches = self._patterns[pattern_name].findall(text)
                if doc_matches:
                    match_count = sum(
                        1 for qm in query_matches
                        if any(str(qm).lower() in str(dm).lower() for dm in doc_matches)
                    )
                    score += match_count * 0.4

        for token in query_tokens:
            if len(token) >= 3 and token.lower() in text_lower:
                score += 0.12

        if any(w in query_lower for w in ("how", "what", "when", "who", "where", "many")):
            if "q:" in text_lower and "a:" in text_lower:
                score += 0.3
            elif "frequently asked questions" in text_lower:
                score += 0.2

        if any(w in query_lower for w in ("gst", "cgst", "sgst", "igst", "gstr", "tax")):
            if any(kw in text_lower for kw in ("gst", "cgst", "sgst", "igst", "gstr", "itc")):
                score += 0.28

        if any(w in query_lower for w in ("payment", "upi", "cash", "cheque", "qr")):
            if any(kw in text_lower for kw in ("payment", "upi", "cash", "cheque", "qr code")):
                score += 0.25

        if any(w in query_lower for w in ("inventory", "stock", "warehouse", "barcode")):
            if any(kw in text_lower for kw in ("inventory", "stock", "warehouse", "barcode", "sku")):
                score += 0.25

        if any(w in query_lower for w in ("pos", "sale", "billing", "invoice")):
            if any(kw in text_lower for kw in ("pos", "sale", "billing", "invoice", "quotation")):
                score += 0.25

        if any(w in query_lower for w in ("ai", "forecast", "prediction", "recommendation")):
            if any(kw in text_lower for kw in ("ai", "forecast", "prediction", "recommendation", "insights")):
                score += 0.22

        if any(w in query_lower for w in ("catalogue", "catalog", "order", "whatsapp")):
            if any(kw in text_lower for kw in ("catalogue", "order", "whatsapp", "sms", "qr")):
                score += 0.22

        if any(w in query_lower for w in ("security", "authentication", "backup", "permission")):
            if any(kw in text_lower for kw in ("security", "authentication", "backup", "permission", "encrypted")):
                score += 0.22

        if any(w in query_lower for w in ("dashboard", "analytics", "revenue", "profit")):
            if any(kw in text_lower for kw in ("dashboard", "analytics", "revenue", "profit", "performance")):
                score += 0.22

        if any(w in query_lower for w in ("company", "setup", "getting started", "warehouse", "product")):
            if any(kw in text_lower for kw in ("company", "warehouse", "product", "getting started", "step")):
                score += 0.20

        return min(score, 1.0)
