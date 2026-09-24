"""Terminal-based RAG chatbot with rule-based response generation."""

import re

from src.preprocessing.paragraph_extractor import ParagraphExtractor
from src.search.query_mapper import QueryMapper
from src.search.retriever import HybridRetriever, RetrievalResult
from src.utils.config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    MAX_PARAGRAPHS,
    MIN_RESPONSE_CONFIDENCE,
)
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


EXIT_COMMANDS = frozenset({
    "exit",
    "quit",
    "finish",
    "stop",
})


MULTI_TOPIC_CONNECTOR_RE = re.compile(
    r"\b(?:and|also|plus)\b",
    re.IGNORECASE,
)


class Chatbot:
    """
    Terminal/UI chatbot that retrieves and displays document content.

    Does NOT generate or invent information.
    It only returns indexed content from the uploaded documents.
    """

    NO_RESULT_MESSAGE = (
        "I couldn't find any relevant information for your question in the "
        "uploaded documents. Please try asking it in a different way."
    )

    LOW_CONFIDENCE_MESSAGE = (
        "I couldn't find a confident answer in the uploaded documents for your "
        "question. Please try rephrasing or asking about a specific policy topic."
    )

    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever
        self.paragraph_extractor = ParagraphExtractor()

    # ============================================================
    # PUBLIC METHOD FOR STREAMLIT / UI
    # ============================================================

    def ask(self, question: str) -> str:
        """
        Answer a question using the existing RAG pipeline.

        This method is intended to be called by the Streamlit UI
        or any other frontend.
        """

        question = question.strip()

        if not question:
            return self.NO_RESULT_MESSAGE

        return self._generate_response(question)

    # ============================================================
    # TERMINAL CHAT
    # ============================================================

    def start(self) -> None:
        """Start the interactive terminal chatbot loop."""

        while True:

            try:
                user_input = input("You: ").strip()

            except (EOFError, KeyboardInterrupt):

                print("\nThanks for chatting!")
                break

            if not user_input:
                continue

            if user_input.lower() in EXIT_COMMANDS:

                print("Thanks for chatting!")
                break

            response = self.ask(user_input)

            print(f"Bot: {response}")

    # ============================================================
    # GENERATE RESPONSE
    # ============================================================

    def _generate_response(self, query: str) -> str:
        """
        Retrieve relevant content and format a rule-based response.
        """

        query_lower = query.lower()

        # --------------------------------------------------------
        # Out of scope
        # --------------------------------------------------------

        if QueryMapper.is_out_of_scope_query(query_lower):

            logger.info(
                "Out-of-scope query refused: %s",
                query,
            )

            return self.NO_RESULT_MESSAGE

        # --------------------------------------------------------
        # Retrieve documents
        # --------------------------------------------------------

        results = self.retriever.retrieve(query)

        if not results:

            logger.info(
                "No results for query: %s",
                query,
            )

            return self.NO_RESULT_MESSAGE

        # --------------------------------------------------------
        # Reject semantic-only matches
        # --------------------------------------------------------

        top = results[0]

        if (
            top.keyword_score == 0
            and top.regex_score == 0
        ):

            logger.info(
                "Rejected semantic-only match for query: %s",
                query,
            )

            return self.NO_RESULT_MESSAGE

        # --------------------------------------------------------
        # Confidence check
        # --------------------------------------------------------

        if top.combined_score < MIN_RESPONSE_CONFIDENCE:

            logger.info(
                "Low confidence (%.3f) for query: %s",
                top.combined_score,
                query,
            )

            return self.LOW_CONFIDENCE_MESSAGE

        # --------------------------------------------------------
        # Multi-topic question
        # --------------------------------------------------------

        if self._is_multi_topic_query(query):

            return self._format_multi_answer(
                results,
                query,
            )

        # --------------------------------------------------------
        # Normal question
        # --------------------------------------------------------

        return self._format_answer(
            results[0],
            query,
        )

    # ============================================================
    # MULTI-TOPIC ANSWER
    # ============================================================

    def _format_multi_answer(
        self,
        results: list[RetrievalResult],
        query: str,
    ) -> str:
        """
        Combine answers from distinct sections for multi-topic queries.
        """

        answers: list[str] = []
        seen_sections: set[str] = set()

        target_sections = {
            section.lower()
            for section in self.retriever._resolve_target_sections(
                query.lower(),
                self.retriever._extract_query_tokens(
                    query.lower()
                ),
            )
        }

        prioritized = sorted(
            results,
            key=lambda result: (
                0
                if str(
                    result.metadata.get(
                        "section_title",
                        "",
                    )
                ).lower() in target_sections
                else 1,
                -result.combined_score,
            ),
        )

        for result in prioritized:

            if len(answers) >= MAX_PARAGRAPHS:
                break

            section_title = str(
                result.metadata.get(
                    "section_title",
                    "",
                )
            )

            section_lower = section_title.lower()

            if (
                section_title
                and section_title in seen_sections
            ):
                continue

            if (
                section_lower
                == "frequently asked questions"
                and target_sections
                and not target_sections.intersection(
                    {"frequently asked questions"}
                )
            ):
                continue

            extracted = self.paragraph_extractor.extract(
                result.text,
                query,
                max_paragraphs=1,
            )

            if not extracted:
                continue

            extracted = self._strip_leading_heading(
                extracted,
                section_title,
            )

            if not extracted:
                continue

            if extracted in answers:
                continue

            answers.append(extracted)

            if section_title:
                seen_sections.add(section_title)

        if not answers:

            return self._format_answer(
                results[0],
                query,
            )

        logger.info(
            "Multi-topic response for '%s' — %d section(s), "
            "scores=%.3f/%.3f",
            query,
            len(answers),
            results[0].combined_score,
            (
                results[1].combined_score
                if len(results) > 1
                else 0.0
            ),
        )

        return "\n\n".join(answers)

    # ============================================================
    # NORMAL ANSWER
    # ============================================================

    def _format_answer(
        self,
        result: RetrievalResult,
        query: str,
    ) -> str:
        """
        Return extracted document text only.

        Metadata is logged, not displayed.
        """

        metadata = result.metadata

        document_name = metadata.get(
            "document_name",
            "Unknown Document",
        )

        page_number = metadata.get(
            "page_number"
        )

        confidence = self._determine_confidence(
            result.combined_score
        )

        extracted = self.paragraph_extractor.extract(
            result.text,
            query,
        )

        if not extracted:

            logger.info(
                "Paragraph extractor found no relevant "
                "paragraph for query: %s",
                query,
            )

            return self.NO_RESULT_MESSAGE

        section_title = str(
            metadata.get(
                "section_title",
                "",
            )
        )

        extracted = self._strip_leading_heading(
            extracted,
            section_title,
        )

        if not QueryMapper.answer_covers_query(
            query.lower(),
            extracted.lower(),
        ):

            logger.info(
                "Answer rejected — content does not cover "
                "query topic: %s",
                query,
            )

            return self.NO_RESULT_MESSAGE

        logger.info(
            "Response for '%s' — doc=%s, page=%s, "
            "confidence=%s, score=%.3f",
            query,
            document_name,
            page_number,
            confidence,
            result.combined_score,
        )

        return extracted

    # ============================================================
    # REMOVE HEADING
    # ============================================================

    @staticmethod
    def _strip_leading_heading(
        text: str,
        heading: str,
    ) -> str:
        """
        Remove a duplicated section heading from
        the start of a response.
        """

        stripped = text.strip()

        if not heading:
            return stripped

        heading_lower = heading.lower()

        prefix = f"{heading}\n"

        if stripped.lower().startswith(
            prefix.lower()
        ):

            remainder = stripped[
                len(prefix):
            ].lstrip(": \n").strip()

            return (
                remainder
                if remainder
                else stripped
            )

        first_line, _, rest = stripped.partition(
            "\n"
        )

        if (
            first_line.strip().lower()
            == heading_lower
        ):

            return (
                rest.strip()
                if rest.strip()
                else stripped
            )

        return stripped

    # ============================================================
    # MULTI-TOPIC DETECTION
    # ============================================================

    @staticmethod
    def _is_multi_topic_query(
        query: str,
    ) -> bool:

        return bool(
            MULTI_TOPIC_CONNECTOR_RE.search(
                query
            )
        )

    # ============================================================
    # CONFIDENCE
    # ============================================================

    @staticmethod
    def _determine_confidence(
        score: float,
    ) -> str:
        """Map combined score to a confidence label."""

        if score >= CONFIDENCE_HIGH:
            return "High"

        if score >= CONFIDENCE_MEDIUM:
            return "Medium"

        return "Low"