"""Logging configuration — file-only output to keep the terminal clean."""

import logging
import warnings

from src.utils.config import LOG_FILE, PROJECT_ROOT

_CONFIGURED = False

warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "chromadb",
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
    "urllib3",
    "filelock",
    "torch",
    "onnxruntime",
)


def setup_logger(name: str = "rag_chatbot", level: int = logging.DEBUG) -> logging.Logger:
    """Configure loggers to write only to logs/chatbot.log (not the terminal)."""
    global _CONFIGURED

    if not _CONFIGURED:
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)

        for logger_name in NOISY_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

        _CONFIGURED = True

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True
    return logger
