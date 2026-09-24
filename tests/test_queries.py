"""Temporary script to verify query matching improvements."""

import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

from main import index_all_documents
from src.database.chroma_manager import ChromaManager
from src.embeddings.embedding_model import EmbeddingModel
from src.search.retriever import HybridRetriever
from src.chatbot.chatbot import Chatbot

print("Re-indexing...")
index_all_documents(show_progress=False)

# Initialize chatbot
chroma = ChromaManager()
emb = EmbeddingModel()
retriever = HybridRetriever(chroma, emb)
bot = Chatbot(retriever)

queries = [
    "I'm a new business owner. How do I get started with Direkt?",
    "Who developed Direkt?",
    "What is the Tax module used for?",
    "tell me about direkt features",
    "how to get start with direkt",
    "tell me about gst management",
    "how direkt manage gst",
    "how to add a product",
    "tell me about tax management",
    "tell me about direkt order purchasing",
    "how to order purchased",
    "tell me account process",
    "tell me about direkt security",
    "direkt is secure or not",
    "direkt is secure",
]

NO_RESULT_PREFIX = "I couldn't find"

print("\n" + "=" * 100)
print("Running Query Tests")
print("=" * 100)

passed = 0
failed = 0

for query in queries:
    response = bot._generate_response(query)

    success = not response.startswith(NO_RESULT_PREFIX)
    status = "OK" if success else "FAIL"

    if success:
        passed += 1
    else:
        failed += 1

    preview = response if len(response) <= 120 else response[:120] + "..."

    print(f"[{status}] {query}")
    print(f"  -> {preview}")
    print()

print("=" * 100)
print(f"Passed : {passed}")
print(f"Failed : {failed}")
print(f"Total  : {len(queries)}")
print("=" * 100)