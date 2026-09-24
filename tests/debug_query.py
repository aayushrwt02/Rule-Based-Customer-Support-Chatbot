"""Debug remaining failing queries."""

import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

from src.database.chroma_manager import ChromaManager
from src.embeddings.embedding_model import EmbeddingModel
from src.search.retriever import HybridRetriever
from src.chatbot.chatbot import Chatbot

# Initialize components
chroma = ChromaManager()
emb = EmbeddingModel()
retriever = HybridRetriever(chroma, emb)
bot = Chatbot(retriever)

queries = [
    "create a sale",
    "how to create a sale",
    "how to make a sale",
    "how to create warehouse",
    "tell me about direkt security",
    "Does Direkt provide an API?",
    "Does Direkt support offline mode?",
    "Which database does Direkt use?",
    "What is Direkt's monthly subscription price?",
    "Does Direkt support SAP integration?",
    "tell me about ai management",
    "what is sale module",
    "POS features",
    "introduce direkt",
    "benefits of direkt",
]

for q in queries:
    tokens = retriever._extract_query_tokens(q.lower())
    targets = retriever._resolve_target_sections(q.lower(), tokens)
    results = retriever.retrieve(q)

    print("=" * 80)
    print(f"Q: {q}")
    print(f"Tokens: {tokens}")
    print(f"Target Sections: {targets}")

    if results:
        top = results[0]

        print(f"Section: {top.metadata.get('section_title', 'Unknown')}")
        print(f"Score: {top.metadata.get('score', 'N/A')}")

        extracted = bot.paragraph_extractor.extract(top.text, q)

        print(f"Extracted Empty: {not extracted}")

        if extracted:
            print("\nExtracted Text:")
            print("-" * 80)
            print(extracted)
            print("-" * 80)
    response = bot._generate_response(q)
    print(f"Bot Response: {response[:150]}{'...' if len(response) > 150 else ''}")
    print()