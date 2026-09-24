"""Application configuration settings — tuned for Direkt manual documents."""

from pathlib import Path

# Project root directory (parent of src/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Directory paths
DOCUMENTS_DIR: Path = PROJECT_ROOT / "documents"
VECTOR_DB_DIR: Path = PROJECT_ROOT / "vector_db"
LOG_FILE: Path = PROJECT_ROOT / "logs" / "chatbot.log"
INDEX_VERSION_FILE: Path = VECTOR_DB_DIR / ".index_version"
DOCUMENTS_FINGERPRINT_FILE: Path = VECTOR_DB_DIR / ".documents_fingerprint"

# ChromaDB collection — Direkt Business OS knowledge base
COLLECTION_NAME: str = "direkt_manual_primary_kb"

# Index version — bump to force re-index after config/document changes
INDEX_VERSION: str = "10"

# Text chunking — sections are short prose blocks (~30–90 words); FAQ ~85 words
CHUNK_SIZE_WORDS: int = 120
CHUNK_OVERLAP_WORDS: int = 10

# Paragraph extraction
MAX_PARAGRAPHS: int = 3

# Embedding model (runs locally via SentenceTransformers)
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE: int = 64

# Retrieval settings
TOP_K_RESULTS: int = 3
KEYWORD_WEIGHT: float = 0.55
REGEX_WEIGHT: float = 0.25
SEMANTIC_WEIGHT: float = 0.20

# Minimum combined score to consider a result relevant
MIN_RELEVANCE_SCORE: float = 0.35

# Minimum score to return an answer
MIN_RESPONSE_CONFIDENCE: float = 0.50

# Topics not covered in the Direkt manual — queries about these should be refused
OUT_OF_SCOPE_KEYWORDS: tuple[str, ...] = (
    "api", "rest api", "graphql", "webhook", "webhooks", "sdk",
    "offline mode", "offline", "without internet",
    "oracle", "oracle database", "mysql", "postgresql", "mongodb", "sql server",
    "database does", "which database", "what database",
    "subscription price", "monthly price", "monthly subscription", "pricing plan",
    "price plan", "cost per month", "how much does", "subscription cost",
    "sap integration", "sap", "erp integration", "tally integration",
    "salesforce", "zoho integration", "quickbooks integration",
    "mobile app store", "ios app", "android app",
    "source code", "open source", "github",
)

# Skip semantic search when keyword match is very strong (faster responses)
KEYWORD_FAST_PATH_THRESHOLD: float = 0.80

# Strong boost when a FAQ section closely matches the user query
FAQ_MATCH_BOOST: float = 0.20

# Boost when query intent maps to a known section title
SECTION_TITLE_MATCH_BOOST: float = 0.15

# Boost when query maps directly to a single official section title
SECTION_EXACT_MATCH_BOOST: float = 0.30

# Max cached query embeddings for repeated questions
QUERY_EMBEDDING_CACHE_SIZE: int = 256

# Supported file extensions
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".csv": "csv",
    ".json": "json",
    ".txt": "txt",
}

# Confidence thresholds
CONFIDENCE_HIGH: float = 0.75
CONFIDENCE_MEDIUM: float = 0.50

# Synonym groups for better query matching against Direkt manual content
QUERY_SYNONYMS: dict[str, list[str]] = {
    "direkt": [
        "direkt", "direct", "business operating system", "bos",
        "glovomega", "platform", "ai-powered", "operating system",
        "business os", "unified platform", "automation platform",
        "glovomega technologies", "overview", "product overview",
        "platform overview", "introduce direkt", "introduce",
        "benefits of direkt", "benefits", "advantages",
    ],
    "developer": [
        "developed", "developer", "developers", "glovomega",
        "technologies", "created", "built", "made", "who developed",
        "who created", "who made", "pvt", "ltd", "manufacturer",
        "company", "who owns",
    ],
    "sales": [
        "sales", "pos", "billing", "invoice", "quotation", "quotations",
        "sales order", "credit sale", "customer ledger", "point of sale",
    ],
    "inventory": [
        "inventory", "stock", "warehouse", "warehouses", "barcode",
        "batch", "transfer", "adjustment", "low stock", "sku",
    ],
    "finance": [
        "finance", "financial", "accounting", "double-entry", "journal",
        "cash book", "bank book", "reconciliation", "ledger",
        "account", "accounts", "account process", "accounting process",
        "financial records", "financial transaction",
    ],
    "tax": [
        "tax", "gst", "cgst", "sgst", "igst", "itc", "gstr",
        "gst billing", "gst compliance", "gst register",
        "tax management", "gst management", "manage gst",
        "tax compliance", "gst filing", "gst calculate",
        "gst invoice", "gstr1", "gstr3b",
    ],
    "ai": [
        "ai", "artificial intelligence", "forecast", "prediction",
        "recommendation", "insights", "purchase recommendation",
    ],
    "catalogue": [
        "catalogue", "catalog", "digital catalogue", "product catalogue",
        "online", "showcase", "browse",
    ],
    "order": [
        "order", "orders", "checkout", "cart", "b2b", "purchase order",
        "order processing", "place order", "purchasing", "order purchasing",
        "procurement", "purchase orders",
    ],
    "payment": [
        "payment", "payments", "upi", "cash", "cheque", "credit",
        "bank transfer", "qr code", "mixed payment", "receivable",
    ],
    "pos": [
        "pos", "point of sale", "omnichannel", "retail", "wholesale",
        "mobile billing", "trade show", "dispatch",
    ],
    "security": [
        "security",  "secure", "safe", "authentication", "authorization",
        "encrypted", "encryption", "data safety", "protection",
        "role-based", "audit", "backup", "permission", "access control",
        "protected", "data security", "data safe", "is secure", "role based",
        "role based access", "audit log", "encrypted",
    ],
    "analytics": [
        "analytics", "dashboard", "revenue", "profit", "performance",
        "top-selling", "top customers", "stock value", "trends",
    ],
    "company": [
        "company", "business", "gstin", "pan", "setup", "create company",
        "business details", "contact information",
    ],
    "product": [
        "product", "products", "sku", "item", "items", "goods", 
        "barcode", "unit", "price", "add product", "create product",
        "new product", "product creation", "product setup", "product master",
        "selling price", "purchase price", "opening stock", "image",
    ],
    "warehouse": [
        "warehouse", "warehouses", "manager", "address", "add warehouse",
        "create warehouse", "create a warehouse",
    ],
    "getting_started": [
        "getting started", "begin", "start", "setup", "onboard",
        "first time", "new user", "get started", "get started with",
        "get start", "new business owner", "business owner",
        "how do i get started", "first steps", "initial setup",
        "begin using", "start with direkt", "new business",
        "how to get started", "starting", "beginning", "new business",
        "first login", "setup", "onboarding", "how to start",
        "where to start", "starting with direkt", "using direkt first time",
    ],
    "gst_filing": [
        "gstr-1", "gstr-3b", "gstr", "sales register", "purchase register",
        "itc register", "filing", "compliance",
    ],
    "features": [
        "feature","features", "modules", "key features", "capabilities",
        "sales module", "inventory module", "finance module",
        "tax module", "ai module", "direkt features", "key functionalities",
        "what can direkt do", "functionality", "what does direkt",
        "module used for", "used for", "functions", "functionalities",
        "capabilities", "advantages", "benefits", "services", 
        "what does direkt offer", "tell me about direkt features", 
        "what is the feature of direkt",
    ],
    "supplier": [
        "supplier", "suppliers", "purchase order", "approve", "send order",
    ],
    "customer" : [
        "customer", "customers", "ledger", "outstanding", "reminder",
        "customer group", "price tier",
    ],
    "whatsapp": [
        "whatsapp", "sms", "qr", "qr code", "share", "link", "web link",
    ],
    "purchase": [
        "purchase", "purchasing", "purchase order", "purchase orders",
        "buy", "buying", "supplier", "vendor", "procurement",
    ],
    "accounting": [
        "account", "accounts", "accounting", "ledger", "journal",
        "double entry", "financial accounting", "finance process",
        "account process", "cash book", "bank book",
    ],
}

# FAQ prose topic keywords — used to split long FAQ sections into topic chunks
FAQ_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "orders": [
        "order", "catalogue", "whatsapp", "sms", "qr", "web link", "install",
        "browse", "checkout", "cart", "place order", "purchasing",
    ],
    "warehouse": [
        "warehouse", "warehouses", "centralized", "inventory management",
        "multiple warehouses", "create warehouse", "add warehouse",
    ],
    "gst": [
        "gst", "invoice", "cgst", "sgst", "igst", "tax",
        "gst management", "tax management", "manage gst", "filing",
        "compliance", "gstr", "itc",
    ],
    "accounting": [
        "accounting", "double-entry", "financial transaction", "recorded",
        "account", "accounts", "account process", "financial records",
    ],
    "payments": [
        "payment", "receivable", "outstanding", "reminder", "customer payments",
    ],
    "getting_started": [
        "company", "warehouse", "product", "get started", "new user",
        "creating", "adding", "entering", "started", "begin", "new business",
         "business owner", "first time", "setup",
    ],
    "features": [
        "sales", "inventory", "finance", "tax", "ai", "modules",
        "comprehensive", "business management", "features", 
        "key features", "capabilities", "direkt features",
    ],
    "security": [
        "security", "secure", "authentication", "authorization", "encrypted", 
        "encryption", "role", "backup", "permission", "access control", 
        "audit", "role-based", "data security", "protected", "safe",
    ],
    "developer": [
        "developed", "developer", "glovomega", "technologies", "created",
        "built", "who", "ai-powered",
    ],
    "purchase": [
        "purchase order", "purchase orders", "purchasing", "supplier",
        "suppliers", "stock levels", "procurement", "approve", "purchase",
        "vendor",
    ],
}

# Curated query-to-section mappings (merged with auto-detected titles at runtime)
SECTION_TITLE_SYNONYMS: dict[str, str] = {
    "introduction": "Introduction",
    "welcome": "Introduction",
    "about": "About Direkt",
    "direkt": "About Direkt",
    "glovomega": "About Direkt",
    "features": "Key Features",
    "key features": "Key Features",
    "modules": "Key Features",
    "key": "Key Features",
    "feature": "Key Features",
    "features": "Key Features",
    "functions": "Key Features",
    "capabilities": "Key Features",
    "sales": "Sales module",
    "pos": "Omnichannel Point of Sale(POS)",
    "billing": "Sales module",
    "quotation": "Sales module",
    "inventory": "Inventory module",
    "stock": "Inventory Management",
    "warehouse": "Step 2: Create Warehouses",
    "warehouses": "Inventory Management",
    "barcode": "Inventory module",
    "finance": "Finance module",
    "accounting": "Accounting",
    "tax": "Tax module",
    "gst": "GST",
    "cgst": "GST",
    "sgst": "GST",
    "igst": "GST",
    "gstr": "GST",
    "itc": "GST",
    "ai": "AI module",
    "forecast": "AI module",
    "prediction": "AI module",
    "getting": "Getting Started",
    "started": "Getting Started",
    "setup": "Getting Started",
    "onboard": "Getting Started",
    "start": "Getting Started",
    "begin": "Getting Started",
    "new business": "Getting Started",
    "new business owner": "Getting Started",
    "new user": "Getting Started",
    "company": "Step 1: Create Your Company",
    "gstin": "Step 1: Create Your Company",
    "pan": "Step 1: Create Your Company",
    "create": "Getting Started",
    "product": "Step 3: Add Products",
    "products": "Step 3: Add Products",
    "sku": "Step 3: Add Products",
    "catalogue": "Product Catalogue",
    "catalog": "Product Catalogue",
    "digital": "Product Catalogue",
    "order": "Order Processing",
    "orders": "Order Processing",
    "checkout": "Order Processing",
    "omnichannel": "Omnichannel Point of Sale(POS)",
    "create a sale": "Creating a Sale",
    "make a sale": "Creating a Sale",
    "create warehouse": "Step 2: Create Warehouses",
    "create a warehouse": "Step 2: Create Warehouses",
    "add warehouse": "Step 2: Create Warehouses",
    "sale": "Creating a Sale",
    "sales module": "Sales module",
    "sale module": "Sales module",
    "inventory module": "Inventory module",
    "finance module": "Finance module",
    "tax module": "Tax module",
    "ai module": "AI module",
    "ai management": "AI module",
    "finance management": "Finance module",
    "sales management": "Sales module",
    "sale management": "Sales module",
    "pos features": "Omnichannel Point of Sale(POS)",
    "pos feature": "Omnichannel Point of Sale(POS)",
    "introduce direkt": "Introduction",
    "benefits of direkt": "Key Features",
    "benefits": "Key Features",
    "advantages": "Key Features",
    "invoice": "Creating a Sale",
    "payment": "Payments",
    "payments": "Payments",
    "upi": "Payments",
    "purchase": "Purchase Orders",
    "supplier": "Purchase Orders",
    "analytics": "Analytics Dashboard",
    "dashboard": "Analytics Dashboard",
    "revenue": "Analytics Dashboard",
    "profit": "Analytics Dashboard",
    "security": "Security",
    "authentication": "Security",
    "backup": "Security",
    "permission": "Security",
    "faq": "Frequently Asked Questions",
    "questions": "Frequently Asked Questions",
    "developed": "Introduction",
    "developer": "Introduction",
    "developers": "Introduction",
    "created": "Introduction",
    "built": "Introduction",
    "technologies": "Introduction",
    "secure": "Security",
    "safe": "Security",
    "protected": "Security",
    "account": "Accounting",
    "accounts": "Accounting",
    "purchasing": "Purchase Orders",
    "procurement": "Purchase Orders",
    "capabilities": "Key Features",
    "capability": "Key Features",
    "begin": "Getting Started",
    "owner": "Getting Started",
    "add": "Step 3: Add Products",
    "used": "Key Features",
    "tax management": "Tax module",
    "gst management": "GST",
    "gst compliance": "GST",
    "gst invoice": "GST",
    "account process": "Accounting",
    "accounting process": "Accounting",
    "secure": "Security",
    "safe": "Security",
    "security": "Security",
    "purchase": "Purchase Orders",
    "purchasing": "Purchase Orders",
    "add product": "Step 3: Add Products",
    "create product": "Step 3: Add Products",
    "new product": "Step 3: Add Products",
}

# Query-intent phrases that strongly indicate specific Direkt manual sections
QUERY_INTENT_SECTIONS: dict[str, list[str]] = {
    "what is direkt": ["Introduction", "About Direkt"],
    "about direkt": ["About Direkt", "Introduction"],
    "key features": ["Key Features"],
    "sales module": ["Sales module", "Key Features"],
    "inventory module": ["Inventory module", "Key Features"],
    "finance module": ["Finance module", "Key Features"],
    "tax module": ["Tax module", "Key Features"],
    "ai module": ["AI module", "Key Features"],
    "getting started": ["Getting Started", "Frequently Asked Questions"],
    "how to start": ["Getting Started", "Step 1: Create Your Company"],
    "create company": ["Step 1: Create Your Company", "Getting Started"],
    "create your company": ["Step 1: Create Your Company"],
    "add warehouse": ["Step 2: Create Warehouses", "Getting Started"],
    "create warehouse": ["Step 2: Create Warehouses"],
    "add product": ["Step 3: Add Products", "Getting Started"],
    "new product": ["Step 3: Add Products"],
    "product catalogue": ["Product Catalogue"],
    "digital catalogue": ["Product Catalogue"],
    "place order": ["Order Processing", "Frequently Asked Questions"],
    "order processing": ["Order Processing"],
    "point of sale": ["Omnichannel Point of Sale(POS)", "Creating a Sale"],
    "create a sale": ["Creating a Sale"],
    "how to create a sale": ["Creating a Sale"],
    "how to make a sale": ["Creating a Sale"],
    "make a sale": ["Creating a Sale"],
    "create warehouse": ["Step 2: Create Warehouses"],
    "how to create warehouse": ["Step 2: Create Warehouses", "Getting Started"],
    "how to create a warehouse": ["Step 2: Create Warehouses", "Getting Started"],
    "pos billing": ["Creating a Sale", "Omnichannel Point of Sale(POS)"],
    "payment methods": ["Payments"],
    "payment method": ["Payments"],
    "inventory management": ["Inventory Management", "Inventory module"],
    "purchase order": ["Purchase Orders"],
    "purchase orders": ["Purchase Orders"],
    "double entry": ["Accounting", "Finance module"],
    "double-entry": ["Accounting", "Finance module"],
    "gst billing": ["GST", "Tax module"],
    "gst invoice": ["GST", "Sales module"],
    "gstr-1": ["GST"],
    "gstr-3b": ["GST"],
    "analytics dashboard": ["Analytics Dashboard"],
    "business dashboard": ["Analytics Dashboard"],
    "security features": ["Security"],
    "data security": ["Security"],
    "security": ["Security"],
    "authentication": ["Security"],
    "backup": ["Security"],
    "permission": ["Security"],
    "encrypted": ["Security"],
    "frequently asked": ["Frequently Asked Questions"],
    "faq": ["Frequently Asked Questions"],
    "whatsapp order": ["Order Processing", "Product Catalogue"],
    "qr code": ["Product Catalogue", "Payments"],
    "low stock": ["Inventory module", "Purchase Orders"],
    "stock transfer": ["Inventory module"],
    "get started with": ["Getting Started", "Step 1: Create Your Company", "Frequently Asked Questions"],
    "get started": ["Getting Started", "Frequently Asked Questions"],
    "how do i get started": ["Getting Started", "Step 1: Create Your Company"],
    "new business owner": ["Getting Started", "Step 1: Create Your Company"],
    "business owner": ["Getting Started", "Step 1: Create Your Company"],
    "get start with": ["Getting Started", "Step 1: Create Your Company"],
    "get start": ["Getting Started", "Step 1: Create Your Company"],
    "how to get start": ["Getting Started", "Step 1: Create Your Company"],
    "begin with direkt": ["Getting Started", "Introduction"],
    "first time using": ["Getting Started", "Frequently Asked Questions"],
    "who developed": ["Introduction", "About Direkt"],
    "who created": ["Introduction", "About Direkt"],
    "who made": ["Introduction", "About Direkt"],
    "developed by": ["Introduction", "About Direkt"],
    "who builds": ["Introduction", "About Direkt"],
    "direkt features": ["Key Features", "Frequently Asked Questions"],
    "about direkt features": ["Key Features", "About Direkt"],
    "tell me about direkt": ["About Direkt", "Key Features", "Introduction"],
    "tell me about features": ["Key Features", "Frequently Asked Questions"],
    "what are the features": ["Key Features", "Frequently Asked Questions"],
    "what can direkt do": ["Key Features", "About Direkt"],
    "tell me about direkt features": ["Key Features", "About Direkt"],
    "gst management": ["GST", "Tax module"],
    "tax management": ["Tax module", "GST"],
    "manage gst": ["GST", "Tax module"],
    "how direkt manage gst": ["GST", "Tax module"],
    "tell me about gst": ["GST", "Tax module"],
    "tell me about tax": ["Tax module", "GST"],
    "what is the tax module": ["Tax module", "Key Features"],
    "tax module used for": ["Tax module", "Key Features"],
    "what does tax module": ["Tax module", "Key Features"],
    "what is tax module": ["Tax module", "Key Features"],
    "direkt security": ["Security"],
    "about direkt security": ["Security"],
    "tell me about direkt security": ["Security"],
    "tell me about security": ["Security"],
    "is direkt secure": ["Security"],
    "direkt is secure": ["Security"],
    "is it secure": ["Security"],
    "is direkt safe": ["Security"],
    "security features": ["Security"],
    "account process": ["Accounting", "Finance module"],
    "accounting process": ["Accounting", "Finance module"],
    "tell me about accounting": ["Accounting", "Finance module"],
    "how does accounting work": ["Accounting", "Finance module"],
    "order purchasing": ["Purchase Orders", "Order Processing"],
    "direkt order purchasing": ["Purchase Orders", "Order Processing"],
    "tell me about purchase": ["Purchase Orders"],
    "how to order purchased": ["Purchase Orders", "Order Processing"],
    "how to add a product": ["Step 3: Add Products", "Getting Started"],
    "how to add product": ["Step 3: Add Products", "Getting Started"],
    "what is sales module": ["Sales module", "Key Features"],
    "what is inventory module": ["Inventory module", "Key Features"],
    "what is finance module": ["Finance module", "Key Features"],
    "what is ai module": ["AI module", "Key Features"],
    "ai management": ["AI module"],
    "tell me about ai management": ["AI module"],
    "what is ai management": ["AI module"],
    "finance management": ["Finance module"],
    "what is finance management": ["Finance module"],
    "tell me about finance management": ["Finance module"],
    "sale management": ["Sales module"],
    "sales management": ["Sales module"],
    "what is sale management": ["Sales module"],
    "what is sale module": ["Sales module"],
    "sale module": ["Sales module"],
    "pos features": ["Omnichannel Point of Sale(POS)", "Sales module"],
    "pos feature": ["Omnichannel Point of Sale(POS)", "Sales module"],
    "introduce direkt": ["Introduction", "About Direkt"],
    "benefits of direkt": ["Key Features", "About Direkt", "Introduction"],
    "how to purchase orders": ["Purchase Orders"],
    "how to purchase order": ["Purchase Orders"],
    "how to create purchase order": ["Purchase Orders"],
    "how to create purchase orders": ["Purchase Orders"],
    "sales module used for": ["Sales module", "Key Features"],
    "inventory module used for": ["Inventory module", "Key Features"],
    "finance module used for": ["Finance module", "Key Features"],
    "ai module used for": ["AI module", "Key Features"],
    "how to create a warehouse": ["Step 2: Create Warehouses", "Getting Started"],
    "create a warehouse in direkt": ["Step 2: Create Warehouses", "Getting Started"],
    "create warehouse": ["Step 2: Create Warehouses"],
}

