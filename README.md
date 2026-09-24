# Rule-Based Customer Support Chatbot

A Python-based Rule-Based Customer Support Chatbot designed to answer customer queries using document processing, keyword search, regex search, and semantic search techniques.

## 📌 Project Overview

This project is a customer support chatbot that processes information from different document formats and retrieves relevant information based on user queries.

The system contains modules for:

- Document reading
- Text preprocessing
- Text chunking
- Keyword search
- Regex-based search
- Semantic search
- Query mapping
- Information retrieval
- Database management
- Logging
- Query testing

## 🚀 Features

- Rule-based customer support
- Multiple document format support
- PDF document reading
- DOCX document reading
- Excel file reading
- CSV file reading
- JSON file reading
- TXT file reading
- Text preprocessing
- Text chunking
- Keyword-based search
- Regex-based search
- Semantic search
- Query mapping
- ChromaDB integration
- Query testing and debugging
- Logging system

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- ChromaDB
- Sentence Transformers
- Regular Expressions
- Jupyter Notebook
- VS Code

## 📂 Project Structure

```text
Rule-Based-Customer-Support-Chatbot/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── documents/
│   └── Direkt_Manual_Report.txt
│
├── src/
│   ├── chatbot/
│   │   └── chatbot.py
│   │
│   ├── database/
│   │   └── chroma_manager.py
│   │
│   ├── embeddings/
│   │   └── embedding_model.py
│   │
│   ├── preprocessing/
│   │   ├── chunker.py
│   │   ├── cleaner.py
│   │   ├── heading_patterns.py
│   │   ├── paragraph_extractor.py
│   │   └── section_registry.py
│   │
│   ├── readers/
│   │   ├── csv_reader.py
│   │   ├── docx_reader.py
│   │   ├── excel_reader.py
│   │   ├── json_reader.py
│   │   ├── pdf_reader.py
│   │   └── txt_reader.py
│   │
│   ├── search/
│   │   ├── keyword_search.py
│   │   ├── query_mapper.py
│   │   ├── regex_search.py
│   │   ├── retriever.py
│   │   └── semantic_search.py
│   │
│   └── utils/
│       ├── config.py
│       ├── file_manager.py
│       └── logger.py
│
└── tests/
    ├── debug_query.py
    └── test_queries.py