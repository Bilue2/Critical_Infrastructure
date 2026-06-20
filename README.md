# 🔐 Energy Sector Cybersecurity RAG System (NERC / NIST / CISA)

A Retrieval-Augmented Generation (RAG) system for **critical infrastructure cybersecurity intelligence**, built for analyzing:

- NERC CIP standards
- NIST Cybersecurity Framework / SP 800-53
- CISA guidance
- OT / ICS security documentation

It enables grounded AI answers with **traceable citations from regulatory documents**.

---

# 🚀 What This System Does

Ask questions like:

- What are NERC CIP access control requirements?
- How should incident response be handled in OT environments?
- What risks exist in energy infrastructure systems?
- How does NIST CSF map to this scenario?
- What mitigations are recommended by CISA?

The system:
- retrieves relevant regulatory chunks
- applies control-family detection
- generates grounded answers with citations
- avoids hallucinations

---

# 🧠 Core Architecture

```text
User Query
   ↓
Embedding (OpenAI)
   ↓
Pinecone Vector Search
   ↓
Control-family filtering (NERC / NIST / CISA)
   ↓
Context assembly
   ↓
LLM grounded generation (GPT-4.1-mini)
   ↓
Structured cybersecurity response

# Project Structure
├── README.md
├── app
│   ├── control_families.py   # Detects NERC/NIST control categories
│   └── rag.py                # Core retrieval + generation engine
│
├── config.py
│
├── data
│   ├── critical_infra_corpus.jsonl
│   └── pdf/
│
├── scripts
│   ├── 02_prepare_energy_json.py
│   ├── 03_index_pinecone.py
│   ├── 04_query_bot.py
│   └── 05_eval_grounding.py
│
├── streamlit_app.py
└── requirements.txt

# Installation
##1.Create Environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

##2. Install dependencies
pip install -r requirements.txt

3. Setup .env
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key

PINECONE_INDEX=energy-cyber-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_MODEL=gpt-4.1-mini

#How to Run Each File
🟡 STEP 1 — Prepare Energy Corpus
Script:

scripts/02_prepare_energy_json.py

Run:
python -m scripts.02_prepare_energy_json \
  --input data/pdf \
  --output data/critical_infra_corpus.jsonl
What it does:
Extracts text from PDFs
Converts NERC/NIST/CISA documents into structured chunks
Produces JSONL dataset
🟠 STEP 2 — Index into Pinecone
Script:

scripts/03_index_pinecone.py

Run:
python -m scripts.03_index_pinecone \
  --chunks data/critical_infra_corpus.jsonl \
  --reset
Optional namespace:
python -m scripts.03_index_pinecone \
  --chunks data/critical_infra_corpus.jsonl \
  --namespace nerc-cip
What it does:
Generates embeddings
Detects control families (Access Control, Incident Response, etc.)
Stores vectors in Pinecone
Adds metadata for filtering and retrieval
🔵 STEP 3 — Query Bot (CLI)
Script:

scripts/04_query_bot.py

Run basic query:
python -m scripts.04_query_bot \
  --q "What are access control requirements under NERC CIP?"
Run with debugging:
python -m scripts.04_query_bot \
  --q "How should incident response be handled in OT systems?" \
  --debug
What it does:
Runs retrieval pipeline
Calls LLM
Returns structured cybersecurity answer
🟣 STEP 4 — Streamlit App (UI)
Run:
streamlit run streamlit_app.py
Features:
Chat interface
Grounded cybersecurity answers
Retrieved chunk evidence display
Policy recommendation breakdown
Debug mode toggle
🔴 STEP 5 — Evaluation
Script:

scripts/05_eval_grounding.py

Run:
python -m scripts.05_eval_grounding
What it checks:
Grounding quality
JSON schema validation
Chunk citation accuracy
Retrieval consistency
🧠 Key Concepts
🔐 1. Retrieval-Augmented Generation (RAG)
Query → Retrieve documents → LLM answers using only evidence

Prevents hallucination and ensures regulatory accuracy.

🧩 2. Control Family Detection

Automatically detects cybersecurity domains:

Access Control
Incident Response
Risk Management
Monitoring & Logging
Patch Management
Network Security
📊 3. Pinecone Vector Search

Stores:

embeddings of regulatory text
metadata (org, sector, chunk_id)
control tags

Enables semantic search across compliance documents.

🏭 4. Energy Sector Focus

Designed for:

Power grids
Utility operators
Industrial control systems (ICS)
OT environments
⚠️ Common Issues
❌ "Missing API Key"

Fix .env file:

OPENAI_API_KEY=...
PINECONE_API_KEY=...
❌ "No module named app"

Always run with:

python -m scripts.03_index_pinecone

NOT:

python scripts/03_index_pinecone.py
❌ Empty search results

Try:

increasing top_k
running --reset
checking embeddings consistency
🚀 Recommended Workflow

Run in order:

# 1. Build corpus
python -m scripts.02_prepare_energy_json --input data/pdf --output data/critical_infra_corpus.jsonl

# 2. Index data
python -m scripts.03_index_pinecone --chunks data/critical_infra_corpus.jsonl --reset

# 3. Query system
python -m scripts.04_query_bot --q "What are NERC CIP access control requirements?"

# 4. UI
streamlit run streamlit_app.py
🔥 Future Improvements
Phase 2 (Advanced RAG)
Hybrid search (BM25 + vector)
Reranking model
Chunk compression
Citation verification layer
Phase 3 (Enterprise)
CIP compliance scoring engine
Asset risk modeling
Audit report generator (PDF)
SOC-style incident response assistant
📌 Summary

This system is a:

🔐 Grounded AI assistant for energy sector cybersecurity compliance

It ensures:

no hallucinations
traceable regulatory evidence
OT/ICS-aware reasoning
structured compliance outputs