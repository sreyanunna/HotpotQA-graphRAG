# Bridging the Gap Between Natural Language and Knowledge Graph Query
### A Hybrid Querying Pipeline

> **MSc Independent Research Project** · Imperial College London  
> Department of Earth Science and Engineering — Environmental Data Science & Machine Learning  
> Author: **Sreya Nunna** · Supervisors: Dr. Christopher Pain & Dr. Olga Buskin

---

## 📌 Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Pipeline Architecture](#pipeline-architecture)
- [Pipeline Evolution](#pipeline-evolution)
- [HippoRAG: Neurobiological Inspiration](#hipporag-neurobiological-inspiration)
- [Dataset Description](#datasets)
- [Evaluation & Results](#evaluation--results)
- [Limitations & Future Work](#limitations--future-work)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [References](#references)

---

## Overview

Knowledge Graphs (KGs) are rapidly becoming a critical infrastructure for modern data management — projected to reach **$6.93 billion USD by 2030** — yet their adoption is hindered by a fundamental accessibility barrier: effective use typically requires technical expertise in graph query languages like SPARQL and Cypher.

This project presents an **end-to-end hybrid querying pipeline** that enables seamless, natural language interaction with knowledge graph databases — without requiring any graph query expertise. The system:

- Ingests heterogeneous, unstructured text documents
- Automatically constructs a structured Knowledge Graph via Subject-Predicate-Object (SPO) triple extraction
- Interprets natural language queries through a hybrid approach combining **semantic entity matching** and **instruction-tuned LLMs**
- Retrieves targeted subgraphs and synthesises contextual answers in plain English

The intended audience spans from non-technical enterprise employees querying meeting notes and reports, to developers conducting cross-document knowledge discovery.

---

## Motivation

Current tools for translating natural language to graph queries are insufficient. **NeoConverse**, Neo4j's state-of-the-art solution, achieves only 45% execution accuracy on complex queries due to semantic misinterpretation of user intent.

Additionally, existing knowledge retrieval approaches suffer from three key gaps:

| Problem | Traditional Approach | This Pipeline |
|---|---|---|
| Static knowledge base | Fixed ontologies | Dynamic KG construction from any document |
| Single-document reasoning | Chunk-based RAG | Cross-document multi-hop reasoning |
| Black-box retrieval | Opaque LLM responses | Interpretable subgraph extraction |
| Technical barrier | Requires SPARQL/Cypher | Natural language interface |

This pipeline directly addresses each of these gaps.

---

## Pipeline Architecture

The final system is a **six-stage orchestrated pipeline**:

```
📄 TEXT DOCUMENTS (heterogeneous input)
        │
        ▼
┌─────────────────────────────┐
│  1. QUERY UNDERSTANDING     │  Natural language → SPO graph query format
│     (Instruction-tuned LLM) │  via OpenAI API
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  2. SEMANTIC ENTITY         │  Linguistic constructs → KG entity nodes
│     RETRIEVAL               │  via all-MiniLM-L6-v2 cosine similarity
│     (SentenceTransformer)   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  3. SUBGRAPH CREATION       │  Bidirectional graph exploration around
│     (Bidirectional BFS)     │  matched entities (depth-1 & depth-2)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  4. SPARQL QUERY            │  Hybrid: local predicate identification
│     GENERATION              │  + distributed semantic search
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  5. QUERY EXECUTION         │  Formal query against RDF graph store
│     (with error fallback)   │  with robust fallback mechanisms
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  6. ANSWER GENERATION       │  Structured results → natural language
│     (LLM synthesis)         │  contextual response
└─────────────────────────────┘
        │
        ▼
💬 NATURAL LANGUAGE ANSWER
```
## Pipeline Evolution

### Stage 1: Traditional RAG (Baseline)

Initial experiments used **LangChain's `RecursiveCharacterTextSplitter`** on Shakespeare's plays (chosen for similar language patterns but different content — enabling complex multi-hop test cases).

- Chunk size: 500 characters with 50-character overlaps
- Embedding: `BAAI/bge-small-en` (HuggingFace SentenceTransformer)
- Vector storage: FAISS indexing
- Limitation: chunk-based retrieval cannot link information across documents

### Stage 2: LLM-Based Triple Extraction

Extensive model comparison for SPO triple generation:

| Model | Parameters | Issue |
|---|---|---|
| Microsoft Phi2 | 2B | High hallucination, repetitive triples — unsuitable for production |
| Qwen2.5-7B-Instruct | 7B | Adequate quality but **56s/chunk** → 21+ hours for full dataset |
| **OpenAI API** | — | Fastest + highest quality, clean SPO with minimal hallucination |

Triples embedded with `all-MiniLM-L6-v2`, achieving **0.84+ cosine similarity** on complex queries like *"Who did Hamlet kill?"*

### Stage 3: HippoRAG-Inspired Final Pipeline

This pipeline draws its core architectural inspiration from [**HippoRAG**](https://arxiv.org/abs/2405.14831) (Gutiérrez et al., 2024) — a neurobiologically-inspired retrieval framework that revolutionises traditional RAG by mimicking the **human hippocampal memory indexing mechanism**.

### HippoRAG's Three Components

| Biological Component | Technical Implementation | Role |
|---|---|---|
| Neocortex | Instruction-tuned LLM | Named entity extraction & reasoning |
| Parahippocampal regions | SentenceTransformer encoders | Semantic similarity & entity matching |
| Hippocampal index | Knowledge Graph | Associative memory storage |

HippoRAG's **Personalised PageRank (PPR)** algorithm activates relevant entity neighbourhoods, enabling single-step multi-hop reasoning — achieving **10–20× cost reduction** and **6–13× speed improvements** over iterative methods like IRCoT, with up to **20% performance gains** on MuSiQue and 2WikiMultiHopQA benchmarks.


## Dataset Description

### HotpotQA — Evaluation Benchmark

This pipeline is evaluated against **[HotpotQA](https://hotpotqa.github.io/)** (Yang, Qi, Zhang et al., EMNLP 2018) — a question answering dataset purpose-built for **diverse, explainable multi-hop question answering**.

> *"HotpotQA features natural, multi-hop questions with strong supervision for supporting facts, to enable more explainable question answering systems."*  
> — Carnegie Mellon University, Stanford University & Université de Montréal

**Why HotpotQA is the right benchmark:**
- Questions require reasoning across **multiple documents** simultaneously — exactly the challenge this pipeline addresses
- Includes **supporting fact supervision** — enabling transparent, explainable evaluation of which knowledge graph nodes contributed to each answer
- Two evaluation settings: **distractor** (10 paragraphs provided, 2 relevant) and **fullwiki** (entire Wikipedia as context)

**Dataset downloads** (CC BY-SA 4.0):

| Split | Size | Link |
|---|---|---|
| Training set | 535 MB | [hotpot_train_v1.1.json](http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json) |
| Dev set (distractor) | 44 MB | [hotpot_dev_distractor_v1.json](http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json) |
| Dev set (fullwiki) | 45 MB | [hotpot_dev_fullwiki_v1.json](http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json) |
| Test set (fullwiki) | 46 MB | [hotpot_test_fullwiki_v1.json](http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json) |


---

## Evaluation & Results

### Hardware Configuration

| Component | Specification |
|---|---|
| Platform | Google Colab Pro |
| LLM Training Runtime | Tesla T4 GPU (16GB VRAM) |
| Pipeline Execution | High RAM CPU (Intel Xeon 2.3GHz) |
| Memory (RAM) | 25GB System RAM |

### End-to-End Performance

| Metric | Value |
|---|---|
| **Execution Success Rate** | **91.7%** |
| Average Entity Match Accuracy | 0.83 |
| Average F1 Score | 0.414 |
| Average Latency | 26.33s |
| Perfect Entity Matches (1.0) | 8/12 (66.7%) |
| Entity Match Accuracy ≥ 0.8 | 10/12 (83.3%) |

### Key Finding

> **High entity match accuracy (0.83) does not guarantee high F1 scores (0.414).**  
> This reveals that the bottleneck lies downstream — in relationship extraction and answer generation — rather than in entity retrieval itself. This finding directly motivates future work on multi-hop reasoning chains.

## Limitations & Future Work

### Current Limitations

**Multi-hop reasoning** is the primary gap. The pipeline executes accurate single-hop traversal (85% success rate) but cannot orchestrate complex relational chains. Example failure:

```
Query: "Which young players were under Alex Ferguson?"

What the pipeline CAN do:
  Alex_Ferguson ──manages──▶ Manchester_United

What it CANNOT yet do:
  Alex_Ferguson ──manages──▶ Manchester_United ──has_player──▶ [Beckham, Scholes, Butt...]
```

Additional limitations:
- **Contextual persistence**: complex queries lose contextual information during the query deciphering step, causing inadequate graph traversal
- **Multi-entity queries**: subgraph extraction currently handles only one anchor entity per query
- **Ambiguous queries**: open-ended questions like *"who is Barack Obama?"* require comprehensive entity summarisation — currently unsupported


---

## Tech Stack

| Component | Technology |
|---|---|
| Triple Extraction LLM | OpenAI API (GPT-4) |
| Semantic Matching | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector Indexing | FAISS |
| Knowledge Graph | RDFLib (SPARQL-queryable RDF graph) |
| Graph Query | SPARQL |
| API Interface | FastAPI |
| Initial RAG Baseline | LangChain |
| Development Platform | Google Colab Pro |
| Experiment Tracking | CSV export + manual evaluation |

---

## Repository Structure

```
📦 knowledge-graph-hybrid-pipeline/
├── pipeline-code/
├── 📁 api/
│   └── main.py                        # FastAPI interface
├── 📁 data/
│   ├── triples/                       # Extracted SPO CSVs          [coming soon]
│   ├── embeddings/                    # Serialised embeddings        [coming soon]
│   └── graphs/                        # RDF graph serialisations     [coming soon]
├── 📁 evaluation/
│   ├── hotpotqa_eval_subset.csv       # Evaluation queries           [coming soon]
│   └── results_analysis.ipynb        # Performance analysis notebook
├── requirements.txt
└── README.md
```
---

## Getting Started

### Prerequisites

```bash
python >= 3.9
```

### Installation

```bash
git clone https://github.com/sreyanunna/HotpotQA-graphRAG.git
cd knowledge-graph-hybrid-pipeline
pip install -r requirements.txt
```

### Environment Variables

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Run the Pipeline

```bash
# Place your documents in data/raw/
# Then run end-to-end:
python pipeline/02_triple_extraction.py
python pipeline/03_kg_construction.py

# Launch the FastAPI interface:
uvicorn api.main:app --reload
# → Visit http://localhost:8000/docs for interactive API
```

### Run Evaluation (HotpotQA)

```bash
# Download HotpotQA dev set first (see Datasets section)
python evaluation/run_eval.py --dataset data/hotpotqa/hotpot_dev_distractor_v1.json
```

---

## References

1. Sequeda et al. (2023). *Knowledge Graphs as a Source of Trust for LLM-powered Enterprise Question Answering.* ScienceDirect.
2. Rodrigues et al. (2023). *Performance Comparison of Graph Database and Relational Database.* ResearchGate.
3. Cai, L. et al. (2025). *Practices, opportunities and challenges in the fusion of knowledge graphs and large language models.* Frontiers in Computer Science, 7, 1590632.
4. Gutiérrez, B.J. et al. (2024). *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models.* arXiv:2405.14831.
5. Yang, Z., Qi, P., Zhang, S. et al. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.* EMNLP 2018. arXiv:1809.09600.
6. Yin, X., Gromann, D. & Rudolph, S. (2019). *Neural Machine Translating from Natural Language to SPARQL.* arXiv:1906.09302.
7. Vargas, H. et al. (2020). *A User Interface for Exploring and Querying Knowledge Graphs.* IJCAI-PRICAI 2020.
8. Knowledge Graph Research Report 2025: Global Market to Reach $6.93 Billion by 2030. Yahoo Finance.

---

## Citation

If you use this pipeline or build upon this work, please cite:

```bibtex
@mastersthesis{nunna2025kghybrid,
  author    = {Sreya Nunna},
  title     = {Bridging the Gap Between Natural Language and Knowledge Graph Query: A Hybrid Querying Pipeline},
  school    = {Imperial College London, Department of Earth Science and Engineering},
  year      = {2025},
  month     = {August},
  note      = {MSc in Environmental Data Science and Machine Learning}
}
```

---

## Acknowledgements

This project was supervised by **Dr. Christopher Pain** and **Dr. Olga Buskin** at Imperial College London.

AI tools used during development: ChatGPT (brainstorming & literature review), Claude Sonnet 4 (code structure & LaTeX), Gemini 1.5 Pro (Google Colab integration), Perplexity Pro (research paper discovery & feedback).

---
