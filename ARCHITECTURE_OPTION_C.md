# Architecture Option C: Modern Agentic RAG for Insurance Q&A

## Executive Summary

This architecture combines **Agentic RAG with Claude's native tool use**, **lightweight
GraphRAG for entity relationships**, **ColBERT late-interaction retrieval**, and
**structured table extraction** into a cohesive system purpose-built for insurance
document Q&A. It avoids heavy frameworks (no LangChain, no LangGraph) in favor of
a thin orchestration layer that lets Claude 4 Sonnet do the reasoning.

The guiding principle: **use the right retrieval strategy per query type**, orchestrated
by an LLM agent that decides dynamically.

---

## Architecture Diagram

```
                         +---------------------------+
                         |      Streamlit / FastAPI   |
                         |        Frontend UI         |
                         +------------+--------------+
                                      |
                                      v
                         +---------------------------+
                         |    AGENT ORCHESTRATOR      |
                         |  (Claude 4 Sonnet +        |
                         |   Native Tool Use)         |
                         |                           |
                         |  Tools available:          |
                         |  - vector_search()         |
                         |  - graph_lookup()          |
                         |  - table_query()           |
                         |  - full_doc_context()      |
                         |  - clarify_question()      |
                         +--+------+------+------+---+
                            |      |      |      |
               +------------+  +---+--+ +-+----+ +--------+
               v               v      v v      v          v
    +----------+--+  +--------+--+ +--+-----+ +-+--------+-+
    | ColBERT      |  | Neo4j Lite | | DuckDB | | Long-Context|
    | (RAGatouille) |  | Graph DB   | | Tables | | Fallback    |
    | Vector Index  |  | (Entities  | | (Copay,| | (Stuff full |
    |               |  |  & Rels)   | | Tiers, | |  doc into   |
    | - Chunked docs|  |            | | Deduct)| |  200K ctx)  |
    | - Late-inter- |  | drugs ->   | |        | |             |
    |   action      |  | tiers ->   | | SQL    | | For small   |
    | - BM25 hybrid |  | plans ->   | | queries| | doc sets    |
    +-------+-------+  | providers  | +---+----+ +------+------+
            |          +-----+------+     |              |
            |                |            |              |
            v                v            v              v
    +--------------------------------------------------------+
    |              DOCUMENT STORE (Local filesystem)          |
    |                                                        |
    |  /raw/         - Original PDFs                         |
    |  /parsed/      - Extracted text + tables (JSON)        |
    |  /chunks/      - Chunked documents for ColBERT         |
    |  /structured/  - Extracted tables as Parquet/JSON      |
    +--------------------------------------------------------+
            ^
            |
    +-------+-------+
    |   INGESTION    |
    |   PIPELINE     |
    |                |
    | 1. PDF parse   |
    |    (PyMuPDF +  |
    |    Docling)    |
    | 2. Table       |
    |    extraction  |
    |    (Docling /  |
    |    Claude)     |
    | 3. Entity      |
    |    extraction  |
    |    (Claude ->  |
    |    Neo4j)      |
    | 4. Chunk +     |
    |    index       |
    |    (ColBERT)   |
    +----------------+
```

---

## Research Findings & Design Decisions

### 1. Agentic RAG: Claude Native Tool Use (not CrewAI/LangGraph)

**Decision: Use Claude's native tool-use API directly, no framework.**

**Why:**
- CrewAI (1.38M monthly downloads) and LangGraph (6.17M) are mature, but both add
  abstraction layers that are unnecessary for a single-agent system.
- For a solo-developer portfolio project, framework lock-in is a liability. Claude's
  tool-use API is stable, well-documented, and gives you full control.
- The agent pattern here is simple: Claude receives a user question, decides which
  tool(s) to call (vector search, graph lookup, table query, or full-context stuff),
  gets results, and synthesizes an answer. If results are insufficient, it re-queries
  with a refined search. This is a single ReAct loop -- no multi-agent coordination needed.

**How it works:**
```python
tools = [
    {
        "name": "vector_search",
        "description": "Search insurance documents using semantic similarity. Best for general coverage questions.",
        "input_schema": { "query": "string", "top_k": "int", "doc_filter": "string (optional)" }
    },
    {
        "name": "graph_lookup",
        "description": "Look up entity relationships: drug -> tier -> copay, procedure -> coverage, provider -> network.",
        "input_schema": { "entity": "string", "relationship_type": "string" }
    },
    {
        "name": "table_query",
        "description": "Query structured tables for copays, deductibles, tier info. Uses SQL on extracted tables.",
        "input_schema": { "sql": "string" }
    },
    {
        "name": "full_doc_context",
        "description": "Load an entire document into context. Use for small docs or when other tools return insufficient results.",
        "input_schema": { "doc_id": "string" }
    }
]
```

The agent decides: "User is asking about Lipitor copay" -> calls `graph_lookup(entity="Lipitor", relationship_type="drug_tier_copay")` first, then if needed `table_query("SELECT copay FROM formulary WHERE drug_name LIKE '%Lipitor%'")`.

**Tradeoff vs. LangGraph/CrewAI:** You lose built-in memory management, streaming
state, and visual debugging. But for a portfolio project with a single agent, these
are not critical. You gain simplicity and zero framework debt.

---

### 2. Graph RAG: Lightweight Neo4j for Entity Relationships

**Decision: Use a lightweight knowledge graph (Neo4j Community or Memgraph) for
entity relationships, but NOT full Microsoft GraphRAG.**

**Why GraphRAG matters for insurance:**
- Insurance documents are inherently relational: Drug -> Formulary Tier -> Copay Amount,
  Procedure -> CPT Code -> Coverage Status -> Prior Auth Required, Provider -> Network
  -> Plan -> Deductible.
- The word "limit" means different things in Personal Auto Liability vs. Commercial
  General Liability. A knowledge graph disambiguates by encoding structural context.
- Standard vector search retrieves *similar text*. Graph search retrieves *connected facts*.
  For "What's my copay for Lipitor on the Gold plan?", you need to traverse:
  Lipitor -> Tier 2 -> Gold Plan Tier 2 Copay -> $30.

**Why NOT full Microsoft GraphRAG:**
- Microsoft's GraphRAG uses LLM extraction to build community summaries and is designed
  for exploratory questions over large corpora. It costs 3-5x more than baseline RAG.
- For insurance docs, the entity types and relationships are well-defined and predictable.
  A targeted extraction approach is more efficient than the generic community-detection
  approach.

**Implementation approach:**
```
Entities:        Drug, Procedure, Provider, Plan, Tier, BenefitCategory
Relationships:   BELONGS_TO_TIER, COVERED_BY, REQUIRES_PRIOR_AUTH,
                 HAS_COPAY, IN_NETWORK, HAS_DEDUCTIBLE
```

Extraction is done at ingestion time using Claude with structured output:
```python
extraction_prompt = """
Extract entities and relationships from this insurance document section.
Return JSON with:
- entities: [{type, name, attributes}]
- relationships: [{from, to, type, attributes}]

Entity types: Drug, Procedure, Provider, Plan, Tier, BenefitCategory
Relationship types: BELONGS_TO_TIER, COVERED_BY, REQUIRES_PRIOR_AUTH, ...
"""
```

**Tradeoff:** Graph construction adds ingestion complexity and cost (LLM calls per
chunk). But for insurance, the entity model is narrow enough that extraction is
reliable, and the query-time benefit is significant for relational questions.

---

### 3. ColBERT Late-Interaction Retrieval (via RAGatouille)

**Decision: Use ColBERTv2 via RAGatouille as the primary retrieval engine, with
BM25 hybrid scoring.**

**Why ColBERT over standard bi-encoder embeddings (OpenAI, Cohere, etc.):**
- ColBERT computes token-level interactions between query and document, approximating
  cross-encoder accuracy with bi-encoder efficiency.
- Insurance jargon is dense and context-dependent. "Deductible" means something
  different in "annual deductible" vs. "per-incident deductible" vs. "embedded
  deductible." ColBERT's token-level matching captures these nuances better.
- ColBERT is strong at zero-shot retrieval in new domains without fine-tuning.
- RAGatouille makes ColBERT simple: `pip install ragatouille`, index documents,
  query. It persists indices on disk and is production-viable for moderate scale.

**Why NOT standard embeddings:**
- Standard bi-encoders compress an entire passage into a single vector, losing
  token-level detail. For insurance documents where specific terms matter enormously,
  this is a meaningful loss.
- ColBERT's multi-vector approach does use more storage (one vector per token), but
  for a personal project with hundreds (not millions) of documents, this is fine.

**Hybrid approach:**
```python
from ragatouille import RAGPretrainedModel

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# At ingestion
RAG.index(
    collection=chunks,
    index_name="insurance_docs",
    split_documents=False  # we pre-chunk ourselves
)

# At query time -- ColBERT handles late interaction scoring
results = RAG.search(query="What is my copay for Tier 2 drugs?", k=10)
```

BM25 is added as a parallel retrieval path (using rank_bm25 or tantivy-py) and
results are fused using Reciprocal Rank Fusion (RRF).

**Tradeoff vs. OpenAI embeddings:** ColBERT runs locally (no API costs for retrieval),
but requires ~2GB RAM for the model. Index size is larger. For a portfolio project,
the local-first approach is actually a benefit (no API dependency for retrieval).

---

### 4. Long-Context LLMs vs. RAG: Hybrid Approach

**Decision: RAG as primary, long-context stuffing as a fallback tool.**

**Key research findings:**
- Stanford's "Lost in the Middle" research shows LLMs fail to use information
  positioned in the middle of long contexts (30%+ performance degradation).
- Gemini 3.0 Pro maintains only 77% accuracy at full 1M token load.
- RAG inference costs are much lower since only relevant chunks are sent.
- However, for small document sets (<50 pages), context stuffing is simpler and
  can outperform RAG (no retrieval errors possible).

**Our hybrid approach:**
- The agent has a `full_doc_context()` tool that loads an entire document into
  Claude's 200K context window.
- The agent uses this tool when: (a) the user asks about a specific document that
  fits in context, (b) vector search returns low-confidence results and the relevant
  doc is identifiable, or (c) the question requires holistic understanding of a
  document (e.g., "Summarize my plan's coverage").
- For most queries, ColBERT retrieval + focused context is preferred (faster, cheaper,
  more accurate for needle-in-haystack).

**Tradeoff:** You pay more tokens when stuffing full documents, but the agent only
does this when it judges retrieval results as insufficient. This is the "best of
both worlds" approach.

---

### 5. Structured Table Extraction

**Decision: Extract tables into DuckDB (SQL-queryable Parquet files) using Docling
+ Claude structured output.**

**Why this matters for insurance:**
- Insurance documents are TABLE-HEAVY: formulary tier tables, copay schedules,
  deductible matrices, coinsurance percentages by service type, out-of-pocket
  maximum tables.
- Naive chunking of tables destroys their structure. A row like
  "| Tier 2 | $30 | $60 | $90 |" is meaningless without the column headers
  "| Tier | Individual | Family | Out-of-Network |".
- Vector search over flattened table text performs poorly for numerical queries
  like "What's the cheapest tier for my drug?".

**Pipeline:**
1. **Docling** (IBM's open-source doc parser) detects and extracts tables from PDFs
   with structure preserved, including headers, merged cells, and multi-line cells.
2. **Claude structured output** normalizes extracted tables into consistent schemas:
   ```json
   {
     "table_type": "formulary_copay",
     "plan": "Gold PPO",
     "rows": [
       {"tier": 1, "generic": true, "copay_individual": 10, "copay_family": 20},
       {"tier": 2, "generic": false, "copay_individual": 30, "copay_family": 60}
     ]
   }
   ```
3. **DuckDB** stores these as Parquet files, queryable via SQL. The agent's
   `table_query()` tool runs SQL directly:
   ```sql
   SELECT copay_individual FROM formulary_copay
   WHERE plan = 'Gold PPO' AND tier = 2
   ```

**Why DuckDB over SQLite or Postgres:**
- DuckDB is embedded (no server), columnar (fast for analytics queries), reads
  Parquet natively, and is pip-installable. Perfect for a single-developer project.
- It handles the kind of queries insurance users ask: aggregations, comparisons,
  filtering by plan/tier/drug.

**Tradeoff:** Table extraction is imperfect. Complex merged cells, multi-page tables,
and inconsistent formatting in insurance PDFs will require manual review for some
documents. Docling + Claude gets you ~85-90% accuracy on well-formatted tables.

---

### 6. Evaluation Framework

**Decision: DeepEval as primary, with a custom insurance-specific test suite.**

**Why DeepEval over RAGAS:**
- DeepEval is pytest-compatible (fits naturally into a Python developer's workflow).
- DeepEval metrics are self-explaining (tells you *why* a score is low), while
  RAGAS metrics are not, making debugging harder.
- DeepEval can generate synthetic test datasets, useful for bootstrapping evaluation
  before you have real user queries.

**Evaluation metrics for insurance Q&A:**
```python
from deepeval.metrics import (
    AnswerRelevancyMetric,    # Is the answer relevant to the question?
    FaithfulnessMetric,       # Is the answer grounded in retrieved context?
    ContextualPrecisionMetric, # Are the retrieved chunks actually relevant?
    ContextualRecallMetric,    # Did we retrieve all necessary information?
    HallucinationMetric,       # Did the LLM make up coverage details?
)

# Custom insurance-specific metrics
class NumericalAccuracyMetric:
    """Does the stated copay/deductible/coinsurance match the source document?"""

class CitationAccuracyMetric:
    """Does the response cite the correct document section?"""

class ActionabilityMetric:
    """Does the response give the user a clear next step to reduce costs?"""
```

**Golden test dataset (build manually, ~50-100 Q&A pairs):**
```python
test_cases = [
    {
        "question": "What is my copay for Lipitor on the Silver plan?",
        "expected_answer": "$45 for a 30-day supply at a preferred pharmacy",
        "expected_source": "2024_Formulary.pdf, page 47",
        "category": "drug_copay"
    },
    {
        "question": "How can I get my MRI claim approved?",
        "expected_answer": "Your plan requires prior authorization for MRI...",
        "expected_source": "Plan_Document.pdf, section 4.2",
        "category": "claim_strategy"
    },
    # ...
]
```

---

## Concrete Technology Stack

| Component              | Technology                        | Version   | Why                                    |
|------------------------|-----------------------------------|-----------|----------------------------------------|
| LLM (Agent + Gen)     | Claude 4 Sonnet                   | Latest    | Best tool-use, 200K ctx, cost-effective|
| Retrieval              | ColBERTv2 via RAGatouille         | 0.0.9+    | Late-interaction, zero-shot strong     |
| Keyword Search         | rank_bm25                         | 0.2.2+    | Lightweight BM25 for hybrid retrieval  |
| Knowledge Graph        | Neo4j Community Edition           | 5.x       | Free, mature, Cypher query language    |
| Structured Tables      | DuckDB                            | 1.1+      | Embedded columnar SQL, reads Parquet   |
| PDF Parsing            | Docling (IBM)                     | 2.x       | Best OSS table extraction from PDFs    |
| PDF Parsing (backup)   | PyMuPDF (fitz)                    | 1.24+     | Fast text extraction, page-level       |
| Agent Orchestration    | Anthropic Python SDK              | 0.40+     | Direct tool-use, no framework overhead |
| Web Framework          | FastAPI                           | 0.115+    | Async, typed, lightweight              |
| Frontend               | Streamlit                         | 1.40+     | Rapid prototyping for portfolio        |
| Evaluation             | DeepEval                          | 1.x       | Pytest-compatible, self-explaining     |
| Data Storage           | Local filesystem + Parquet        | -         | No infra overhead                      |
| Graph Visualization    | (Optional) Neo4j Browser / yfiles | -         | Debug entity extraction                |

**Python version:** 3.11+

---

## Project Structure

```
InsuranceRAGChatBot/
|-- app/
|   |-- main.py                  # FastAPI app
|   |-- agent.py                 # Claude tool-use agent loop
|   |-- tools/
|   |   |-- vector_search.py     # ColBERT retrieval via RAGatouille
|   |   |-- graph_lookup.py      # Neo4j Cypher queries
|   |   |-- table_query.py       # DuckDB SQL queries
|   |   |-- full_doc_context.py  # Long-context document stuffing
|   |-- models.py                # Pydantic schemas
|   |-- config.py                # Settings
|-- ingestion/
|   |-- pipeline.py              # Orchestrates all ingestion steps
|   |-- pdf_parser.py            # Docling + PyMuPDF
|   |-- table_extractor.py       # Table -> structured JSON -> Parquet
|   |-- entity_extractor.py      # Claude -> entities -> Neo4j
|   |-- chunker.py               # Smart chunking for ColBERT
|   |-- indexer.py               # ColBERT index builder
|-- evaluation/
|   |-- test_rag.py              # DeepEval test suite
|   |-- golden_dataset.json      # Manual Q&A pairs
|   |-- metrics.py               # Custom insurance metrics
|-- data/
|   |-- raw/                     # Original PDFs
|   |-- parsed/                  # Extracted text
|   |-- structured/              # Parquet tables
|   |-- index/                   # ColBERT index
|-- frontend/
|   |-- streamlit_app.py         # Chat UI
|-- docker-compose.yml           # Neo4j + app
|-- pyproject.toml
|-- README.md
```

---

## Why This Architecture Is Well-Suited for Insurance Documents

1. **Multi-modal retrieval matches multi-modal documents.** Insurance docs contain
   prose (plan descriptions), tables (copay schedules), and structured data
   (formularies). Having vector search, graph lookup, AND SQL query as parallel
   retrieval paths means the right tool gets used for each question type.

2. **Entity relationships are first-class citizens.** Insurance is fundamentally
   about relationships: Drug->Tier->Copay, Provider->Network->Plan. A knowledge
   graph captures these directly rather than hoping vector search finds the right
   chunk.

3. **Numerical precision matters.** "Your copay is $30" vs "$35" is the difference
   between a correct and incorrect answer. SQL queries over extracted tables give
   exact numbers, not LLM approximations from embedded text.

4. **The agent can explain its reasoning.** Because Claude decides which tool to
   call and why, the system can show users: "I looked up Lipitor in the formulary
   graph (Tier 2), then queried the copay table for your Gold plan's Tier 2 rate."
   This transparency builds trust for financial/health information.

5. **Cost optimization questions require multi-step reasoning.** "How can I reduce
   my costs for this medication?" requires: find current drug -> find tier -> find
   alternatives in lower tiers -> compare copays -> check if prior auth could move
   tier. An agentic approach handles this naturally through multi-tool calls.

---

## Honest Tradeoffs vs. Simpler Approaches

### vs. Option A (Basic RAG: OpenAI embeddings + Pinecone + LangChain)
| Dimension          | Option A (Simple)              | Option C (This)                       |
|--------------------|-------------------------------|---------------------------------------|
| Build time         | 2-3 weeks                     | 6-10 weeks                            |
| Retrieval accuracy | Good for prose, weak on tables| Strong across all doc types            |
| Numerical accuracy | Poor (embedding-based)        | Strong (SQL on extracted tables)       |
| Relational queries | Poor (no entity awareness)    | Strong (knowledge graph)              |
| Cost per query     | ~$0.01 (embedding + LLM)      | ~$0.02-0.05 (multi-tool + LLM)        |
| Infrastructure     | Managed services              | Neo4j + local ColBERT + DuckDB        |
| Portfolio impact   | "Standard RAG"                | "Shows depth and modern techniques"   |

### vs. Option B (Full context stuffing with Claude 200K)
| Dimension          | Option B (Stuff)               | Option C (This)                      |
|--------------------|-------------------------------|--------------------------------------|
| Build time         | 1 week                        | 6-10 weeks                           |
| Works for <50 pages| Great                         | Overkill                             |
| Works for 500+ pages| Fails (exceeds context)      | Scales well                          |
| Cost per query     | $0.05-0.50 (massive context)  | $0.02-0.05                           |
| Accuracy           | Good (no retrieval errors)    | Better (focused context + structured)|
| Latency            | 5-30 seconds                  | 2-5 seconds                          |

### What could go wrong with Option C:
- **Table extraction is the weakest link.** Insurance PDFs have wildly inconsistent
  formatting. Budget 30% of development time for table extraction tuning.
- **Neo4j is the heaviest dependency.** For a portfolio project, you could substitute
  with NetworkX (in-memory Python graph) to avoid running a database server. You lose
  Cypher queries but gain simplicity.
- **ColBERT index size.** For large document sets, ColBERT indices can be 10-50x larger
  than single-vector indices. For a personal project with <1000 documents, this is
  not a practical concern.
- **Over-engineering risk.** If your document set is small (a few PDFs), Option B
  (context stuffing) genuinely works better. This architecture shines when you have
  10+ documents with complex overlapping information.

---

## Complexity Estimate (Solo Developer)

| Phase                        | Time Estimate | Notes                                    |
|------------------------------|---------------|------------------------------------------|
| Project setup + PDF parsing  | 1 week        | Docling setup, basic text extraction      |
| Table extraction pipeline    | 1.5 weeks     | Hardest part; Docling + Claude cleanup    |
| ColBERT indexing + retrieval | 1 week        | RAGatouille makes this straightforward    |
| Neo4j graph construction     | 1.5 weeks     | Entity extraction prompts, Cypher queries |
| Agent orchestration          | 1 week        | Claude tool-use loop, tool definitions    |
| FastAPI backend              | 0.5 weeks     | Thin API layer                            |
| Streamlit frontend           | 0.5 weeks     | Chat UI with source citations             |
| Evaluation suite             | 1 week        | Golden dataset, DeepEval integration      |
| Testing + polish             | 1.5 weeks     | Edge cases, error handling, demo prep     |
| **Total**                    | **~10 weeks** | Part-time (10-15 hrs/week)                |

**If you want to ship faster (6-week version):**
- Drop Neo4j, use in-memory NetworkX graph instead
- Skip DuckDB, use Claude to answer table questions from extracted JSON directly
- Use a simpler frontend (Gradio instead of Streamlit + FastAPI)
- Reduce golden dataset to 25 Q&A pairs

---

## Getting Started: First 3 Steps

1. **Get sample documents.** Find publicly available insurance documents: CMS
   Medicare plan documents, state marketplace formularies (many are public PDFs),
   sample EOBs. You need 5-10 real documents to develop against.

2. **Build the ingestion pipeline first.** Everything depends on good document
   parsing. Start with Docling for table extraction + PyMuPDF for text. Get this
   working well before touching retrieval.

3. **Build the agent loop second.** Get Claude tool-use working with a mock
   `vector_search` tool that returns hardcoded results. Once the agent loop works,
   swap in real retrieval backends one at a time.

---

## Key Research Sources

- [Agentic RAG in 2026: Enterprise Guide](https://datanucleus.dev/rag-and-agentic-ai/agentic-rag-enterprise-guide-2026)
- [The Ultimate RAG Blueprint 2025/2026](https://langwatch.ai/blog/the-ultimate-rag-blueprint-everything-you-need-to-know-about-rag-in-2025-2026)
- [Agentic RAG Survey (arXiv 2501.09136)](https://arxiv.org/abs/2501.09136)
- [GraphRAG & Knowledge Graphs for 2026](https://flur.ee/fluree-blog/graphrag-knowledge-graphs-making-your-data-ai-ready-for-2026/)
- [Neo4j: What Is GraphRAG?](https://neo4j.com/blog/genai/what-is-graphrag/)
- [Agentic AI & RAG: The 2026 Roadmap to the Cognitive Insurer](https://alexostrovskyy.com/agentic-ai-rag-ml-the-2026-roadmap-to-the-cognitive-insurer/)
- [ColBERT Late Interaction Overview (Weaviate)](https://weaviate.io/blog/late-interaction-overview)
- [ColPali: Efficient Document Retrieval (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/99e9e141aafc314f76b0ca3dd66898b3-Paper-Conference.pdf)
- [Jina ColBERT v2](https://arxiv.org/abs/2408.16672)
- [RAGatouille GitHub](https://github.com/AnswerDotAI/RAGatouille)
- [ColBERT in Practice: Bridging Research and Industry](https://sease.io/2025/11/colbert-in-practice-bridging-research-and-industry.html)
- [Long Context vs. RAG: An Evaluation (arXiv)](https://arxiv.org/abs/2501.01880)
- [RAG vs Long-Context 2026: Is Retrieval Dead?](https://byteiota.com/rag-vs-long-context-2026-retrieval-debate/)
- [Databricks: Long Context RAG Performance](https://www.databricks.com/blog/long-context-rag-performance-llms)
- [LLMs for Structured Data Extraction from PDFs in 2026](https://unstract.com/blog/comparing-approaches-for-using-llms-for-structured-data-extraction-from-pdfs/)
- [Standard RAG Is Dead: Why AI Architecture Split in 2026](https://ucstrategies.com/news/standard-rag-is-dead-why-ai-architecture-split-in-2026/)
- [DeepEval: The LLM Evaluation Framework](https://github.com/confident-ai/deepeval)
- [Top 5 RAG Evaluation Tools for 2026](https://www.getmaxim.ai/articles/the-5-best-rag-evaluation-tools-you-should-know-in-2026/)
- [CrewAI vs LangGraph vs AutoGen Comparison](https://www.datacamp.com/tutorial/crewai-vs-langgraph-vs-autogen)
- [14 AI Agent Frameworks Compared](https://softcery.com/lab/top-14-ai-agent-frameworks-of-2025-a-founders-guide-to-building-smarter-systems)
