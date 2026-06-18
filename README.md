<div align="center">
  <h1>DocMate AI</h1>
  <img width="300" height="300" alt="logo" src="https://github.com/wattacodeAnubhav/DocMate-AI/blob/main/logo.png" />
</div>

DocMate AI is an enterprise-grade, multi-document intelligence engine built using a modular **Retrieval-Augmented Generation (RAG)** architecture. It bridges the gap between unstructured document collections and structured visual analytics by combining advanced orchestration layers, memory-safe data processing pipelines, and a strict command-driven UI layout.

---

## 🚀 Key Features

* **Command-Driven Architecture:** Specialized analysis tasks can be instantly triggered using slash commands (`/dash`, `/mdash`, `/flow`, `/report`, `/search`, `/sql`) to bypass standard conversational QA paths.
* **Interactive Visual Analytics:** Automatically translates raw document data into structured native Altair data visualizations (bar, line, and scatter charts) alongside cross-document entity networks rendered via Mermaid.js.
* **Cognitive Persona Routing:** Allows users to switch the analytical focus of the LLM dynamically using curated lenses (Analyst, Researcher, BI Expert, Teacher) paired with strict anti-hallucination guardrails.
* **Glass-Box Auditing Layer:** Fully exposes the system's internal reasoning stages (`<thinking>`, `<draft>`, `<audit>`) and injects real-time source context tracing down to the exact file name and page number.
* **Resilient Web Integration:** Seamlessly switches execution paths to scrape the live web using Tavily if local context is insufficient or if specifically requested by the user.

---

## ⚙️ Core Engineering Architecture

### 1. Regex-Based Semantic Chunking
Instead of relying on naive character limits that tear text mid-sentence, the ingestion pipeline splits documents at natural sentence boundaries (`[.!?]`). Sentences are then sequentially packed into semantic chunks up to a specific token density, preserving absolute contextual integrity.

### 2. Memory-Safe Batch Generator
To prevent RAM spikes and server crashes when analyzing high-volume text (e.g., 500+ page documents), the execution engine processes PDFs in isolated page batches utilizing Python generators (`yield`) and explicitly triggers hardware garbage collection (`gc.collect()`) after every chunking iteration.

### 3. Dual-Model "Gatekeeper" Pattern
To minimize latency and optimize token expenditure, a fast, lightweight model (`llama3-8b-8192`) handles high-speed intent routing. Once a precise structural intent is verified, processing is routed to a high-capacity reasoning model (`llama-3.3-70b-versatile`).

### 4. Schema Enforced State Machines
Outputs are governed by strict Pydantic models (`ChartConfig`, `MultiChartDashboard`, `GraphData`). This ensures that complex data outputs (like automated datasets and graph coordinates) are mathematically validated before hitting the UI layer, preventing unexpected application crashes.

### 5. Multi-Step Prompt Chaining
The deep-synthesis engine uses a strict 3-step compilation process: 
$$\text{Extract Isolated Facts} \longrightarrow \text{Generate Structural Outline} \longrightarrow \text{Final Report Production}$$ .
This completely prevents common RAG hallucinations by forcing the LLM to write only from verified intermediate structural assets.

---

## 📂 Project Repository Structure

```text
docmate-ai/
├── app.py                 # Main Streamlit Frontend & Design System
├── requirements.txt       # Production Dependency Index
├── .env.example           # Shared Configuration Template
├── .gitignore             # Git Gatekeeper (Excludes local assets/secrets)
└── modules/
    ├── __init__.py
    ├── agent.py           # Core Intelligence, Personas, & Routing
    ├── ingestion.py       # Memory-Safe Semantic PDF Processing Pipeline
    ├── retrieval.py       # Local Vector DB Controller (ChromaDB)
    └── structures.py      # Rigid Pydantic Data Structures & Layouts

```

---

## 🛠️ Installation & Setup

### 1. Clone and Navigate

```bash
git clone <your-repository-url>
cd docmate-ai

```

### 2. Setup Virtual Environment

```bash
# Create Environment
python3 -m venv venv

# Activate Environment (Mac/Linux)
source venv/bin/activate

# Activate Environment (Windows)
venv\Scripts\activate

```

### 3. Install Third-Party Dependencies

```bash
pip install -r requirements.txt

```

### 4. Environment Variables Configuration

Create a `.env` file in the root directory and add your production access tokens:

```text
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

```

### 5. Launch the Application

```bash
streamlit run app.py

```

---

## 🎛️ Command Reference Manual

| Command | Action Target | Underlying System Output Structure |
| --- | --- | --- |
| `/dash` | `/mdash` | Multi-Chart Analytic Dashboard | High-Speed Tabular Mapping + Altair Render |
| `/flow` | Entity Relationship Extraction | Structural Mermaid.js Network Graphs |
| `/report` | Deep Unbiased Document Synthesis | 3-Step Asynchronous Prompt Chain |
| `/search` | Live Internet Search Fallback | Tavily Scraper API Execution |
| `/sql` | Tabular Data Processing | Structured DB Schema Mapping |
| `/table` | Relational Grid Formatting | Clean Markdown Layout Matrix |


```
