# 📈 Finance ChatBot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about stocks, companies, and financial markets — powered by **Pinecone**, **HuggingFace Embeddings**, and **DeepSeek-V3** via the HuggingFace Inference Router. Built with **Streamlit**.

---

## Project Highlights
-Intelligent Q&A: Answers Financial questions based on factual information from provided PDF documents.

-RAG Architecture: Implements a robust RAG pipeline for enhanced accuracy and context awareness.

-Efficient Information Retrieval: Utilizes a Pinecone vector database for fast and semantically relevant document chunk retrieval.

-Data Processing Pipeline: Includes modules for loading, splitting, and embedding text from PDF documents.

-LLM Integration: Seamlessly integrates with powerful language models to generate coherent and informative responses.

---

## How It Works

1. **Ingest** (`ingest.py`) — Loads PDF documents from the `data/` folder, splits them into chunks, generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`, and uploads them to a Pinecone vector index.
2. **Chat** (`financeBot.py`) — Accepts user questions, retrieves the most relevant chunks from Pinecone, and passes them to DeepSeek-V3 to generate a grounded answer.

```
data/ (PDFs)  →  ingest.py  →  Pinecone Index
                                     ↓
User Question  →  financeBot.py  →  Retrieved Context  →  LLM  →  Answer
```

---

## Project Structure

```
chatBot/
├── financeBot.py      # Streamlit chat app
├── ingest.py          # PDF ingestion & embedding pipeline
├── requirements.txt   # Python dependencies
├── data/              # Place your PDF files here
│   └── Stock_Market.pdf
└── .env               # API keys (not committed)
```

---

## Setup

### 1. Clone & Install Dependencies

```bash
git clone <your-repo-url>
cd chatBot
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=your_pinecone_index_name
HF_TOKEN=your_huggingface_token
```

| Variable          | Description                                      |
|-------------------|--------------------------------------------------|
| `PINECONE_API_KEY` | API key from [pinecone.io](https://www.pinecone.io) |
| `PINECONE_INDEX`   | Name of your Pinecone index                     |
| `HF_TOKEN`         | HuggingFace token (used to access inference router) |

### 3. Ingest Documents

Add your PDF files to the `data/` folder, then run:

```bash
python ingest.py
```

This will chunk the PDFs and upload the embeddings to Pinecone.

### 4. Run the Chat App

```bash
streamlit run financeBot.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Tech Stack

| Component        | Tool/Model                                      |
|------------------|-------------------------------------------------|
| UI               | Streamlit                                       |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store     | Pinecone                                        |
| LLM              | DeepSeek-V3 via HuggingFace Inference Router    |
| PDF Loader       | PyMuPDF (`pymupdf`)                             |
| RAG Framework    | LangChain                                       |

---

## Notes

- The bot only answers based on the content of the ingested PDFs — it will not hallucinate answers outside of that context.
- It does **not** provide financial advice or investment recommendations.
- Re-run `ingest.py` whenever you add new documents to the `data/` folder.
