# RAG Frontend

A modern chat UI for the RAG document Q&A system.

## Prerequisites

- Node.js 18+
- Python backend running (see below)

## Quick Start

**Terminal 1 — Backend API**

```bash
cd rag_system
source ../venv/bin/activate   # or your venv path
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — Frontend**

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**

## Pages

| Route | Purpose |
|-------|---------|
| `/` | **Chat** — ask questions about indexed documents |
| `/documents` | **Documents** — upload PDFs, view library, run indexing |

Use the **Chat** / **Documents** links in the header to switch pages.

## Features

- Chat interface with markdown answers
- Expandable source citations per response
- Response timing breakdown
- Suggested starter questions
- **PDF upload** with drag-and-drop — auto-indexes after upload
- Document library list with file sizes
- Ingestion progress polling
- Index status badge and re-index button
- Dark theme with glassmorphism UI

## Build for production

```bash
npm run build
npm run preview
```

Serve the `dist/` folder behind your API, or configure your web server to proxy `/api` to the FastAPI backend.
