# NCTB Physics Question Generator

A RAG (Retrieval-Augmented Generation) pipeline for generating SSC Physics questions based on the NCTB curriculum of Bangladesh.

## Features

- 📚 Processes PDF textbooks and past exam papers
- 🔍 Semantic search using ChromaDB vector database
- ✨ Generates MCQ and CQ (Creative Questions) in SSC Board exam style
- 🌐 Supports both English and Bengali content
- 🔄 Works with OpenAI (GPT-4) or Anthropic (Claude)

## Quick Start

### 1. Install Dependencies

```bash
cd physics_rag
pip install -r requirements.txt
```

### 2. Set Up API Keys

```bash
# Required: OpenAI API key (for embeddings)
export OPENAI_API_KEY='your-openai-key-here'

# Optional: Anthropic API key (if you want to use Claude for generation)
export ANTHROPIC_API_KEY='your-anthropic-key-here'
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
```

### 3. Add Your PDF Files

```
physics_rag/
├── data/
│   ├── textbooks/      <- Put NCTB Physics textbook PDFs here
│   └── questions/      <- Put past board exam question PDFs here
```

### 4. Build the Database

```bash
python main.py --setup
```

This will:
- Extract text from all PDFs
- Chunk the text into searchable segments
- Create embeddings and store in ChromaDB

### 5. Generate Questions

**Interactive Mode (recommended for first-time use):**
```bash
python main.py --interactive
```

**Command Line:**
```bash
# Generate 5 MCQs on Newton's Laws
python main.py --generate --topic "Newton's laws of motion" --type mcq --num 5

# Generate 2 Creative Questions on Heat
python main.py --generate --topic "তাপ ও তাপমাত্রা" --type cq --num 2 --difficulty hard
```

## Usage Examples

### Generate MCQ Questions
```bash
python main.py --generate --topic "Work, Energy and Power" --type mcq --num 10
```

### Generate Creative Questions (CQ)
```bash
python main.py --generate --topic "Light and Reflection" --type cq --num 3 --difficulty medium
```

## Project Structure

```
physics_rag/
├── main.py              # Main RAG pipeline
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── data/
│   ├── textbooks/      # NCTB Physics textbook PDFs
│   └── questions/      # Past exam question PDFs
└── chroma_db/          # Vector database (created automatically)
```
