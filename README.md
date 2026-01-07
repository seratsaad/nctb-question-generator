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

## Configuration

Edit the `CONFIG` dictionary in `main.py` to customize:

```python
CONFIG = {
    "chunk_size": 1000,        # Characters per chunk
    "chunk_overlap": 200,       # Overlap between chunks
    "embedding_model": "text-embedding-3-small",
    "llm_model": "gpt-4o",      # Or "claude-sonnet-4-20250514"
    "use_anthropic": False,     # Set True to use Claude
}
```

## Tips for Best Results

1. **Quality PDFs**: Use searchable PDFs (not scanned images) for best text extraction
2. **Organize by Chapter**: Name your PDFs clearly (e.g., `chapter_01_motion.pdf`)
3. **Include Past Papers**: The more past exam questions you add, the better the generated questions will match the board exam style
4. **Specific Topics**: Be specific with topic names for better retrieval (e.g., "Newton's second law F=ma" instead of just "Newton")

## Troubleshooting

**"No relevant content found"**
- Make sure you've run `python main.py --setup` first
- Check that your PDFs are in the correct folders
- Try a different topic name or be more/less specific

**"Error processing PDF"**
- The PDF might be scanned/image-based. Consider using OCR first
- The PDF might be corrupted or password-protected

**"API Error"**
- Check your API key is correctly set
- Ensure you have API credits available

## Next Steps

This is a simple RAG implementation. For the web interface version, you'll need:
- A web framework (Flask/FastAPI for backend)
- A frontend (React/Next.js)
- User authentication
- Chapter/topic selection UI
- Question history storage

Let me know when you're ready to build the web interface!
