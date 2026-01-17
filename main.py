"""
NCTB Physics Question Generator - RAG Pipeline
Generates MCQ and CQ questions for Class 9-10 Physics (Bangladesh curriculum)

Usage:
    1. Place your PDF files in the 'data/textbooks/' and 'data/questions/' folders
    2. Run: python main.py --setup (first time only, to build the vector database)
    3. Run: python main.py --generate (to generate questions)
    
Or use interactively:
    python main.py --interactive
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import re

# PDF Processing
import pdfplumber

# OCR Support - Try Surya first, fall back to Tesseract
OCR_ENGINE = None
SURYA_MODELS = {}

try:
    from surya.ocr import run_ocr
    from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_processor
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_ENGINE = "surya"
except ImportError:
    try:
        from pdf2image import convert_from_path
        import pytesseract
        OCR_ENGINE = "tesseract"
    except ImportError:
        OCR_ENGINE = None

# Vector Store
import chromadb
from chromadb.utils import embedding_functions

# LLM - Using OpenAI (can switch to Anthropic)
from openai import OpenAI
# Uncomment below if using Anthropic
# from anthropic import Anthropic

# Configuration
CONFIG = {
    "data_dir": "data",
    "textbook_dir": "data/textbooks",
    "questions_dir": "data/questions",
    "chroma_dir": "chroma_db",
    "collection_name": "physics_nctb",
    "chunk_size": 1000,  # characters per chunk
    "chunk_overlap": 200,
    "embedding_model": "text-embedding-3-small",  # OpenAI embedding model
    "llm_model": "gpt-4o",  # or "claude-sonnet-4-20250514" for Anthropic
    "use_anthropic": False,  # Set to True to use Claude instead of GPT
    "ocr_language": "eng+ben",  # English + Bengali for Tesseract OCR
    "ocr_dpi": 200,  # DPI for PDF to image conversion (lower = faster, higher = better quality)
}


class PDFProcessor:
    """Handles PDF loading and text extraction with OCR fallback"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF, using OCR if needed"""
        text = ""
        
        # First try direct text extraction
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Page {page_num}]\n{page_text}"
        except Exception as e:
            print(f"  Warning: pdfplumber error: {e}")
        
        # If no text extracted or very little text, try OCR
        if not text.strip() or len(text.strip()) < 100:
            print(f"  -> No/little text found via direct extraction, attempting OCR...")
            ocr_text = self.extract_text_with_ocr(pdf_path)
            if ocr_text:
                text = ocr_text
        
        return text
    
    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """Extract text from scanned PDF using OCR (Surya or Tesseract)"""
        global SURYA_MODELS
        
        if OCR_ENGINE is None:
            print("  -> OCR not available. Please install surya-ocr or pytesseract.")
            print("     For best results: pip install surya-ocr")
            print("     Alternative: pip install pytesseract (+ install Tesseract)")
            return ""
        
        text = ""
        try:
            # Convert PDF pages to images
            print(f"  -> Converting PDF to images (this may take a while)...")
            images = convert_from_path(
                pdf_path, 
                dpi=CONFIG["ocr_dpi"],
                fmt='jpeg'
            )
            
            total_pages = len(images)
            print(f"  -> Processing {total_pages} pages with {OCR_ENGINE.upper()} OCR...")
            
            if OCR_ENGINE == "surya":
                # Load Surya models once (lazy loading)
                if not SURYA_MODELS:
                    print("  -> Loading Surya OCR models (first time only)...")
                    SURYA_MODELS['det_processor'] = load_det_processor()
                    SURYA_MODELS['det_model'] = load_det_model()
                    SURYA_MODELS['rec_model'] = load_rec_model()
                    SURYA_MODELS['rec_processor'] = load_rec_processor()
                
                # Process in batches for efficiency
                batch_size = 5
                for batch_start in range(0, total_pages, batch_size):
                    batch_end = min(batch_start + batch_size, total_pages)
                    batch_images = [img.convert("RGB") for img in images[batch_start:batch_end]]
                    
                    print(f"  -> OCR progress: {batch_start + 1}-{batch_end}/{total_pages} pages...")
                    
                    # Run Surya OCR on batch
                    langs = [["en"]] * len(batch_images)  # English for all pages
                    results = run_ocr(
                        batch_images, 
                        langs,
                        SURYA_MODELS['det_model'],
                        SURYA_MODELS['det_processor'],
                        SURYA_MODELS['rec_model'],
                        SURYA_MODELS['rec_processor']
                    )
                    
                    # Extract text from results
                    for i, result in enumerate(results):
                        page_num = batch_start + i + 1
                        page_text = "\n".join([line.text for line in result.text_lines])
                        if page_text.strip():
                            text += f"\n[Page {page_num}]\n{page_text}"
            
            else:  # Tesseract fallback
                for page_num, image in enumerate(images, 1):
                    if page_num % 10 == 0 or page_num == 1:
                        print(f"  -> OCR progress: {page_num}/{total_pages} pages...")
                    
                    try:
                        page_text = pytesseract.image_to_string(
                            image, 
                            lang=CONFIG["ocr_language"]
                        )
                        if page_text.strip():
                            text += f"\n[Page {page_num}]\n{page_text}"
                    except Exception as e:
                        print(f"  -> Warning: OCR failed on page {page_num}: {e}")
            
            print(f"  -> OCR complete! Extracted text from {total_pages} pages.")
            
        except Exception as e:
            print(f"  -> OCR error: {e}")
            if "poppler" in str(e).lower():
                print("  -> Hint: You need to install Poppler.")
                print("     Mac: brew install poppler")
                print("     Ubuntu: sudo apt-get install poppler-utils")
                print("     Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases")
        
        return text
    
    def chunk_text(self, text: str, source: str, doc_type: str = "textbook") -> List[Dict]:
        """Split text into overlapping chunks with metadata"""
        chunks = []
        
        # Clean and normalize text
        text = text.replace('\n\n\n', '\n\n').strip()
        
        # Remove excessive whitespace from OCR
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n +', '\n', text)  # Remove leading spaces after newlines
        
        if len(text) <= self.chunk_size:
            chunks.append({
                "text": text,
                "source": source,
                "doc_type": doc_type,
                "chunk_id": 0
            })
            return chunks
        
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for punct in ['. ', '। ', '\n\n', '\n']:
                    last_punct = text[start:end].rfind(punct)
                    if last_punct != -1 and last_punct > self.chunk_size // 2:
                        end = start + last_punct + len(punct)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "doc_type": doc_type,
                    "chunk_id": chunk_id
                })
                chunk_id += 1
            
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
            if start >= len(text):
                break
                
        return chunks
    
    def process_directory(self, directory: str, doc_type: str = "textbook") -> List[Dict]:
        """Process all PDFs in a directory"""
        all_chunks = []
        pdf_files = list(Path(directory).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return all_chunks
        
        for pdf_path in pdf_files:
            print(f"Processing: {pdf_path.name}")
            text = self.extract_text_from_pdf(str(pdf_path))
            
            if text:
                chunks = self.chunk_text(text, pdf_path.name, doc_type)
                all_chunks.extend(chunks)
                print(f"  -> Created {len(chunks)} chunks")
            else:
                print(f"  -> No text extracted (even with OCR)")
        
        return all_chunks


class VectorStore:
    """Manages ChromaDB vector database"""
    
    def __init__(self, persist_dir: str, collection_name: str, openai_api_key: str):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Use OpenAI embeddings
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=CONFIG["embedding_model"]
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": "NCTB Physics Class 9-10"}
        )
    
    def add_chunks(self, chunks: List[Dict]):
        """Add chunks to the vector database"""
        if not chunks:
            print("No chunks to add")
            return
        
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            # Create unique ID
            chunk_hash = hashlib.md5(
                f"{chunk['source']}_{chunk['chunk_id']}_{chunk['text'][:100]}".encode()
            ).hexdigest()[:12]
            
            documents.append(chunk["text"])
            metadatas.append({
                "source": chunk["source"],
                "doc_type": chunk["doc_type"],
                "chunk_id": chunk["chunk_id"]
            })
            ids.append(f"{chunk['doc_type']}_{chunk_hash}")
        
        # Add in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
            print(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        print(f"Total documents in collection: {self.collection.count()}")
    
    def search(self, query: str, n_results: int = 5, doc_type: Optional[str] = None) -> List[Dict]:
        """Search for relevant chunks"""
        where_filter = None
        if doc_type:
            where_filter = {"doc_type": doc_type}
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        # Format results
        formatted = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                formatted.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else None
                })
        
        return formatted
    
    def clear(self):
        """Clear the collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
        print("Collection cleared")


class QuestionGenerator:
    """Generates physics questions using LLM"""
    
    def __init__(self, api_key: str, use_anthropic: bool = False):
        self.use_anthropic = use_anthropic
        
        if use_anthropic:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
            self.model = "claude-sonnet-4-20250514"
        else:
            self.client = OpenAI(api_key=api_key)
            self.model = CONFIG["llm_model"]
    
    def generate_mcq(self, topic: str, context: str, num_questions: int = 5, 
                     difficulty: str = "medium") -> str:
        """Generate MCQ questions"""
        
        prompt = f"""You are an expert Physics teacher creating questions for SSC (Class 9-10) students in Bangladesh following the NCTB curriculum.

Based on the following content from the Physics textbook and past exam questions, generate {num_questions} new MCQ (Multiple Choice Questions) on the topic: "{topic}"

REFERENCE CONTENT:
{context}

REQUIREMENTS:
1. Each question should have 4 options (ক, খ, গ, ঘ) - you may write in English or Bengali
2. Questions should match SSC Board exam style and difficulty level: {difficulty}
3. Include the correct answer after each question
4. Questions should test conceptual understanding, not just memorization
5. Include numerical problems where appropriate
6. Make sure wrong options are plausible but clearly incorrect

FORMAT each question as:
---
Question [number]:
[Question text]

ক) [Option A]
খ) [Option B]  
গ) [Option C]
ঘ) [Option D]

Correct Answer: [Letter]
Explanation: [Brief explanation why this is correct]
---

Generate the questions now:"""

        return self._call_llm(prompt)
    
    def generate_cq(self, topic: str, context: str, num_questions: int = 2,
                    difficulty: str = "medium") -> str:
        """Generate Creative Questions (CQ) - structured questions with multiple parts"""
        
        prompt = f"""You are an expert Physics teacher creating questions for SSC (Class 9-10) students in Bangladesh following the NCTB curriculum.

Based on the following content from the Physics textbook and past exam questions, generate {num_questions} new CQ (Creative Questions / সৃজনশীল প্রশ্ন) on the topic: "{topic}"

REFERENCE CONTENT:
{context}

CQ STRUCTURE (SSC Board Format):
Each CQ has a stimulus (উদ্দীপক) followed by 4 parts:
- ক) Knowledge-based (জ্ঞানমূলক) - 1 mark - Simple recall
- খ) Comprehension (অনুধাবনমূলক) - 2 marks - Explain/describe
- গ) Application (প্রয়োগমূলক) - 3 marks - Apply concepts to solve
- ঘ) Higher-order (উচ্চতর দক্ষতা) - 4 marks - Analyze/evaluate/create

REQUIREMENTS:
1. Create a realistic stimulus (a short scenario, diagram description, or problem setup)
2. Questions should progress from simple to complex
3. Part গ) should involve calculations where appropriate
4. Part ঘ) should require analysis or comparison
5. Difficulty level: {difficulty}
6. Match SSC Board exam style exactly

FORMAT each CQ as:
---
Creative Question [number]:

Stimulus (উদ্দীপক):
[A short scenario or problem description - 2-4 sentences]

ক) [Knowledge question - 1 mark]
খ) [Comprehension question - 2 marks]
গ) [Application question - 3 marks]  
ঘ) [Higher-order question - 4 marks]

Answer Key:
ক) [Answer]
খ) [Answer]
গ) [Step-by-step solution]
ঘ) [Detailed analysis]
---

Generate the questions now:"""

        return self._call_llm(prompt)
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API"""
        try:
            if self.use_anthropic:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0.7
                )
                return response.choices[0].message.content
        except Exception as e:
            return f"Error generating questions: {e}"


class PhysicsRAG:
    """Main RAG pipeline combining all components"""
    
    def __init__(self, openai_api_key: str, anthropic_api_key: str = None):
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        
        # Initialize components
        self.pdf_processor = PDFProcessor(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"]
        )
        
        self.vector_store = VectorStore(
            persist_dir=CONFIG["chroma_dir"],
            collection_name=CONFIG["collection_name"],
            openai_api_key=openai_api_key
        )
        
        # Use Anthropic if key provided and configured
        use_anthropic = CONFIG["use_anthropic"] and anthropic_api_key
        api_key = anthropic_api_key if use_anthropic else openai_api_key
        
        self.question_generator = QuestionGenerator(
            api_key=api_key,
            use_anthropic=use_anthropic
        )
    
    def setup_database(self):
        """Process PDFs and build vector database"""
        print("\n" + "="*50)
        print("Setting up Vector Database")
        print("="*50)
        
        # Check OCR availability
        if OCR_ENGINE == "surya":
            print("✓ Surya OCR is available (high quality)")
        elif OCR_ENGINE == "tesseract":
            print("✓ Tesseract OCR is available (basic quality)")
            print("  Tip: Install surya-ocr for better results: pip install surya-ocr")
        else:
            print("⚠ OCR not available - scanned PDFs won't be processed")
            print("  To enable OCR, run: pip install surya-ocr pdf2image")
        
        # Create directories if they don't exist
        os.makedirs(CONFIG["textbook_dir"], exist_ok=True)
        os.makedirs(CONFIG["questions_dir"], exist_ok=True)
        
        # Process textbooks
        print(f"\nProcessing textbooks from: {CONFIG['textbook_dir']}")
        textbook_chunks = self.pdf_processor.process_directory(
            CONFIG["textbook_dir"], 
            doc_type="textbook"
        )
        
        # Process past questions
        print(f"\nProcessing past questions from: {CONFIG['questions_dir']}")
        question_chunks = self.pdf_processor.process_directory(
            CONFIG["questions_dir"],
            doc_type="past_question"
        )
        
        # Combine and add to vector store
        all_chunks = textbook_chunks + question_chunks
        
        if all_chunks:
            print(f"\nAdding {len(all_chunks)} chunks to vector database...")
            self.vector_store.add_chunks(all_chunks)
            print("Database setup complete!")
        else:
            print("\nNo chunks to add. Please add PDF files to:")
            print(f"  - Textbooks: {CONFIG['textbook_dir']}/")
            print(f"  - Past Questions: {CONFIG['questions_dir']}/")
    
    def generate_questions(self, topic: str, question_type: str = "mcq",
                          num_questions: int = 5, difficulty: str = "medium") -> str:
        """Generate questions on a topic"""
        
        print(f"\nSearching for content on: {topic}")
        
        # Search for relevant content
        textbook_results = self.vector_store.search(
            query=topic,
            n_results=3,
            doc_type="textbook"
        )
        
        question_results = self.vector_store.search(
            query=topic,
            n_results=3,
            doc_type="past_question"
        )
        
        # Combine context
        context_parts = []
        
        if textbook_results:
            context_parts.append("=== FROM TEXTBOOK ===")
            for r in textbook_results:
                context_parts.append(r["text"])
        
        if question_results:
            context_parts.append("\n=== FROM PAST QUESTIONS ===")
            for r in question_results:
                context_parts.append(r["text"])
        
        if not context_parts:
            return "No relevant content found in the database. Please make sure you've run --setup first."
        
        context = "\n\n".join(context_parts)
        
        print(f"Found {len(textbook_results)} textbook chunks and {len(question_results)} question chunks")
        print(f"Generating {num_questions} {question_type.upper()} questions...")
        
        # Generate questions
        if question_type.lower() == "mcq":
            return self.question_generator.generate_mcq(
                topic=topic,
                context=context,
                num_questions=num_questions,
                difficulty=difficulty
            )
        elif question_type.lower() == "cq":
            return self.question_generator.generate_cq(
                topic=topic,
                context=context,
                num_questions=num_questions,
                difficulty=difficulty
            )
        else:
            return f"Unknown question type: {question_type}. Use 'mcq' or 'cq'"


def interactive_mode(rag: PhysicsRAG):
    """Interactive question generation mode"""
    print("\n" + "="*50)
    print("NCTB Physics Question Generator")
    print("Interactive Mode")
    print("="*50)
    
    while True:
        print("\nOptions:")
        print("1. Generate MCQ questions")
        print("2. Generate CQ (Creative Questions)")
        print("3. Rebuild database")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "4":
            print("Goodbye!")
            break
        
        if choice == "3":
            rag.setup_database()
            continue
        
        if choice not in ["1", "2"]:
            print("Invalid choice")
            continue
        
        question_type = "mcq" if choice == "1" else "cq"
        
        # Get parameters
        topic = input("\nEnter topic/chapter (e.g., 'Newton's laws of motion', 'তাপ ও তাপমাত্রা'): ").strip()
        if not topic:
            print("Topic cannot be empty")
            continue
        
        num_q = input(f"Number of questions (default: {'5' if question_type == 'mcq' else '2'}): ").strip()
        num_questions = int(num_q) if num_q.isdigit() else (5 if question_type == "mcq" else 2)
        
        diff = input("Difficulty (easy/medium/hard, default: medium): ").strip().lower()
        difficulty = diff if diff in ["easy", "medium", "hard"] else "medium"
        
        # Generate
        result = rag.generate_questions(
            topic=topic,
            question_type=question_type,
            num_questions=num_questions,
            difficulty=difficulty
        )
        
        print("\n" + "="*50)
        print("GENERATED QUESTIONS")
        print("="*50)
        print(result)
        
        # Option to save
        save = input("\nSave to file? (y/n): ").strip().lower()
        if save == "y":
            filename = f"generated_{question_type}_{topic.replace(' ', '_')[:20]}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"Saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description="NCTB Physics Question Generator")
    parser.add_argument("--setup", action="store_true", help="Process PDFs and build database")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--generate", action="store_true", help="Generate questions")
    parser.add_argument("--topic", type=str, help="Topic for question generation")
    parser.add_argument("--type", type=str, choices=["mcq", "cq"], default="mcq", help="Question type")
    parser.add_argument("--num", type=int, default=5, help="Number of questions")
    parser.add_argument("--difficulty", type=str, choices=["easy", "medium", "hard"], default="medium")
    
    args = parser.parse_args()
    
    # Load API keys from environment
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not openai_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Initialize RAG
    rag = PhysicsRAG(
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key
    )
    
    if args.setup:
        rag.setup_database()
    
    elif args.interactive:
        interactive_mode(rag)
    
    elif args.generate:
        if not args.topic:
            print("Error: --topic is required for generation")
            print("Example: python main.py --generate --topic 'গতি' --type mcq --num 5")
            return
        
        result = rag.generate_questions(
            topic=args.topic,
            question_type=args.type,
            num_questions=args.num,
            difficulty=args.difficulty
        )
        print(result)
    
    else:
        # Default to interactive mode
        interactive_mode(rag)


if __name__ == "__main__":
    main()
