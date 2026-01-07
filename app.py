"""
NCTB Question Generator - Streamlit Web App
A web interface for generating physics questions for Class 9-10 students in Bangladesh
"""

import streamlit as st
import os
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import re
from io import BytesIO

# PDF Processing
import pdfplumber

# OCR Support
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Vector Store
import chromadb
from chromadb.utils import embedding_functions

# PDF Generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# LLM APIs
from openai import OpenAI

# Page config
st.set_page_config(
    page_title="NCTB Question Generator",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
    .success-box {
        padding: 1rem;
        background-color: #E8F5E9;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
    .question-box {
        padding: 1.5rem;
        background-color: #F5F5F5;
        border-radius: 0.5rem;
        margin: 1rem 0;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# Configuration for subjects and chapters
CURRICULUM_DATA = {
    "Class 9-10": {
        "Physics": {
            "chapters": [
                "Chapter 1: Physical Quantities and Measurement",
                "Chapter 2: Motion",
                "Chapter 3: Force",
                "Chapter 4: Work, Power and Energy",
                "Chapter 5: Matter: Structure and Properties",
                "Chapter 6: Effects of Heat on Matter",
                "Chapter 7: Waves and Sound",
                "Chapter 8: Light",
                "Chapter 9: Electricity",
                "Chapter 10: Magnetic Effects of Electric Current",
                "Chapter 11: Electronics",
                "Chapter 12: Modern Physics and Radioactivity",
            ]
        }
    }
}


class VectorStoreManager:
    """Manages ChromaDB vector database"""
    
    def __init__(self, persist_dir: str, collection_name: str, api_key: str):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Use OpenAI embeddings
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
    
    def search(self, query: str, n_results: int = 5, doc_type: Optional[str] = None) -> List[Dict]:
        """Search for relevant chunks"""
        if self.collection.count() == 0:
            return []
            
        where_filter = None
        if doc_type:
            where_filter = {"doc_type": doc_type}
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        formatted = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                formatted.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else None
                })
        
        return formatted
    
    def get_count(self) -> int:
        return self.collection.count()


class QuestionGenerator:
    """Generates questions using LLM"""
    
    def __init__(self, api_key: str, provider: str = "openai"):
        self.provider = provider
        
        if provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
            self.model = "claude-sonnet-4-20250514"
        else:
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4o"
    
    def generate_mcq(self, topic: str, context: str, num_questions: int = 5, 
                     difficulty: str = "medium") -> str:
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
        try:
            if self.provider == "anthropic":
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


def create_pdf(content: str, title: str = "Generated Questions") -> BytesIO:
    """Create a PDF from the generated questions"""
    buffer = BytesIO()
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1  # Center
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        spaceAfter=12
    )
    
    story = []
    
    # Title
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 20))
    
    # Content - split by lines and add as paragraphs
    lines = content.split('\n')
    for line in lines:
        if line.strip():
            # Escape special characters for ReportLab
            safe_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(safe_line, body_style))
        else:
            story.append(Spacer(1, 6))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


def main():
    # Header
    st.markdown('<p class="main-header">📚 NCTB Question Generator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate MCQ and Creative Questions for SSC Students</p>', unsafe_allow_html=True)
    
    # Create two columns for layout
    col_settings, col_output = st.columns([1, 1.5])
    
    with col_settings:
        st.markdown("### ⚙️ Settings")
        
        # API Configuration
        with st.expander("🔑 API Configuration", expanded=True):
            api_provider = st.selectbox(
                "Select AI Provider",
                ["OpenAI (GPT-4)", "Anthropic (Claude)"],
                index=0
            )
            
            api_key = st.text_input(
                "Enter API Key",
                type="password",
                placeholder="sk-... or sk-ant-..."
            )
            
            if api_key:
                st.success("✓ API Key entered")
            else:
                st.warning("Please enter your API key")
        
        st.markdown("---")
        
        # Curriculum Selection
        st.markdown("### 📖 Select Curriculum")
        
        # Class selection
        selected_class = st.selectbox(
            "Class",
            list(CURRICULUM_DATA.keys()),
            index=0
        )
        
        # Subject selection
        available_subjects = list(CURRICULUM_DATA[selected_class].keys())
        selected_subject = st.selectbox(
            "Subject",
            available_subjects,
            index=0
        )
        
        # Chapter selection
        available_chapters = CURRICULUM_DATA[selected_class][selected_subject]["chapters"]
        selected_chapters = st.multiselect(
            "Select Chapter(s)",
            available_chapters,
            default=[available_chapters[0]] if available_chapters else []
        )
        
        st.markdown("---")
        
        # Question Configuration
        st.markdown("### 📝 Question Settings")
        
        question_type = st.selectbox(
            "Question Type",
            ["MCQ (Multiple Choice)", "CQ (Creative Questions)", "Both"],
            index=0
        )
        
        col_num, col_diff = st.columns(2)
        
        with col_num:
            if question_type == "MCQ (Multiple Choice)":
                num_questions = st.number_input("Number of MCQs", min_value=1, max_value=20, value=5)
            elif question_type == "CQ (Creative Questions)":
                num_questions = st.number_input("Number of CQs", min_value=1, max_value=10, value=2)
            else:
                num_mcq = st.number_input("Number of MCQs", min_value=1, max_value=20, value=5)
                num_cq = st.number_input("Number of CQs", min_value=1, max_value=10, value=2)
        
        with col_diff:
            difficulty = st.selectbox(
                "Difficulty",
                ["Easy", "Medium", "Hard"],
                index=1
            )
        
        st.markdown("---")
        
        # Generate button
        generate_button = st.button("🚀 Generate Questions", type="primary", use_container_width=True)
    
    # Output column
    with col_output:
        st.markdown("### 📄 Generated Questions")
        
        # Check if we should generate
        if generate_button:
            if not api_key:
                st.error("❌ Please enter your API key first!")
            elif not selected_chapters:
                st.error("❌ Please select at least one chapter!")
            else:
                # Determine provider
                provider = "anthropic" if "Anthropic" in api_provider else "openai"
                
                with st.spinner("🔄 Generating questions... This may take a moment."):
                    try:
                        # Initialize vector store
                        vector_store = VectorStoreManager(
                            persist_dir="chroma_db",
                            collection_name="physics_nctb",
                            api_key=api_key if provider == "openai" else os.environ.get("OPENAI_API_KEY", api_key)
                        )
                        
                        # Check if database has content
                        if vector_store.get_count() == 0:
                            st.warning("⚠️ No content in database. Please run setup first using the command line: `python main.py --setup`")
                            st.stop()
                        
                        # Build search query from selected chapters
                        chapter_topics = " ".join([ch.split(": ")[-1] for ch in selected_chapters])
                        
                        # Search for relevant content
                        textbook_results = vector_store.search(
                            query=chapter_topics,
                            n_results=5,
                            doc_type="textbook"
                        )
                        
                        question_results = vector_store.search(
                            query=chapter_topics,
                            n_results=3,
                            doc_type="past_question"
                        )
                        
                        # Build context
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
                            st.warning("⚠️ No relevant content found for selected chapters.")
                            st.stop()
                        
                        context = "\n\n".join(context_parts)
                        
                        # Initialize question generator
                        generator = QuestionGenerator(api_key=api_key, provider=provider)
                        
                        # Generate questions
                        all_questions = ""
                        
                        if question_type == "MCQ (Multiple Choice)":
                            all_questions = generator.generate_mcq(
                                topic=chapter_topics,
                                context=context,
                                num_questions=num_questions,
                                difficulty=difficulty.lower()
                            )
                        elif question_type == "CQ (Creative Questions)":
                            all_questions = generator.generate_cq(
                                topic=chapter_topics,
                                context=context,
                                num_questions=num_questions,
                                difficulty=difficulty.lower()
                            )
                        else:
                            mcq_questions = generator.generate_mcq(
                                topic=chapter_topics,
                                context=context,
                                num_questions=num_mcq,
                                difficulty=difficulty.lower()
                            )
                            cq_questions = generator.generate_cq(
                                topic=chapter_topics,
                                context=context,
                                num_questions=num_cq,
                                difficulty=difficulty.lower()
                            )
                            all_questions = f"=== MCQ QUESTIONS ===\n\n{mcq_questions}\n\n=== CREATIVE QUESTIONS ===\n\n{cq_questions}"
                        
                        # Store in session state
                        st.session_state['generated_questions'] = all_questions
                        st.session_state['question_title'] = f"{selected_subject} - {', '.join([ch.split(': ')[-1] for ch in selected_chapters])}"
                        
                        st.success("✅ Questions generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Display generated questions
        if 'generated_questions' in st.session_state and st.session_state['generated_questions']:
            st.markdown('<div class="question-box">', unsafe_allow_html=True)
            st.text(st.session_state['generated_questions'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download buttons
            st.markdown("---")
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                # Download as TXT
                st.download_button(
                    label="📥 Download as TXT",
                    data=st.session_state['generated_questions'],
                    file_name="generated_questions.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col_dl2:
                # Download as PDF
                try:
                    pdf_buffer = create_pdf(
                        st.session_state['generated_questions'],
                        title=st.session_state.get('question_title', 'Generated Questions')
                    )
                    st.download_button(
                        label="📥 Download as PDF",
                        data=pdf_buffer,
                        file_name="generated_questions.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.warning(f"PDF generation failed: {e}. Please use TXT download.")
        else:
            st.info("👈 Configure settings and click 'Generate Questions' to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888;'>Built for NCTB Curriculum | Bangladesh Education Board</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
