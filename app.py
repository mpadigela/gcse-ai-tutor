import streamlit as st
from pydantic import BaseModel
from typing import List
import re
import time 

# The officially supported Google GenAI SDK imports
from google import genai
from google.genai import types

# Extraction Libraries
import PyPDF2
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

# PDF Generation Library
from fpdf import FPDF

# --- 1. PYDANTIC SCHEMAS ---
class Flashcard(BaseModel):
    front: str
    back: str

class ExamQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str

class StudyMaterial(BaseModel):
    summary: str  
    flashcards: List[Flashcard]
    exam_questions: List[ExamQuestion]

# --- 2. EXTRACTION FUNCTIONS ---
def extract_pdf_text(uploaded_file) -> str:
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def extract_web_text(url: str) -> str:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.extract()
        
    text = soup.get_text(separator=' ', strip=True)
    return text

def extract_youtube_transcript(url: str) -> str:
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if not video_id_match:
        raise ValueError("Could not extract YouTube Video ID.")
    video_id = video_id_match.group(1)
    
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id).to_raw_data()
    text = " ".join([item['text'] for item in transcript_list])
    return text

# --- 3. AI GENERATION FUNCTION ---
def generate_study_materials(text: str, num_cards: int, num_qs: int, complexity: str, exam_board: str, api_key: str) -> StudyMaterial:
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an expert UK GCSE tutor and official examiner specifically trained in the {exam_board} specification. 
    Based on the following source text, generate:
    1. A concise, structured summary of the core concepts.
    2. EXACTLY {num_cards} flashcards.
    3. EXACTLY {num_qs} multiple-choice exam questions.
    
    CRITICAL INSTRUCTIONS FOR {exam_board.upper()} STYLE:
    - If AQA: Focus heavily on standard AQA command words (State, Describe, Explain, Evaluate). Questions should be direct, clearly structured, and test precise syllabus definitions.
    - If Edexcel: Incorporate more scenario-based or context-driven phrasing where applicable. Focus on logical deduction, applied knowledge, and data interpretation.
    
    The target complexity level is: {complexity.upper()}.
    - If BEGINNER: Focus on fundamental definitions and simple recall.
    - If INTERMEDIATE: Focus on standard GCSE level understanding and application.
    - If ADVANCED: You MUST mimic the exact tone, structure, and difficulty of REAL higher-tier past exam papers for {exam_board.upper()}. Flashcard answers must reflect actual mark-scheme key points (use bullet points if necessary). Exam questions must use authentic {exam_board.upper()} phrasing, require multi-step reasoning, and include highly plausible distractors designed to catch common student misconceptions.
    
    Source Text:
    {text[:30000]}
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StudyMaterial,
            temperature=0.2, 
        )
    )
    
    return StudyMaterial.model_validate_json(response.text)

# --- 4. PDF GENERATION FUNCTION ---
def create_study_guide_pdf(materials: StudyMaterial, board: str, complexity: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    
    def clean_text(text: str) -> str:
        return text.encode('latin-1', 'replace').decode('latin-1')
    
    pdf.set_font("helvetica", style="B", size=16)
    pdf.cell(0, 10, f"GCSE Study Guide: {board} ({complexity})", new_y="NEXT", new_x="LMARGIN", align="C")
    pdf.ln(10)
    
    pdf.set_font("helvetica", style="B", size=14)
    pdf.cell(0, 10, "1. Topic Summary", new_y="NEXT", new_x="LMARGIN")
    pdf.set_font("helvetica", size=12)
    pdf.multi_cell(0, 6, clean_text(materials.summary), new_y="NEXT", new_x="LMARGIN")
    pdf.ln(10)
    
    pdf.set_font("helvetica", style="B", size=14)
    pdf.cell(0, 10, "2. Revision Flashcards", new_y="NEXT", new_x="LMARGIN")
    for i, card in enumerate(materials.flashcards, 1):
        pdf.set_font("helvetica", style="B", size=12)
        pdf.multi_cell(0, 6, clean_text(f"Q{i}: {card.front}"), new_y="NEXT", new_x="LMARGIN")
        pdf.set_font("helvetica", size=12)
        pdf.multi_cell(0, 6, clean_text(f"A: {card.back}"), new_y="NEXT", new_x="LMARGIN")
        pdf.ln(5)
        
    pdf.add_page()
    
    pdf.set_font("helvetica", style="B", size=14)
    pdf.cell(0, 10, "3. Mock Exam Questions", new_y="NEXT", new_x="LMARGIN")
    pdf.ln(5)
    for i, q in enumerate(materials.exam_questions, 1):
        pdf.set_font("helvetica", style="B", size=12)
        pdf.multi_cell(0, 6, clean_text(f"{i}. {q.question}"), new_y="NEXT", new_x="LMARGIN")
        
        pdf.set_font("helvetica", size=12)
        for opt in q.options:
            pdf.multi_cell(0, 6, clean_text(f"   - {opt}"), new_y="NEXT", new_x="LMARGIN")
            
        pdf.set_font("helvetica", style="I", size=11)
        pdf.multi_cell(0, 6, clean_text(f"   [Correct Answer: {q.correct_answer}]"), new_y="NEXT", new_x="LMARGIN")
        pdf.ln(6)
        
    return bytes(pdf.output())

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="GCSE AI Tutor", page_icon="🎓", layout="wide")

if "study_material" not in st.session_state:
    st.session_state.study_material = None
if "exam_submitted" not in st.session_state:
    st.session_state.exam_submitted = False

with st.sidebar:
    st.header("⚙️ Configuration")
    st.divider()
    
    input_type = st.selectbox("Source Type", ["PDF", "Web Article", "YouTube Video"], index=1)
    exam_board = st.selectbox("Exam Board", ["AQA", "Edexcel"], index=0)
    complexity = st.selectbox("Complexity Level", ["Beginner", "Intermediate", "Advanced"], index=2)
    num_cards = st.slider("Number of Flashcards", 5, 20, 10)
    
    # --- UPDATED: Set default value to 5 ---
    num_questions = st.slider("Number of Questions", 5, 20, 5)
    
    st.markdown("<br>" * 5, unsafe_allow_html=True) 
    st.divider()
    st.markdown("<span style='color: gray;'><i>*Built for Manvika</i></span>", unsafe_allow_html=True)

st.title("GCSE Prep Assistant 🎓")
st.markdown("Turn any document, article, or video into interactive study materials, tailored to your exam board! 🎯")
st.caption("👈 *See the options on the left to customise your output.*")

source_input = None
if input_type == "YouTube Video":
    source_input = st.text_input("Enter YouTube URL (e.g., https://www.youtube.com/watch?v=...)")
elif input_type == "Web Article":
    source_input = st.text_input("Enter Article URL:")
else:
    source_input = st.file_uploader("Upload GCSE Material", type=["pdf"])

if st.button("Generate Materials", type="primary"):
    if not source_input:
        st.error("Please provide a valid input source.")
    else:
        with st.status("Initializing processing pipeline...", expanded=True) as status:
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                
                status.update(label="📥 Extracting raw text from source...")
                raw_text = ""
                if input_type == "PDF":
                    raw_text = extract_pdf_text(source_input)
                elif input_type == "Web Article":
                    raw_text = extract_web_text(source_input)
                elif input_type == "YouTube Video":
                    raw_text = extract_youtube_transcript(source_input)
                
                if len(raw_text.strip()) < 100:
                    status.update(label="Extraction Failed", state="error")
                    st.error("Could not extract enough text from this source. The website might be blocking scrapers, or it is an image-heavy page.")
                    st.stop()
                
                status.update(label="🧠 AI is writing your study guide... (Usually takes 15-30 seconds)")
                
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        materials = generate_study_materials(raw_text, num_cards, num_questions, complexity, exam_board, api_key)
                        break 
                    except Exception as e:
                        if "429" in str(e) and attempt < max_retries - 1:
                            status.update(label=f"⏳ Google API limit reached. Waiting 60 seconds for quota to reset... (Attempt {attempt+1}/{max_retries})")
                            time.sleep(60)
                        else:
                            raise e 
                
                st.session_state.study_material = materials
                st.session_state.exam_submitted = False 
                status.update(label="Generation Complete!", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="An error occurred", state="error")
                st.error(f"Error Details: {e}")

if st.session_state.study_material:
    
    col1, col2 = st.columns([3,1])
    with col2:
        pdf_data = create_study_guide_pdf(st.session_state.study_material, exam_board, complexity)
        st.download_button(
            label="📄 Export Full Guide to PDF",
            data=pdf_data,
            file_name=f"gcse_{exam_board.lower()}_{complexity.lower()}_guide.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    st.divider()
    
    selected_view = st.radio(
        "Navigation", 
        ["📝 Summary", "📇 Flashcards", "🎓 Exam Mode"], 
        horizontal=True, 
        label_visibility="collapsed"
    )
    st.divider()
    
    if selected_view == "📝 Summary":
        st.subheader("Topic Summary")
        st.write(st.session_state.study_material.summary)
        
    elif selected_view == "📇 Flashcards":
        st.subheader("Interactive Flashcards")
        for i, card in enumerate(st.session_state.study_material.flashcards):
            with st.expander(f"Card {i+1}: **{card.front}**"):
                st.info(f"**Answer:** {card.back}")
                
    elif selected_view == "🎓 Exam Mode":
        st.subheader("Mock Exam")
        
        if st.session_state.exam_submitted:
            st.success("Exam Completed! Review your results below:")
            score = 0
            
            for i, q in enumerate(st.session_state.study_material.exam_questions):
                user_ans = st.session_state.get(f"q_{i}", "No answer selected") 
                
                st.write(f"**Q{i+1}: {q.question}**")
                if user_ans == q.correct_answer:
                    score += 1
                    st.success(f"Your Answer: {user_ans} (Correct!)")
                else:
                    st.error(f"Your Answer: {user_ans} | Correct Answer: **{q.correct_answer}**")
                st.divider()
            
            st.metric(label="Final Score", value=f"{score} / {len(st.session_state.study_material.exam_questions)}")
            
            if st.button("Retake Exam", type="primary"):
                st.session_state.exam_submitted = False
                st.rerun()
                
        else:
            with st.form("exam_form"):
                for i, q in enumerate(st.session_state.study_material.exam_questions):
                    st.write(f"**Q{i+1}: {q.question}**")
                    options = ["Select an answer..."] + q.options
                    st.radio("Options", options, key=f"q_{i}", label_visibility="collapsed")
                    st.divider()
                    
                submitted = st.form_submit_button("Submit Exam")
                
                if submitted:
                    st.session_state.exam_submitted = True
                    st.rerun()