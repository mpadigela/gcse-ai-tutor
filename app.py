import streamlit as st
from pydantic import BaseModel
from typing import List, Optional
import re
import time 
import os

# The officially supported Google GenAI SDK imports
from google import genai
from google.genai import types

# Extraction Libraries
import PyPDF2
import requests
import subprocess
import json
import tempfile
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

# PDF Generation Library
from fpdf import FPDF

# --- 1. PYDANTIC SCHEMAS ---
class Flashcard(BaseModel):
    front: str
    back: str

# Schema for Tab 3 (Purely Multiple Choice)
class ExamQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    explanation: str

# Schema for Tab 4 (Written GCSE Questions)
class GCSEQuestion(BaseModel):
    question: str
    marks: int
    question_type: str # e.g., "Short Answer", "Extended Response"
    mark_scheme: str
    explanation: str

class StudyMaterial(BaseModel):
    summary: str  
    flashcards: List[Flashcard]
    exam_questions: List[ExamQuestion] # Stores the MCQs
    gcse_questions: List[GCSEQuestion] # Stores the Written Questions

# --- GRADING SCHEMAS ---
class QuestionResult(BaseModel):
    marks_awarded: int
    status: str # "Full", "Partial", or "Incorrect"
    examiner_comment: str

class ExamGrading(BaseModel):
    results: List[QuestionResult]

# --- 2. EXTRACTION FUNCTIONS (CACHED) ---
@st.cache_data(show_spinner=False)
def extract_pdf_text(uploaded_file) -> str:
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

@st.cache_data(show_spinner=False)
def extract_web_text(url: str) -> str:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.extract()
        
    text = soup.get_text(separator=' ', strip=True)
    return text

def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from any YouTube URL format."""
    regex_pattern = r"(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|shorts\/|watch\?v=|watch\?.+&v=))([A-Za-z0-9_-]{11})"
    match = re.search(regex_pattern, url)
    if not match:
        raise ValueError("Could not extract a valid YouTube Video ID from the provided URL.")
    return match.group(1)


def _fetch_via_youtube_data_api(video_id: str) -> str:
    """
    Method 1: Official YouTube Data API v3 — fetches captions via Google's
    own servers. Never IP-blocked because it's an authenticated API call,
    not a scraping request. Requires YOUTUBE_DATA_API_KEY in st.secrets.

    How to get a key (free, ~$0 cost for caption fetching):
      1. Go to https://console.cloud.google.com/
      2. Enable "YouTube Data API v3"
      3. Create an API key under Credentials
      4. Add it to Streamlit secrets as YOUTUBE_DATA_API_KEY
    """
    try:
        api_key = st.secrets["YOUTUBE_DATA_API_KEY"]
    except Exception:
        raise RuntimeError("YOUTUBE_DATA_API_KEY not found in Streamlit secrets.")

    # Step 1: List available caption tracks for this video
    list_url = "https://www.googleapis.com/youtube/v3/captions"
    params = {"part": "snippet", "videoId": video_id, "key": api_key}
    resp = requests.get(list_url, params=params, timeout=15)
    resp.raise_for_status()
    tracks = resp.json().get("items", [])

    if not tracks:
        raise RuntimeError("YouTube Data API: no caption tracks found for this video.")

    # Step 2: Pick the best English track (prefer manual > auto-generated)
    preferred_langs = ["en", "en-GB", "en-US"]
    chosen = None

    # First pass: manual captions in preferred languages
    for track in tracks:
        snip = track["snippet"]
        if snip["language"] in preferred_langs and not snip.get("trackKind") == "asr":
            chosen = track
            break

    # Second pass: auto-generated captions
    if not chosen:
        for track in tracks:
            snip = track["snippet"]
            if snip["language"] in preferred_langs:
                chosen = track
                break

    # Last resort: any English-ish track
    if not chosen:
        for track in tracks:
            if track["snippet"]["language"].startswith("en"):
                chosen = track
                break

    if not chosen:
        raise RuntimeError(
            f"YouTube Data API: no English caption track found. "
            f"Available languages: {[t['snippet']['language'] for t in tracks]}"
        )

    # Step 3: Download the caption track as plain text (SRV3 / ttml / vtt)
    caption_id = chosen["id"]
    download_url = f"https://www.googleapis.com/youtube/v3/captions/{caption_id}"
    dl_params = {"tfmt": "vtt", "key": api_key}
    dl_resp = requests.get(download_url, params=dl_params, timeout=30)

    # Note: caption download requires OAuth for non-owner videos via the API.
    # If that returns 403, fall through to the timedtext endpoint which is public.
    if dl_resp.status_code == 403:
        raise RuntimeError(
            "YouTube Data API: caption download requires OAuth (video not owned by key holder). "
            "Falling through to timedtext method."
        )

    dl_resp.raise_for_status()
    raw_vtt = dl_resp.text

    # Parse WebVTT: strip header, timestamps and tags, deduplicate lines
    lines = raw_vtt.splitlines()
    text_lines = []
    seen = set()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("WEBVTT") or "-->" in line or re.match(r"^\d+$", line):
            continue
        clean = re.sub(r"<[^>]+>", "", line)  # strip inline tags
        if clean and clean not in seen:
            seen.add(clean)
            text_lines.append(clean)

    text = " ".join(text_lines)
    if not text.strip():
        raise RuntimeError("YouTube Data API: caption track downloaded but contained no readable text.")
    return text


def _fetch_via_timedtext(video_id: str) -> str:
    """
    Method 2: YouTube's internal timedtext API endpoint.
    This is the undocumented endpoint that youtube_transcript_api itself uses,
    but called directly with a more browser-like request. Works on cloud IPs
    when the standard library gets blocked because we control the exact headers
    and can mimic a real browser more precisely.
    """
    # First, fetch the video page to get the available caption tracks
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    page_resp = requests.get(
        f"https://www.youtube.com/watch?v={video_id}",
        headers=headers,
        timeout=20,
    )
    page_resp.raise_for_status()

    # Extract captionTracks JSON from the page source
    match = re.search(r'"captionTracks":(\[.*?\])', page_resp.text)
    if not match:
        raise RuntimeError("timedtext: no captionTracks found in page source (video may have no captions).")

    caption_tracks = json.loads(match.group(1))

    # Pick the best English track
    chosen_url = None
    for priority in [
        lambda t: t.get("languageCode", "").startswith("en") and not t.get("kind") == "asr",
        lambda t: t.get("languageCode", "").startswith("en"),
        lambda t: True,  # any track as last resort
    ]:
        for track in caption_tracks:
            if priority(track):
                chosen_url = track.get("baseUrl")
                break
        if chosen_url:
            break

    if not chosen_url:
        raise RuntimeError("timedtext: no suitable caption track URL found.")

    # Fetch the actual transcript XML
    transcript_resp = requests.get(chosen_url, headers=headers, timeout=20)
    transcript_resp.raise_for_status()

    # Parse the XML transcript
    from xml.etree import ElementTree as ET
    root = ET.fromstring(transcript_resp.text)
    texts = []
    for elem in root.iter("text"):
        raw = elem.text or ""
        # Decode HTML entities and strip tags
        clean = re.sub(r"<[^>]+>", "", raw)
        clean = clean.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'").replace("&quot;", '"')
        if clean.strip():
            texts.append(clean.strip())

    text = " ".join(texts)
    if not text.strip():
        raise RuntimeError("timedtext: parsed transcript was empty.")
    return text


def _fetch_via_ytdlp(url: str) -> str:
    """
    Method 3: yt-dlp without ffmpeg (no format conversion).
    Downloads raw subtitle files only. Works on Streamlit Cloud
    where ffmpeg is not available.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_template = os.path.join(tmpdir, "transcript")
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-sub",
            "--write-sub",
            "--sub-lang", "en",
            "--sub-format", "json3",   # Request json3 natively — no ffmpeg needed
            "--no-check-formats",      # Skip format checks that trigger ffmpeg
            "--output", output_template,
            "--quiet",
            "--no-warnings",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # Parse any subtitle file written, regardless of exit code
        for fname in sorted(os.listdir(tmpdir)):
            fpath = os.path.join(tmpdir, fname)

            if fname.endswith(".json3"):
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                lines = []
                for event in data.get("events", []):
                    chunk = "".join(s.get("utf8", "") for s in event.get("segs", [])).strip()
                    if chunk and chunk != "\n":
                        lines.append(chunk)
                text = " ".join(lines)
                if text.strip():
                    return text

            elif fname.endswith(".vtt"):
                with open(fpath, "r", encoding="utf-8") as f:
                    raw = f.read()
                clean = re.sub(r"WEBVTT.*?\n\n", "", raw, flags=re.DOTALL)
                clean = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3} --> \S+.*?\n", "", clean)
                clean = re.sub(r"<[^>]+>", "", clean)
                text = " ".join(clean.split())
                if text.strip():
                    return text

        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")

    raise RuntimeError("yt-dlp ran but no subtitle file was produced.")


def _fetch_via_transcript_api_direct(video_id: str) -> str:
    """
    Method 4: youtube_transcript_api directly (works locally, blocked on cloud IPs).
    """
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.list(video_id)
    transcript = transcript_list.find_transcript(['en', 'en-GB', 'en-US'])
    transcript_data = transcript.fetch()
    return " ".join([item.text for item in transcript_data])


@st.cache_data(show_spinner=False)
def extract_youtube_transcript(url: str) -> str:
    """
    Fetch a YouTube transcript using a 4-method fallback chain:

      1. YouTube Data API v3  — official Google API, never IP-blocked (needs YOUTUBE_DATA_API_KEY secret)
      2. timedtext endpoint   — direct HTTP call mimicking a browser, works on most cloud IPs
      3. yt-dlp               — resilient CLI tool, no ffmpeg needed
      4. youtube_transcript_api (direct) — works locally

    Each method is tried in order; the first success is returned.
    """
    video_id = _extract_video_id(url)
    errors = []

    for label, fn in [
        ("YouTube Data API v3", lambda: _fetch_via_youtube_data_api(video_id)),
        ("timedtext (direct HTTP)", lambda: _fetch_via_timedtext(video_id)),
        ("yt-dlp", lambda: _fetch_via_ytdlp(url)),
        ("transcript_api (direct)", lambda: _fetch_via_transcript_api_direct(video_id)),
    ]:
        try:
            return fn()
        except Exception as e:
            errors.append(f"{label}: {e}")

    raise ValueError(
        "Could not retrieve a transcript after trying all available methods.\n\n"
        "Attempted:\n" + "\n".join(f"  • {err}" for err in errors) + "\n\n"
        "Most likely fix: Add YOUTUBE_DATA_API_KEY to your Streamlit secrets.\n"
        "Get a free key at https://console.cloud.google.com/ → Enable 'YouTube Data API v3' → Credentials → Create API Key"
    )

# --- 3. AI GENERATION FUNCTIONS ---
def generate_study_materials(text: str, num_cards: int, num_qs: int, complexity: str, exam_board: str, api_key: str) -> StudyMaterial:
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an expert UK GCSE tutor and official examiner specifically trained in the {exam_board} specification. 
    Based on the following source text, generate:
    1. A concise, structured summary of the core concepts.
    2. EXACTLY {num_cards} flashcards.
    3. EXACTLY {num_qs} multiple-choice questions (for the `exam_questions` list).
    4. EXACTLY {num_qs} written GCSE questions (for the `gcse_questions` list).
    
    CRITICAL INSTRUCTIONS FOR {exam_board.upper()} STYLE:
    - If AQA: Focus heavily on standard AQA command words (State, Describe, Explain, Evaluate).
    - If Edexcel: Incorporate scenario-based or context-driven phrasing, focusing on applied knowledge.
    - If OCR: Emphasize synoptic links, practical applications, and scenario-based problem-solving.
    
    QUESTION GENERATION RULES:
    - Multiple Choice (`exam_questions`): Must have 4 plausible options, the correct answer, and an explanation of why it is correct.
    - Written Questions (`gcse_questions`): Must NOT be multiple choice. Focus on short-answer (2-4 marks) and extended-response (6+ marks). Include the max `marks`, the specific `question_type`, the exact `mark_scheme` (bullet points), and examiner `explanation`.
    
    The target complexity level is: {complexity.upper()}.
    
    Source Text:
    {text[:30000]}
    """
    
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StudyMaterial,
            temperature=0.2, 
        )
    )
    
    return StudyMaterial.model_validate_json(response.text)

def grade_exam_submission(questions: List[GCSEQuestion], user_answers: List[str], api_key: str) -> ExamGrading:
    client = genai.Client(api_key=api_key)
    
    exam_data = ""
    for i, q in enumerate(questions):
        exam_data += f"--- QUESTION {i+1} ---\n"
        exam_data += f"Max Marks: {q.marks}\n"
        exam_data += f"Question: {q.question}\n"
        exam_data += f"Official Mark Scheme: {q.mark_scheme}\n"
        exam_data += f"Student Answer: {user_answers[i]}\n\n"
        
    prompt = f"""
    You are a strict but fair UK GCSE examiner.
    Grade the following student written exam submission based strictly on the provided Mark Schemes.
    
    For each question:
    1. Determine the 'marks_awarded' (integer between 0 and the Max Marks).
    2. Determine the 'status':
       - "Full" if marks_awarded == Max Marks.
       - "Partial" if 0 < marks_awarded < Max Marks.
       - "Incorrect" if marks_awarded == 0.
    3. Provide a brief 'examiner_comment' explaining exactly where the student picked up or dropped marks, explicitly referencing the mark scheme. If they got it wrong, tell them why.
    
    Exam Data to Grade:
    {exam_data}
    """
    
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExamGrading,
            temperature=0.1, 
        )
    )
    
    return ExamGrading.model_validate_json(response.text)

# --- 4. PDF GENERATION FUNCTION ---
def create_study_guide_pdf(materials: StudyMaterial, board: str, complexity: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    
    def clean_text(text: str) -> str:
        return text.encode('latin-1', 'replace').decode('latin-1')
    
    # 1. Summary
    pdf.set_font("helvetica", style="B", size=16)
    pdf.cell(0, 10, f"GCSE Study Guide: {board} ({complexity})", new_y="NEXT", new_x="LMARGIN", align="C")
    pdf.ln(10)
    pdf.set_font("helvetica", style="B", size=14)
    pdf.cell(0, 10, "1. Topic Summary", new_y="NEXT", new_x="LMARGIN")
    pdf.set_font("helvetica", size=12)
    pdf.multi_cell(0, 6, clean_text(materials.summary), new_y="NEXT", new_x="LMARGIN")
    pdf.ln(10)
    
    # 2. Flashcards
    pdf.set_font("helvetica", style="B", size=14)
    pdf.cell(0, 10, "2. Revision Flashcards", new_y="NEXT", new_x="LMARGIN")
    for i, card in enumerate(materials.flashcards, 1):
        pdf.set_font("helvetica", style="B", size=12)
        pdf.multi_cell(0, 6, clean_text(f"Q{i}: {card.front}"), new_y="NEXT", new_x="LMARGIN")
        pdf.set_font("helvetica", size=12)
        pdf.multi_cell(0, 6, clean_text(f"A: {card.back}"), new_y="NEXT", new_x="LMARGIN")
        pdf.ln(5)
        
    pdf.add_page()
    
    # 3. Multiple Choice Exam
    pdf.set_font("helvetica", style="B", size=14)
    pdf.cell(0, 10, "3. Multiple Choice Exam", new_y="NEXT", new_x="LMARGIN")
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

    pdf.add_page()
    
    # 4. Written GCSE Exam
    pdf.set_font("helvetica", style="B", size=14)
    pdf.cell(0, 10, "4. GCSE Written Questions", new_y="NEXT", new_x="LMARGIN")
    pdf.ln(5)
    for i, q in enumerate(materials.gcse_questions, 1):
        pdf.set_font("helvetica", style="B", size=12)
        pdf.multi_cell(0, 6, clean_text(f"{i}. [{q.marks} Marks] {q.question}"), new_y="NEXT", new_x="LMARGIN")
        pdf.set_font("helvetica", style="I", size=11)
        pdf.multi_cell(0, 6, clean_text(f"   [Mark Scheme: {q.mark_scheme}]"), new_y="NEXT", new_x="LMARGIN")
        pdf.ln(6)
        
    return bytes(pdf.output())

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="GCSE AI Tutor", page_icon="🎓", layout="wide")

# Separate tracking states for the two different exams
if "study_material" not in st.session_state:
    st.session_state.study_material = None
if "mcq_submitted" not in st.session_state:
    st.session_state.mcq_submitted = False
if "gcse_submitted" not in st.session_state:
    st.session_state.gcse_submitted = False
if "gcse_grades" not in st.session_state:
    st.session_state.gcse_grades = None
if "nav_view" not in st.session_state:
    st.session_state.nav_view = "📝 Summary"    

with st.sidebar:
    st.header("⚙️ Configuration")
    st.divider()
    
    input_type = st.selectbox("Source Type", ["PDF", "Web Article", "YouTube Video"], index=1)
    exam_board = st.selectbox("Exam Board", ["AQA", "Edexcel", "OCR"], index=0)
    complexity = st.selectbox("Complexity Level", ["Beginner", "Intermediate", "Advanced"], index=2)
    num_cards = st.slider("Number of Flashcards", 5, 20, 10)
    num_questions = st.slider("Number of Questions", 5, 20, 5)
    
    st.markdown("<br>" * 5, unsafe_allow_html=True) 
    st.divider()
    st.markdown("<span style='color: gray;'><i>*Built for Manvika</i></span>", unsafe_allow_html=True)

st.title("GCSE Prep Assistant 🎓")

st.markdown("Turn any document, article, or video into interactive study materials, tailored to your GCSE exam board! 🎯")
st.caption("👈 *See the options on the left to customise your output.*")

source_input = None
if input_type == "YouTube Video":
    source_input = st.text_input("Enter YouTube URL (e.g., https://youtu.be/...)")
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
                    st.error("Could not extract enough text from this source.")
                    st.stop()
                
                status.update(label="🧠 AI is writing your study guide... (Usually takes 15-30 seconds)")
                
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        materials = generate_study_materials(raw_text, num_cards, num_questions, complexity, exam_board, api_key)
                        break 
                    except Exception as e:
                        error_msg = str(e)
                        # --- UPDATED: Catch both 429 (Rate Limit) and 503 (Server Busy) ---
                        if ("429" in error_msg or "503" in error_msg) and attempt < max_retries - 1:
                            status.update(label=f"⏳ Google API is experiencing high demand. Retrying in 30 seconds... (Attempt {attempt+1}/{max_retries})")
                            time.sleep(30)
                        else:
                            raise e
                
                st.session_state.study_material = materials
                st.session_state.mcq_submitted = False 
                st.session_state.gcse_submitted = False 
                st.session_state.gcse_grades = None
                # --- ADD THIS NEW LINE TO FORCE THE TAB RESET ---
                st.session_state.nav_view = "📝 Summary" 
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
        ["📝 Summary", "📇 Flashcards", "🎓 Exam Mode (Quick)", "✍️ GCSE Style Questions (AI Graded)"], 
        horizontal=True, 
        label_visibility="collapsed",
        key="nav_view"
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
                
    elif selected_view == "🎓 Exam Mode (Quick)":
        st.subheader("Multiple Choice Exam")
        st.caption("Quickly test your knowledge with these multiple choice questions.")
        
        if st.session_state.mcq_submitted:
            st.success("Exam Completed! Review your results below:")
            score = 0
            
            for i, q in enumerate(st.session_state.study_material.exam_questions):
                user_ans = st.session_state.get(f"mcq_{i}", "No answer selected") 
                st.write(f"**Q{i+1}: {q.question}**")
                
                # --- NEW: Robust Grading Logic ---
                user_clean = user_ans.strip()
                correct_clean = q.correct_answer.strip()
                
                is_correct = False
                if user_clean == correct_clean:
                    is_correct = True
                elif user_clean.startswith(f"{correct_clean}.") or user_clean.startswith(f"{correct_clean})"):
                    is_correct = True
                elif correct_clean in user_clean and len(correct_clean) > 3:
                    is_correct = True
                # ---------------------------------
                
                if is_correct:
                    score += 1
                    st.success(f"Your Answer: {user_ans} (Correct!)")
                    with st.expander("💡 See explanation"):
                        st.write(q.explanation)
                else:
                    st.error(f"Your Answer: {user_ans} | Correct Answer: **{q.correct_answer}**")
                    st.info(f"**Why?** {q.explanation}")
                st.divider()
            
            st.metric(label="Final Score", value=f"{score} / {len(st.session_state.study_material.exam_questions)}")
            if st.button("Retake MCQ Exam", type="primary"):
                st.session_state.mcq_submitted = False
                st.rerun()
                
        else:
            with st.form("mcq_form"):
                for i, q in enumerate(st.session_state.study_material.exam_questions):
                    st.write(f"**Q{i+1}: {q.question}**")
                    options = ["Select an answer..."] + q.options
                    st.radio("Options", options, key=f"mcq_{i}", label_visibility="collapsed")
                    st.divider()
                    
                submitted = st.form_submit_button("Submit Exam")
                if submitted:
                    st.session_state.mcq_submitted = True
                    st.rerun()

    elif selected_view == "✍️ GCSE Style Questions (AI Graded)":
        st.subheader("GCSE Written Questions (AI Graded)")
        st.caption("Practice your written responses. The AI Examiner will mark your paper based on the official Mark Scheme.")
        
        if st.session_state.gcse_submitted and st.session_state.gcse_grades:
            st.success("📝 Exam Graded! Review your AI Examiner feedback below:")
            
            total_score = 0
            total_possible = sum(q.marks for q in st.session_state.study_material.gcse_questions)
            
            for i, q in enumerate(st.session_state.study_material.gcse_questions):
                user_ans = st.session_state.get(f"written_{i}", "No answer provided") 
                grade = st.session_state.gcse_grades.results[i] 
                total_score += grade.marks_awarded
                
                st.write(f"**Q{i+1} [{q.marks} Marks] ({q.question_type}): {q.question}**")
                
                # Red/Amber/Green visual feedback
                if grade.status == "Full":
                    st.success(f"**Your Answer ({grade.marks_awarded}/{q.marks} Marks):**\n\n{user_ans}")
                elif grade.status == "Partial":
                    st.warning(f"**Your Answer ({grade.marks_awarded}/{q.marks} Marks):**\n\n{user_ans}")
                else:
                    st.error(f"**Your Answer (0/{q.marks} Marks):**\n\n{user_ans}")
                
                st.info(f"**👨‍🏫 Examiner Comment:** {grade.examiner_comment}")
                
                with st.expander("Official Mark Scheme & Common Mistakes"):
                    st.write(f"**Mark Scheme:**\n{q.mark_scheme}")
                    st.write(f"**Explanation:**\n{q.explanation}")
                st.divider()
            
            st.metric(label="Final Exam Score", value=f"{total_score} / {total_possible}")
            if st.button("Retake Written Exam", type="primary"):
                st.session_state.gcse_submitted = False
                st.session_state.gcse_grades = None
                st.rerun()
                
        else:
            with st.form("gcse_form"):
                for i, q in enumerate(st.session_state.study_material.gcse_questions):
                    st.write(f"**Q{i+1} [{q.marks} Marks] ({q.question_type}): {q.question}**")
                    st.text_area("Type your answer here:", key=f"written_{i}", height=150, label_visibility="collapsed")
                    st.divider()
                    
                submitted = st.form_submit_button("Submit Exam to AI Examiner")
                if submitted:
                    with st.spinner("👨‍🏫 AI Examiner is grading your exam... (This takes about 10 seconds)"):
                        try:
                            api_key = st.secrets["GEMINI_API_KEY"]
                            user_answers = [st.session_state.get(f"written_{i}", "No answer provided") for i in range(len(st.session_state.study_material.gcse_questions))]
                            
                            max_retries = 3
                            for attempt in range(max_retries):
                                try:
                                    grades = grade_exam_submission(st.session_state.study_material.gcse_questions, user_answers, api_key)
                                    st.session_state.gcse_grades = grades
                                    st.session_state.gcse_submitted = True
                                    st.rerun()
                                    break
                                except Exception as e:
                                    error_msg = str(e)
                                    if ("429" in error_msg or "503" in error_msg) and attempt < max_retries - 1:
                                        st.warning(f"⏳ API is busy. Retrying in 30 seconds... (Attempt {attempt+1}/{max_retries})")
                                        time.sleep(30)
                                    else:
                                        raise e
                                        
                        except Exception as e:
                            st.error(f"Grading Failed: {e}")