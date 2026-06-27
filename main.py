"""
ResumeAI — Hackathon Backend
Run: uvicorn main:app --reload
"""

import os, json, re
import fitz                          # PyMuPDF  →  pip install pymupdf
import google.generativeai as genai  # pip install google-generativeai
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv       # pip install python-dotenv

load_dotenv()

# ── Gemini setup ───────────────────────────────────────────────
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# ── App ────────────────────────────────────────────────────────
app = FastAPI(title="ResumeAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # open for hackathon; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic response model ────────────────────────────────────
class ScoreBreakdown(BaseModel):
    keywords: int
    formatting: int
    content_quality: int
    completeness: int

class Improvement(BaseModel):
    original: str
    improved: str
    reason: str

class AnalysisResult(BaseModel):
    ats_score: int
    score_breakdown: ScoreBreakdown
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    missing_skills: list[str]
    improvements: list[Improvement]

# ── PDF / DOCX / TXT text extraction ──────────────────────────
def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        doc  = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()

    if ext == "docx":
        import docx, io
        d = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in d.paragraphs if p.text.strip())

    # .txt fallback
    return file_bytes.decode("utf-8", errors="ignore")


# ── Gemini prompt ──────────────────────────────────────────────
SYSTEM = """You are an expert ATS resume analyst and career coach.
Analyze the resume and return ONLY a valid JSON object — no markdown, no backticks, no explanation.

Required JSON shape:
{
  "ats_score": <integer 0-100>,
  "score_breakdown": {
    "keywords":       <integer 0-25>,
    "formatting":     <integer 0-25>,
    "content_quality":<integer 0-25>,
    "completeness":   <integer 0-25>
  },
  "summary": "<2-3 sentence overall assessment>",
  "strengths":      ["<strength>", ...],
  "weaknesses":     ["<weakness>", ...],
  "missing_skills": ["<skill>", ...],
  "improvements": [
    { "original": "<exact bullet or phrase>", "improved": "<better version>", "reason": "<why it's better>" },
    ...
  ]
}

Rules:
- ats_score = sum of score_breakdown values.
- Give at least 3 strengths, 3 weaknesses, 5 missing skills, 3 improvements.
- improvements must use action verbs and quantifiable metrics where possible.
"""

def call_gemini(resume_text: str, job_title: str | None) -> dict:
    prompt = SYSTEM
    if job_title:
        prompt += f"\n\nTarget role: {job_title}"
    prompt += f"\n\nResume:\n{resume_text[:12000]}"   # token safety

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.MULTILINE).strip()
    raw = re.sub(r"```$", "", raw, flags=re.MULTILINE).strip()

    return json.loads(raw)


# ── Route ──────────────────────────────────────────────────────
@app.post("/analyze", response_model=AnalysisResult)
async def analyze(
    file: UploadFile = File(...),
    job_title: Optional[str] = Form(None),
):
    file_bytes = await file.read()

    # Extract text
    try:
        resume_text = extract_text(file_bytes, file.filename or "resume.pdf")
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}")

    if len(resume_text.strip()) < 50:
        raise HTTPException(400, "Resume appears to be empty or unreadable.")

    # Gemini analysis
    try:
        data = call_gemini(resume_text, job_title)
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Gemini returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(500, f"Gemini API error: {e}")

    return AnalysisResult(**data)


@app.get("/health")
def health():
    return {"status": "ok"}
