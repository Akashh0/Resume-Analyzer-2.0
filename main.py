from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io

# Import your actual AI Engine and Utils
from analyzer import ResumeAnalyzerAI
from utils import extract_text_from_pdf, clean_text

app = FastAPI(title="Resume Architect API")

app.add_middleware(
    CORSMiddleware,
    # Replace the URL below with your actual Netlify link (no trailing slash!)
    allow_origins=["https://analyseyourresume.netlify.app", "http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the AI Engine once when the server boots
print("Initializing AI Engine... (Loading Sentence Transformers)")
ai_engine = ResumeAnalyzerAI()
print("AI Engine Ready!")

@app.get("/")
async def health_check():
    """
    A lightweight endpoint for uptime bots to ping.
    """
    return {"status": "awake", "message": "The AI Engine is online."}

@app.post("/api/analyze")
async def analyze_resume(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...)
):
    try:
        # 1. Read the PDF file directly from the HTTP request into memory
        pdf_bytes = await resume_file.read()
        file_obj = io.BytesIO(pdf_bytes)
        
        # 2. Extract and clean text using your existing utils
        raw_text = extract_text_from_pdf(file_obj)
        cleaned_resume = clean_text(raw_text)
        cleaned_jd = clean_text(job_description)

        # 3. Run the AI Analyses
        match_score = ai_engine.calculate_match_score(cleaned_resume, cleaned_jd)
        role_info = ai_engine.analyze_target_role(cleaned_resume, status="Experienced Professional")
        audit_report = ai_engine.get_detailed_audit(cleaned_resume, cleaned_jd, is_fresher=False, years_exp=3)
        
        token_count = len(cleaned_resume.split())

        # 4. Return the massive payload of real data back to React
        return {
            "status": "success",
            "match_score": match_score,
            "tokens": token_count,
            "role_info": role_info,
            "audit_report": audit_report
        }
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)