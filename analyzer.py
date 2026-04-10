import os
import re
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ResumeAnalyzerAI:
    """
    Core AI Engine for Resume Analysis, Scoring, and Generation.
    Independent of any UI framework.
    """
    def __init__(self, hf_token: str = None, llm_model: str = "Qwen/Qwen2.5-7B-Instruct"):
        # Load environment variables if token not explicitly provided
        if not hf_token:
            load_dotenv()
            hf_token = os.getenv("HF_TOKEN")
            
        if not hf_token:
            raise ValueError("Hugging Face API token is required. Set HF_TOKEN in .env or pass it directly.")

        # Initialize LLM Client
        self.llm_client = InferenceClient(model=llm_model, token=hf_token)
        
        # Initialize Local Embedding Model (Loads once into memory)
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    def calculate_match_score(self, resume_text: str, job_text: str) -> float:
        """
        Calculates the semantic match percentage using Vector Embeddings and Cosine Similarity.
        """
        resume_embed = self.embed_model.encode([resume_text])
        job_embed = self.embed_model.encode([job_text])
        
        similarity = float(cosine_similarity(resume_embed, job_embed)[0][0])
        match_score = round(similarity * 100, 2)
        
        # Ensure bounds
        return max(0.0, min(match_score, 100.0))

    def analyze_target_role(self, resume_text: str, status: str) -> Dict[str, str]:
        """
        Dynamically analyzes the resume to suggest the perfect Job Title and Focus Area.
        Returns a dictionary for easy backend parsing.
        """
        prompt = f"""
        Role: Senior Career Coach.
        Task: Analyze the resume and determine the BEST Job Title and Career Focus for this candidate.
        
        Candidate Status: {status}
        
        Resume Snippet:
        {resume_text[:2500]}
        
        CRITICAL RULE: Output MUST be exactly 3 lines in the following format. Do not add intro or outro text.
        Title: [Recommended Job Title]
        Focus: [1 sentence on what they should emphasize, e.g., "Focus on backend scalability..."]
        Advice: [1 short actionable tip, e.g., "Highlight your Python projects more."]
        """
        
        try:
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}], 
                max_tokens=150, 
                stream=False
            )
            content = response.choices[0].message.content.strip()
            
            # Structured Parsing
            result = {
                "title": "AI Application Engineer", # Fallbacks
                "focus": "Building robust systems.",
                "advice": "Review match criteria."
            }
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith("Title:"): result["title"] = line.replace("Title:", "").strip()
                elif line.startswith("Focus:"): result["focus"] = line.replace("Focus:", "").strip()
                elif line.startswith("Advice:"): result["advice"] = line.replace("Advice:", "").strip()
                
            return result
        except Exception as e:
            return {"error": str(e), "title": "Error", "focus": "Error processing", "advice": "Please try again."}

    def get_detailed_audit(self, resume_text: str, job_text: str, is_fresher: bool, years_exp: int) -> str:
        """
        Provides a deep-dive assessment (Strengths, Weaknesses, Context, Recommendations).
        """
        profile_context = "Fresher/Student" if is_fresher else f"Experienced Professional ({years_exp} years)"
        
        # YOUR FULL, UNDEGRADED PROMPT RESTORED
        prompt = f"""
        Role: Expert Resume Auditor & Technical Recruiter.
        Candidate Profile: {profile_context}
        Task: Provide a deep-dive assessment based strictly on the provided Resume and Job Description.
        
        CRITICAL INSTRUCTION: Output strictly using the exact Markdown headers below. Ensure every point is detailed, highly specific, and actionable. Avoid generic fluff.
        DO NOT output conversational filler, metadata, or timestamps. Begin immediately with the first header.
        You MUST format the start of each bullet point exactly as shown with bold text (e.g., 1. **[Specific Strength]**:).
        
        ### 🎯 Strong Matches
        1. **[Specific Strength 1]**: [Detail]
        2. **[Specific Strength 2]**: [Detail]
        3. **[Specific Strength 3]**: [Detail]
        4. **[Specific Strength 4]**: [Detail]
        5. **[Specific Strength 5]**: [Detail]
        
        ### ⚠️ Weaknesses & Gaps
        1. **[Specific Weakness 1]**: [Detail]
        2. **[Specific Weakness 2]**: [Detail]
        3. **[Specific Weakness 3]**: [Detail]
        4. **[Specific Weakness 4]**: [Detail]
        5. **[Specific Weakness 5]**: [Detail]
        
        ### 📝 Context & Market Fit
        * **[Market Fit]**: [Detailed insight on overall fit relative to market standard]
        * **[Career Trajectory]**: [Observation on candidate's career trajectory vs JD requirements]
        * **[Domain Alignment]**: [Analysis of specific domain/industry alignment]
        * **[Presentation]**: [Comment on the formatting, tone, and professionalism]
        
        ### 💡 Strategic Recommendations
        * **[Action Item 1]**: [Specific technical skill or tool to learn/add]
        * **[Action Item 2]**: [Specific resume section to rewrite with impact metrics]
        * **[Action Item 3]**: [Strategy for answering interview questions about the gaps]
        * **[Action Item 4]**: [Final strategic positioning advice]
        
        Resume: {resume_text[:4000]}
        
        Job Description: {job_text[:4000]}
        """
        try:
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}], 
                max_tokens=1500, # Increased to handle your full 5-point structure
                temperature=0.4, # Prevents looping while keeping formatting strict
                top_p=0.9,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating audit: {e}"

    def generate_screening_questions(self, resume_text: str, job_text: str) -> List[str]:
        """
        Generates 3 strict Yes/No screening questions based on missing requirements.
        """
        prompt = f"""
        Role: Strict Technical Recruiter.
        Task: Identify 3 critical skills/requirements in the Job Description that are MISSING from the Resume. 
        Create 3 strict YES/NO screening questions to ask the candidate about these missing skills.
        
        STRICT RULES:
        1. Output ONLY the 3 questions, separated by newlines. No intro, no numbering, no outro text.
        2. EVERY question MUST begin with "Do you have...", "Have you used...", or "Did you...".
        3. DO NOT ask open-ended questions like "How familiar are you...", "Can you describe...", or "What is your experience...".
        
        Resume Context: {resume_text[:2500]}
        
        Job Context: {job_text[:2500]}
        """
        try:
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}], 
                max_tokens=200, 
                stream=False
            )
            
            raw_text = response.choices[0].message.content.strip()
            questions = [q.strip() for q in raw_text.split('\n') if q.strip()]
            clean_questions = [re.sub(r'^\d+\.\s*', '', q) for q in questions]
            return clean_questions[:3] 
        except Exception as e:
            return [f"Error generating questions: {e}"]

    def draft_application_material(self, resume_text: str, job_text: str, doc_type: str = "Cover Letter") -> str:
        """
        Drafts a tailored Cover Letter or Cold Email to the Hiring Manager.
        """
        prompt = f"""
        Role: Expert Career Strategist & Copywriter.
        Task: Write a complete, highly persuasive {doc_type} for this candidate applying to this specific job.
        
        CRITICAL RULES:
        1. Write ONE single, polished draft from start to finish.
        2. DO NOT include meta-text, multiple options, or headers like "Alternatively:".
        3. DO NOT use generic placeholders if the info is available in the resume. 
        4. If it is a "Cover Letter": Use standard formal structure (Opening, Value Proposition, Closing).
        5. If it is a "Cold Email to Hiring Manager": Keep it under 150 words, punchy, and focused on immediate value.
        6. Start directly with the greeting: "Dear Hiring Manager," or "Hi [Company Name] Team,".
        
        Resume Context (Extract candidate achievements from here):
        {resume_text[:3000]}
        
        Job Description (Align the narrative to these needs):
        {job_text[:3000]}
        """
        
        try:
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}], 
                max_tokens=800, 
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating application draft: {e}"

if __name__ == "__main__":
    analyzer = ResumeAnalyzerAI()
    
    sample_resume = "Full Stack Developer with 3 years of React and Django experience. Built AI apps."
    sample_jd = "Looking for a Senior Python Developer with Django and Machine Learning expertise."
    
    print("\n--- 1. Match Score ---")
    score = analyzer.calculate_match_score(sample_resume, sample_jd)
    print(f"Score: {score}%")
    
    print("\n--- 2. Role Analysis ---")
    role_info = analyzer.analyze_target_role(sample_resume, "Experienced Professional")
    print(role_info)
    
    print("\n--- 3. Screening Questions ---")
    questions = analyzer.generate_screening_questions(sample_resume, sample_jd)
    for q in questions:
        print(f"- {q}")