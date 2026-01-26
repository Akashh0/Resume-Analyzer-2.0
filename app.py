import streamlit as st
from utils import extract_text_from_pdf, clean_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from streamlit_extras.metric_cards import style_metric_cards

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="AI Resume Architect", layout="wide", page_icon="🚀")
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize Clients
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=HF_TOKEN)

# --- 2. CUSTOM CSS (MODERN LOOK) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .stCard { background-color: #f0f2f6; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #4F8BF9; color: white; border: none; }
</style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---

def generate_strong_jd(resume_text, status):
    """Dynamically analyzes the resume to suggest the perfect Job Title and Focus Area."""
    prompt = f"""
    Role: Senior Career Coach.
    Task: Analyze the resume and determine the BEST Job Title and Career Focus for this candidate.
    
    Candidate Status: {status}
    Resume Snippet:
    {resume_text[:2500]}
    
    Output strictly in this format (3 lines only):
    Title: [Recommended Job Title]
    Focus: [1 sentence on what they should emphasize, e.g., "Focus on backend scalability..."]
    Advice: [1 short tip, e.g., "Highlight your Python projects more."]
    """
    try:
        response = client.chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=150, stream=False)
        content = response.choices[0].message.content.strip()
        lines = content.split('\n')
        title, focus, advice = "AI Engineer", "General", "Focus on projects."
        for line in lines:
            if line.startswith("Title:"): title = line.replace("Title:", "").strip()
            elif line.startswith("Focus:"): focus = line.replace("Focus:", "").strip()
            elif line.startswith("Advice:"): advice = line.replace("Advice:", "").strip()
        return title, focus, advice
    except Exception as e:
        return "Software Engineer", "Development", "Check matches."

def get_ai_advice(resume_text, job_text, is_fresher, years_exp):
    profile_context = "Fresher/Student" if is_fresher else f"Experienced Professional ({years_exp} years)"
    prompt = f"""
    Role: Expert Resume Auditor.
    Candidate Profile: {profile_context}
    Task: Provide a deep-dive assessment.
    
    CRITICAL: Output strictly using these headers:
    
    ### 🎯 Strong Matches
    1. **[Strength 1]**: [Detail]
    2. **[Strength 2]**: [Detail]
    3. **[Strength 3]**: [Detail]
    4. **[Strength 4]**: [Detail]
    5. **[Strength 5]**: [Detail]
    
    ### ⚠️ Weakness in the Profile
    1. **[Weakness 1]**: [Detail]
    2. **[Weakness 2]**: [Detail]
    3. **[Weakness 3]**: [Detail]
    4. **[Weakness 4]**: [Detail]
    5. **[Weakness 5]**: [Detail]
    
    ### 📝 Context
    * [Detailed insight 1]
    * [Detailed insight 2]
    * [Detailed insight 3]
    * [Detailed insight 4]
    
    ### 💡 Recommendation
    * [Action Item 1]
    * [Action Item 2]
    * [Action Item 3]
    * [Action Item 4]
    
    Resume: {resume_text[:4000]}
    Job: {job_text[:4000]}
    """
    try:
        response = client.chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=1000, stream=False)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def generate_application_email(resume_text, job_text, type="Cover Letter"):
    """Generates a tailored Cover Letter or Cold Email."""
    
    # 1. Cleaner Prompt to stop hallucinating "Alternatively"
    prompt = f"""
    Role: Professional Career Coach.
    Task: Write a complete, professional {type} for this candidate.
    
    CRITICAL RULES:
    1. Do NOT output headers like "Alternatively" or multiple options. Write ONE single, polished draft.
    2. Start directly with "Dear Hiring Manager," or "Hi [Name],".
    3. Use the Resume to prove skills mentioned in the Job Description.
    4. Keep it human, confident, and specific.
    
    Resume Context:
    {resume_text[:3000]}
    
    Job Description:
    {job_text[:3000]}
    """
    
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}], 
            max_tokens=800,  # <-- INCREASED from 600 to 800
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def get_interview_questions(resume_text, job_text):
    prompt = f"""
    Role: Interviewer.
    Task: Create 3 strict YES/NO screening questions based on missing skills.
    
    RULES:
    1. Output ONLY the questions. No intro text.
    2. EVERY question must start with "Do you have..." or "Have you used...".
    3. Do NOT ask "How familiar..." or "Can you describe...".
    
    Bad Example: "How familiar are you with Python?"
    Good Example: "Do you have professional experience with Python?"
    
    Resume: {resume_text[:2000]}
    Job: {job_text[:2000]}
    """
    try:
        response = client.chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=200, stream=False)
        return [q.strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
    except:
        return []

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=50)
    st.title("Resume Architect")
    st.divider()
    
    st.subheader("1. Profile Settings")
    status = st.radio("Current Status:", ("Fresher / Student", "Experienced Professional"))
    years_exp = 0
    if status == "Experienced Professional":
        years_exp = st.number_input("Years of Experience:", min_value=1, max_value=30, step=1)
    
    st.divider()
    st.subheader("2. Job Details")
    job_description = st.text_area("Paste Job Description", height=200, placeholder="Paste JD here...")
    st.subheader("3. Candidate Resume")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# --- 5. MAIN DASHBOARD ---

if uploaded_file is not None and job_description:
    
    with st.spinner("🤖 Analyzing profile..."):
        raw_resume_text = extract_text_from_pdf(uploaded_file)
        cleaned_resume_text = clean_text(raw_resume_text)
        cleaned_job_desc = clean_text(job_description)
        resume_embed = model.encode([cleaned_resume_text])
        job_embed = model.encode([cleaned_job_desc])
        similarity = float(cosine_similarity(resume_embed, job_embed)[0][0])
        match_score = round(similarity * 100, 2)

    # --- ROW 1: SCORE ---
    st.subheader("📊 Match Report")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric(label="Match Confidence", value=f"{match_score}%", delta="Base Score")
    with col2: 
        if match_score >= 75: st.metric(label="Status", value="Strong Match", delta="High Priority")
        elif match_score >= 50: st.metric(label="Status", value="Moderate", delta="- Needs Optimization", delta_color="inverse")
        else: st.metric(label="Status", value="Weak Match", delta="- Critical Gaps", delta_color="inverse")
    with col3: st.metric(label="Word Count", value=f"{len(cleaned_resume_text.split())} Words")
    style_metric_cards(background_color="#000000", border_left_color="#4F8BF9") 

    st.divider()

    # --- ROW 2: TARGET ROLE ---
    st.subheader("🎯 Targeted Role Analysis")
    if 'dynamic_role' not in st.session_state: st.session_state['dynamic_role'] = None
    if st.session_state['dynamic_role'] is None:
        with st.spinner("Identifying role fit..."):
            role_title, key_focus, role_advice = generate_strong_jd(cleaned_resume_text, status)
            st.session_state['dynamic_role'] = (role_title, key_focus, role_advice)

    d_title, d_focus, d_advice = st.session_state['dynamic_role']
    r_col1, r_col2 = st.columns([2, 1])
    with r_col1: st.markdown(f"### **{d_title}**\n**Focus:** {d_focus}")
    with r_col2: st.info(f"💡 **Tip:** {d_advice}")

    st.divider()

    # --- ROW 3: INTERVIEW ---
    st.subheader("🗣️ Smart Pre-Screening")
    if 'interview_qs' not in st.session_state: st.session_state['interview_qs'] = None
    if st.button("Start Interview"):
        with st.spinner("Generating questions..."):
            st.session_state['interview_qs'] = get_interview_questions(cleaned_resume_text, cleaned_job_desc)

    if st.session_state['interview_qs']:
        q_col1, q_col2 = st.columns(2)
        with q_col1:
            st.markdown("#### 1️⃣ Experience Verification")
            is_fresher = (status == "Fresher / Student")
            st.checkbox(f"Confirms {status} Status?", value=True)
            if not is_fresher: st.checkbox(f"Has {years_exp}+ Years experience?", value=True)
            else: st.checkbox("Has Internship / Project experience?", value=False)
        with q_col2:
            st.markdown("#### 2️⃣ Skill Gap Verification")
            for i, q in enumerate(st.session_state['interview_qs']):
                with st.expander(f"Q{i+1}: {q}", expanded=True):
                    ans = st.radio("Confirm:", ["Select...", "Yes", "No"], key=f"ai_q_{i}")
                    if ans == "Yes": st.success("Tip: Add this to your Skills section.")
                    elif ans == "No": st.error("Gap detected.")

    st.divider()

    # --- ROW 4: ADVICE ---
    st.subheader("💡 AI Consultant")
    with st.expander("✨ View Detailed Improvement Plan", expanded=False):
        if st.button("Analyze Gaps with Zephyr-7B"):
            with st.spinner("Consulting AI..."):
                is_fresher = (status == "Fresher / Student")
                advice = get_ai_advice(cleaned_resume_text, cleaned_job_desc, is_fresher, years_exp)
                st.markdown(advice)
    
    st.divider()

    # --- ROW 5: SMART APPLY (NEW FEATURE) ---
    st.subheader("✉️ Smart Apply")
    st.caption("Generate a tailored Cover Letter or Cold Email based on this analysis.")
    
    col_type, col_gen = st.columns([1, 3])
    with col_type:
        app_type = st.radio("Application Type:", ["Cover Letter", "Cold Email to Hiring Manager"])
    
    with col_gen:
        if st.button(f"Draft {app_type}"):
            with st.spinner(f"Drafting your {app_type}..."):
                email_draft = generate_application_email(cleaned_resume_text, cleaned_job_desc, app_type)
                st.text_area("Copy your draft:", value=email_draft, height=300)
                st.success("✅ Draft generated! Review and personalize before sending.")

else:
    if 'dynamic_role' in st.session_state: del st.session_state['dynamic_role']
    st.info("👈 Please upload a Resume and Job Description in the sidebar to begin.")