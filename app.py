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

def generate_strong_jd(is_fresher, years_of_exp=0):
    """Generates a specific Job Description based on user profile."""
    if is_fresher:
        role_title = "Junior AI Application Engineer / Graduate Trainee"
        exp_req = "0-1 Years (Projects & Internships valued)"
        key_focus = "Translating academic projects into production code. Learning agility & basic stack proficiency."
        advice = "**Why this fits:** Your profile highlights potential. Focus on showcasing your GitHub projects."
    else:
        if years_of_exp < 3:
            role_title = "AI Application Engineer (Mid-Level)"
        else:
            role_title = "Senior Full-Stack Data Scientist"
        exp_req = f"{years_of_exp}+ Years of Industry Experience"
        key_focus = "System architecture, scalability, and deploying models to production under load."
        advice = "**Gap Analysis:** Ensure your 'Work Experience' section is detailed with metrics (e.g., 'Improved latency by 20%')."

    return role_title, exp_req, key_focus, advice

def get_ai_advice(resume_text, job_text, is_fresher, years_exp):
    profile_context = "Fresher/Student" if is_fresher else f"Experienced Professional ({years_exp} years)"
    
    prompt = f"""
    Role: Senior Technical Recruiter.
    Task: Perform a balanced review of the Resume against the Job Description.
    Candidate Profile: {profile_context}
    
    Output Format (Strictly follow this Markdown structure):
    
    ### ✅ Strong Matches
    1. **[Strength]**: [Brief explanation of why this fits the JD]
    2. **[Strength]**: [Brief explanation of why this fits the JD]
    3. **[Strength]**: [Brief explanation of why this fits the JD]
    
    ---
    
    ### ⚠️ Critical Gaps & Recommendations
    1. **[Weakness]**: [Actionable Fix]
    2. **[Weakness]**: [Actionable Fix]
    3. **[Weakness]**: [Actionable Fix]
    
    Context:
    Resume: {resume_text[:3000]}
    Job: {job_text[:3000]}
    """
    
    try:
        response = client.chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=600, stream=False)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def get_interview_questions(resume_text, job_text):
    prompt = f"""
    Role: Strict Recruiter.
    Task: Create 3 'Yes/No' interview questions based on MISSING skills in the resume.
    Constraint: Start with "Do you...", "Have you...".
    Context:
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

# --- 4. SIDEBAR (INPUTS) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=50)
    st.title("Resume Architect")
    
    st.divider()
    
    # --- NEW: CANDIDATE PROFILE SECTION ---
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
    
    # Process Files
    with st.spinner("🤖 Analyzing profile..."):
        raw_resume_text = extract_text_from_pdf(uploaded_file)
        cleaned_resume_text = clean_text(raw_resume_text)
        cleaned_job_desc = clean_text(job_description)
        
        # Calculate Score
        resume_embed = model.encode([cleaned_resume_text])
        job_embed = model.encode([cleaned_job_desc])
        similarity = float(cosine_similarity(resume_embed, job_embed)[0][0])
        match_score = round(similarity * 100, 2)

    # --- ROW 1: SCORE CARDS ---
    st.subheader("📊 Match Report")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Match Confidence", value=f"{match_score}%", delta="Base Score")
    with col2:
        if match_score >= 75:
            st.metric(label="Status", value="Strong Match", delta="High Priority")
        elif match_score >= 50:
            st.metric(label="Status", value="Moderate", delta="- Needs Optimization", delta_color="inverse")
        else:
            st.metric(label="Status", value="Weak Match", delta="- Critical Gaps", delta_color="inverse")
    with col3:
        word_count = len(cleaned_resume_text.split())
        st.metric(label="Resume Length", value=f"{word_count} Words", delta="Optimal Range" if 400 < word_count < 1000 else "Check Length", delta_color="off")
    style_metric_cards(background_color="#000000", border_left_color="#4F8BF9") # White bg for cleaner look

    st.divider()

    # --- ROW 2: DYNAMIC TARGET ROLE (NEW LOGIC) ---
    st.subheader("🎯 Targeted Role Analysis")
    
    is_fresher = (status == "Fresher / Student")
    role_title, exp_req, key_focus, role_advice = generate_strong_jd(is_fresher, years_exp)
    
    role_col1, role_col2 = st.columns([2, 1])
    
    with role_col1:
        st.markdown(f"""
        ### **{role_title}**
        **Expectation:** {exp_req}  
        **Key Focus:** {key_focus}
        """)
    
    with role_col2:
        st.info(role_advice)

    st.divider()

    # --- ROW 3: INTERACTIVE INTERVIEW ---
    st.subheader("🗣️ Smart Pre-Screening")
    
    if 'interview_qs' not in st.session_state:
        st.session_state['interview_qs'] = None

    col_btn, _ = st.columns([1, 4])
    if col_btn.button("Start Interview"):
        with st.spinner("Generating questions..."):
            qs = get_interview_questions(cleaned_resume_text, cleaned_job_desc)
            st.session_state['interview_qs'] = qs

    if st.session_state['interview_qs']:
        # 2-Column Layout
        q_col1, q_col2 = st.columns(2)
        
        with q_col1:
            st.markdown("#### 1️⃣ Experience Verification")
            # These are now clickable as requested
            st.checkbox(f"Confirms {status} Status?", value=True)
            if not is_fresher:
                st.checkbox(f"Has {years_exp}+ Years relevant experience?", value=True)
            else:
                st.checkbox("Has Internship / Project experience?", value=False)
                
        with q_col2:
            st.markdown("#### 2️⃣ Skill Gap Verification")
            for i, q in enumerate(st.session_state['interview_qs']):
                with st.expander(f"Q{i+1}: {q}", expanded=True):
                    ans = st.radio("Confirm:", ["Select...", "Yes", "No"], key=f"ai_q_{i}")
                    if ans == "Yes":
                        st.success("Tip: Ensure this is clearly listed in your Skills section.")
                    elif ans == "No":
                        st.error("Critical Gap detected.")

    st.divider()

    # --- ROW 4: DETAILED ADVICE ---
    st.subheader("💡 AI Consultant")
    
    with st.expander("✨ View Detailed Improvement Plan", expanded=False):
        if st.button("Analyze Gaps with Zephyr-7B"):
            with st.spinner("Consulting AI..."):
                advice = get_ai_advice(cleaned_resume_text, cleaned_job_desc, is_fresher, years_exp)
                st.markdown(advice)

else:
    st.info("👈 Please upload a Resume and Job Description in the sidebar to begin.")