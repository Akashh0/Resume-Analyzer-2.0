import streamlit as st
from utils import extract_text_from_pdf, clean_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

def get_ai_advice(resume_text, job_text):
    """
    Sends the resume and job description to the LLM for specific feedback.
    """
    # Truncate text to avoid token limits (approx 3000 chars each is safe for free tier)
    resume_snippet = resume_text[:3000]
    job_snippet = job_text[:3000]
    
    prompt = f"""
    You are an expert technical recruiter. I will give you a Resume and a Job Description. 
    Your task is to identify 3 specific GAPS or WEAKNESSES in the resume compared to the job description.
    
    Resume Content:
    {resume_snippet}
    
    Job Description:
    {job_snippet}
    
    Output strictly in this format:
    1. [Gap/Weakness]: [Brief advice on how to fix it]
    2. [Gap/Weakness]: [Brief advice on how to fix it]
    3. [Gap/Weakness]: [Brief advice on how to fix it]
    """
    
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to AI: {e}"

# Load API Key
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize the AI Client
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=HF_TOKEN)
# Page Configuration
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("📄 AI Resume Analyzer")
st.markdown("Upload your resume and the job description to get a match score and AI feedback.")

# --- 1. Load Local Model (Cached) ---
# @st.cache_resource keeps the model in memory so it doesn't reload on every click.
# This makes the app fast after the first load.
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- 2. The Inputs ---
col1, col2 = st.columns(2)

with col1:
    st.header("1. Upload Resume")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

with col2:
    st.header("2. Job Description")
    job_description = st.text_area("Paste the Job Description here", height=300)

# --- 3. The Processing Logic ---
if uploaded_file is not None and job_description:
    # A. Extract & Clean
    with st.spinner("Reading PDF..."):
        raw_resume_text = extract_text_from_pdf(uploaded_file)
        cleaned_resume_text = clean_text(raw_resume_text)
        cleaned_job_desc = clean_text(job_description)

    # B. The Match Score (Vector Embedding)
    with st.spinner("Calculating Match Score..."):
        # Convert text to numbers (Embeddings)
        resume_embedding = model.encode([cleaned_resume_text])
        job_embedding = model.encode([cleaned_job_desc])

        # Calculate Cosine Similarity
        # FIX: We wrap the result in float() to convert it from numpy.float32 to python float
        similarity_score = float(cosine_similarity(resume_embedding, job_embedding)[0][0])
        
        # Convert to percentage
        match_percentage = round(similarity_score * 100, 2)

    # --- 4. Display Results ---
    st.divider()
    st.subheader("📊 Analysis Results")
    
    # Create a nice metric card
    col_score, col_status = st.columns([1, 2])
    
    with col_score:
        st.metric(label="Match Score", value=f"{match_percentage}%")
        
        # Visual Progress Bar
        if match_percentage >= 75:
            st.progress(match_percentage / 100, text="✅ Great Match!")
        elif match_percentage >= 50:
            st.progress(match_percentage / 100, text="⚠️ Good, but needs work.")
        else:
            st.progress(match_percentage / 100, text="❌ Low match.")

    with col_status:
        st.info("This score is calculated using **Cosine Similarity** on vector embeddings. It measures how conceptually similar your resume content is to the job description.")

    # Debugging (Optional)
    with st.expander("View Extracted Text"):
        st.text(cleaned_resume_text[:500] + "...") # Show first 500 chars

    st.divider()
    st.divider()
    st.subheader("💡 AI Improvement Advice")
    
    # Only run the expensive API call if the user clicks a button (Saves calls)
    if st.button("Generate AI Feedback"):
        with st.spinner("Analyzing your resume against the job description... (This may take 10-20 seconds)"):
            ai_advice = get_ai_advice(cleaned_resume_text, cleaned_job_desc)
            st.markdown(ai_advice)
            
    else:
        st.info("Click the button above to ask the AI for specific improvement tips.")