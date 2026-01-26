# 🚀 Resume Architect AI

**A Next-Gen AI Resume Analyzer & Optimizer built with Streamlit, Hugging Face, and Vector Embeddings.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![AI Model](https://img.shields.io/badge/AI-Zephyr--7B-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📖 Overview

**Resume Architect AI** is a smart application designed to bridge the gap between job seekers and Applicant Tracking Systems (ATS). Unlike simple keyword matchers, this tool uses **Semantic Analysis** to understand the *meaning* of a resume compared to a job description.

It features a hybrid AI architecture:
* **Local Embeddings (`all-MiniLM-L6-v2`)**: For fast, privacy-first similarity scoring.
* **Cloud LLM (`Zephyr-7B-Beta`)**: For generating deep qualitative insights, interview questions, and cover letters via the Hugging Face Inference API.

## ✨ Key Features

### 1. 📊 Semantic Match Scoring
Calculates a percentage score based on **Cosine Similarity** between the resume and job description vectors. This detects fit even if exact keywords are missing (e.g., "ML" vs. "Machine Learning").

### 2. 🎯 Dynamic Role Analysis
The AI autonomously analyzes the candidate's experience level and the job description to suggest the **ideal Job Title** and **Career Focus** area tailored to that specific application.

### 3. 🗣️ Smart Pre-Screening Interview
Simulates a recruiter phone screen by generating **3 context-aware "Yes/No" screening questions** based specifically on skills missing from the resume. This helps users identify critical gaps before applying.

### 4. 💡 Deep-Dive AI Audit
Provides a structured report including:
* ✅ **Strong Matches:** What stands out.
* ⚠️ **Critical Weaknesses:** Specific missing skills or formatting issues.
* 📝 **Contextual Insights:** Career trajectory analysis.
* 💡 **Strategic Recommendations:** Actionable steps to improve the profile.

### 5. ✉️ "Smart Apply" Generator
Instantly drafts a **tailored Cover Letter** or a **Cold Email to a Hiring Manager** that specifically addresses the job requirements using the candidate's actual project experience.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (with `streamlit-extras` for custom UI components).
* **PDF Processing:** `pdfplumber` for high-fidelity text extraction.
* **Vector Embeddings:** `sentence-transformers` (Hugging Face).
* **LLM Integration:** Hugging Face Inference API (`HuggingFaceH4/zephyr-7b-beta`).
* **Mathematics:** `scikit-learn` (Cosine Similarity) and `NumPy`.

---

## 🚀 Installation & Setup

Follow these steps to run the project locally.

### Prerequisites
* Python 3.9 or higher installed.
* A Hugging Face Account (for the free API token).

### Step 1: Clone the Repository
```bash
git clone (https://github.com/Akashh0/Resume-Analyzer-2.0.git)
cd resume-architect-ai
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv venv
* On Windows:
venv\Scripts\activate
* On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys
* Ensure to create a file named .env in the root directory and add your Hugging Face token:
```bash
HF_TOKEN=hf_YourActualTokenHere
```

### Step 5: Run the app
```bash
streamlit run app.py
```
* The app will open in your default browser at http://localhost:8501

## 📂 Project Structure
```bash
resume-architect-ai/
├── app.py                # Main application logic & UI
├── utils.py              # Helper functions (PDF extraction, text cleaning)
├── requirements.txt      # Project dependencies
├── .env                  # API keys (Not uploaded to GitHub)
├── .gitignore            # Files to ignore (venv, .env, etc.)
└── README.md             # Project documentation
```

## Future Improvements

* 1. GitHub Integration: Auto-fetch skills from a user's pinned repositories. 

* 2. Resume Rewriter: An agent that auto-suggests sentence improvements.

* 3. Multi-Page Support: Ability to analyze multiple resumes against one JD.

## Contributions

* Contributions are welcome! Please fork the repository and submit a Pull Request.