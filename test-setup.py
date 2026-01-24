import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

print("✅ Libraries imported successfully!")

# Check for API Key (It will be None if you haven't added it to .env yet, that's okay)
key = os.getenv("HF_TOKEN")
if key:
    print("✅ API Key found.")
else:
    print("⚠️ API Key not found in .env yet.")

# Test Loading the Local Model (This triggers the download)
print("⬇️  Downloading/Loading embedding model... (This runs once)")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded successfully!")