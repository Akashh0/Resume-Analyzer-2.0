# Use official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your Python files
COPY . .

# Hugging Face Spaces exposes port 7860 by default!
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]