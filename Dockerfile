# RAG QA API - minimal deployment
# Build chroma_db locally first: python pre_processing/agent.py
# Run: docker build -t rag-api . && docker run -p 8000:8000 -v $(pwd)/chroma_db:/app/chroma_db rag-api

FROM python:3.11-slim

WORKDIR /app

# Install deps (base + API + AWS)
COPY requirements.txt requirements-api.txt requirements-aws.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt -r requirements-aws.txt

COPY . .

# Chroma DB path (mount at runtime or copy pre-built chroma_db)
ENV RAG_CHROMA_PATH=/app/chroma_db
ENV RAG_BACKEND=local

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
