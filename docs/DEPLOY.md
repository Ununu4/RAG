# RAG API Deployment

## Local

```bash
# 1. Build chroma_db
python pre_processing/agent.py

# 2. Run API
pip install -r requirements.txt -r requirements-api.txt
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
# Build
docker build -t rag-api .

# Run (mount chroma_db)
docker run -p 8000:8000 -v "$(pwd)/chroma_db:/app/chroma_db" rag-api

# Groq (fast, free tier)
docker run -p 8000:8000 -v "$(pwd)/chroma_db:/app/chroma_db" \
  -e RAG_BACKEND=groq \
  -e GROQ_API_KEY=your_key \
  rag-api

# AWS Bedrock
docker run -p 8000:8000 -v "$(pwd)/chroma_db:/app/chroma_db" \
  -e RAG_BACKEND=aws \
  -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... -e AWS_REGION=us-east-1 \
  rag-api
```

## EC2 (t2.micro)

1. Launch t2.micro (free tier), Amazon Linux 2.
2. Install Docker: `sudo yum install -y docker && sudo service docker start && sudo usermod -aG docker ec2-user`
3. Clone repo, build chroma_db locally, scp `chroma_db/` to EC2.
4. Build and run:
   ```bash
   docker build -t rag-api .
   docker run -d -p 8000:8000 -v "$(pwd)/chroma_db:/app/chroma_db" --restart unless-stopped rag-api
   ```
5. Open port 8000 in security group.

## Endpoints

| Method | Path   | Description        |
|--------|--------|--------------------|
| GET    | /health | Liveness check     |
| GET    | /metrics | Placeholder       |
| POST   | /query | `{"query":"...","collection":null,"tier":"balanced"}` |
