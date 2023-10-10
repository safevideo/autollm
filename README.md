# doc-qa-ai

## installation
```bash
conda create --name aidocs python=3.10
conda activate aidocs
pip install -r requirements.txt
```

## Sample .env file

```.env
OPENAI_API_KEY=<your-openai-api-key>

GIT_REPO_URL="https://github.com/ultralytics/ultralytics.git"
GIT_REPO_PATH="./ultralytics"

MAX_TOKENS = 1024
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 20
CONTEXT_WINDOW = 4096
SIMILARITY_TOP_K = 4
```

## Run FastAPI Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Access the API Documentation
- Swagger UI: http://0.0.0.0:8000/docs

