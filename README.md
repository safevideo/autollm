# doc-qa-ai

## installation
```bash
conda create --name aidocs python=3.10
conda activate aidocs
pip install -r requirements.txt
```

## Sample .env file

```bash
OPENAI_API_KEY=<your-openai-api-key>
GIT_REPO_URL="https://github.com/ultralytics/ultralytics.git"
GIT_REPO_PATH="./ultralytics"
```

## Run FastAPI Server
```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

### Access the API Documentation
- Swagger UI: http://0.0.0.0:8000/docs

