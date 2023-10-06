PINECONE_INDEX_NAME = "quickstart"

DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 20
DEFAULT_CONTEXT_WINDOW = 4096

DEFAULT_LLM_BACKEND = "OPENAI"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_OPENAI_MODEL_NAME = "gpt-3.5-turbo"
DEFAULT_PALM_MODEL_NAME = "models/text-bison-001"
DEFAULT_ANYSCALE_MODEL_NAME = "meta-llama/Llama-2-70b-chat-hf"

DEFAULT_ENABLE_TOKEN_COUNTING = True

MODEL_COST = {
    'gpt-3.5-turbo': {'prompt': {'price': 0.0015, 'unit': 1000}, 'completion': {'price': 0.002, 'unit': 1000}},
    'gpt-4': {'prompt': {'price': 0.03, 'unit': 1000}, 'completion': {'price': 0.06, 'unit': 1000}},
    # ... Add other models as needed
}