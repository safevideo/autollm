DEFAULT_VECTORE_STORE_TYPE = 'pinecone'
DEFAULT_INDEX_NAME = 'quickstart'
DEFAULT_LLM_CLASS_NAME = 'OpenAI'

DEFAULT_RELATIVE_DOCS_PATH = 'docs'

DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 20
DEFAULT_CONTEXT_WINDOW = 4096

DEFAULT_MAX_TOKENS = 1024
DEFAULT_OPENAI_MODEL = 'gpt-3.5-turbo'
DEFAULT_PALM_MODEL = 'models/text-bison-001'
DEFAULT_ANYSCALE_MODEL = 'meta-llama/Llama-2-70b-chat-hf'

DEFAULT_ENABLE_TOKEN_COUNTING = True

# TODO: handle key error (for unsupported models)
# TODO: cost_calculation.py ayri module olusturulabilir: llm_class_name > model > prompt/completion > price/unit
MODEL_COST = {
    'gpt-3.5-turbo': {
        'prompt': {
            'price': 0.0015,
            'unit': 1000
        },
        'completion': {
            'price': 0.002,
            'unit': 1000
        }
    },
    'gpt-4': {
        'prompt': {
            'price': 0.03,
            'unit': 1000
        },
        'completion': {
            'price': 0.06,
            'unit': 1000
        }
    },
    # TODO: add more models (aws bedrock: https://aws.amazon.com/tr/bedrock/pricing/, anyscale: https://docs.endpoints.anyscale.com/pricing/#pricing, palm: https://cloud.google.com/vertex-ai/docs/generative-ai/pricing, azure-openai: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)
}
