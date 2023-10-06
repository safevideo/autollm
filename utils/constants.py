PINECONE_INDEX_NAME = "quickstart"

MODEL_COST = {
    'gpt-3.5-turbo': {'prompt': {'price': 0.0015, 'unit': 1000}, 'completion': {'price': 0.002, 'unit': 1000}},
    'gpt-4': {'prompt': {'price': 0.03, 'unit': 1000}, 'completion': {'price': 0.06, 'unit': 1000}},
    # ... Add other models as needed
}