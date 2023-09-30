from llama_index import VectorStoreIndex
from pathlib import Path
from multi_markdown_reader import MultiMarkdownReader

# Initialize the MultiMarkdownReader
markdown_reader = MultiMarkdownReader()

# Folder containing the markdown files
folder_path = Path('./data')

# Load the data
documents = markdown_reader.load_data_from_folder(folder_path)

# Print the number of documents loaded
print(f"Number of 'header-documents' loaded: {len(documents)}")

# Print the first document as a sample
if documents:
    print("Sample 'header-document':")
    print(documents[0].text)
documents[0].get_metadata_str()

# Create an index from the documents
index = VectorStoreIndex.from_documents(documents)

# Create a query engine
query_engine = index.as_query_engine()

# Sample user query
user_query = "Tell me about authors favourite food and exercise."

# Get response
response = query_engine.query(user_query)
print(response)