# AutoLLM

## Base Package for Large Language Model Applications

## [![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/) ![Version 0.0.1](https://img.shields.io/badge/version-0.0.1-blue) ![GNU AGPL 3.0](https://img.shields.io/badge/license-AGPL_3.0-green)

## Introduction

Welcome to AutoLLM, the foundational package for building large language model applications. Designed for extensibility and high performance, AutoLLM serves as a base package for more specialized applications in querying, document processing, and more.

______________________________________________________________________

## Installation

AutoLLM is available as a Python package for Python>=3.8 environments. Install it using pip:

```bash
pip install autollm
```

______________________________________________________________________

## Features

### AutoLLM (Supports [80+ LLMs](https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json))

- Microsoft Azure - OpenAI example:

```python
from autollm import AutoLLM

## set ENV variables
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""

# Dynamically initialize a llama_index llm instance with the same AutoLLM api
llm = AutoLLM(model="azure/<your_deployment_name>")
```

- Google - VertexAI example:

```python
from autollm import AutoLLM

## set ENV variables
os.environ["VERTEXAI_PROJECT"] = "hardy-device-38811"  # Your Project ID`
os.environ["VERTEXAI_LOCATION"] = "us-central1"  # Your Location

# Dynamically initialize a llama_index llm instance with the same AutoLLM api
llm = AutoLLM(model="text-bison@001")
```

- AWS Bedrock - Claude v2 example:

```python
from autollm import AutoLLM

## set ENV variables
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

# Dynamically initialize a llama_index llm instance with the same AutoLLM api
llm = AutoLLM(model="anthropic.claude-v2")
```

### AutoVectorStore (Supports following VectorDBs: Pinecone, Qdrant, InMemory)

```python
from autollm import AutoVectorStore

# Dynamically initialize a VectorDB instance
vector_store = AutoVectorStore.from_defaults(
    vector_store_type="qdrant", index_name="quickstart", size=1536, distance="EUCLID"
)

vector_store = AutoVectorStore.from_defaults(
    vector_store_type="pinecone",
    index_name="quickstart",
    dimension=1536,
    metric_type="euclidean",
    pod_type="p1",
)

vector_store = AutoVectorStore.from_defaults(
    vector_store_type="inmemory", path_or_files="path/to/documents"
)
```

### AutoQueryEngine (Easy integration with any LLM + VectorStore + Query Template)

```python
```

### Automated Cost Calculation (Supports any OpenAI, Cohere, Anthropic and Llama2 model)

```
Embedding Token Usage: 7
LLM Prompt Token Usage: 1482
LLM Completion Token Usage: 47
LLM Total Token Cost: $0.002317
```

______________________________________________________________________

## FAQ

**Q: Can I use this for commercial projects?**

A: Yes, QuickLLM is licensed under GNU Affero General Public License (AGPL 3.0), which allows for commercial use under certain conditions. [Contact](#contact) us for more information.

______________________________________________________________________

## Roadmap

Our roadmap outlines upcoming features and integrations aimed at making QuickLLM the most extensible and powerful base package for large language model applications.

- [ ] **Bedrok Integrations**:

  - [ ] Claude2 support
  - [ ] Cohere support
  - [ ] LLAMA2 support

- [ ] **VectorDB Integrations**:

  - [ ] Chroma support
  - [ ] Weviate support
  - [ ] LanceDB support

______________________________________________________________________

## Contributing

We welcome contributions to QuickLLM! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information.

______________________________________________________________________

## License

QuickLLM is available under the [GNU Affero General Public License (AGPL 3.0)](LICENSE.txt).

______________________________________________________________________

## Contact

For more information, support, or questions, please contact:

- **Email**: [support@safevideo.ai](mailto:support@quickllm.com)
- **Website**: [SafeVideo](https://safevideo.ai/)
- **LinkedIn**: [SafeVideo AI](https://www.linkedin.com/company/safevideo/)

______________________________________________________________________
