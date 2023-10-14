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

### AutoLLM (Supports all llama-index LLMs + AWS Bedrock LLMs)

```python
from autollm import AutoLLM
from llama_index.llms.base import LLM

# Dynamically initialize a llama_index llm instance
llm = AutoLLM(llm_class_name="OpenAI", model="gpt4")
llm = AutoLLM(llm_class_name="PaLM", model="models/text-bison-001")
llm = AutoLLM(llm_class_name="Bedrock", model_id="anthropic.claude-v2")
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

### Automated Cost Calculation (Supports OpenAI)

```bash
Embedding Token Usage: 4
LLM Prompt Token Usage: 1289
LLM Completion Token Usage: 73
Total cost for this query: {total_cost} USD
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
