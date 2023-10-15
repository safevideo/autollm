# AutoLLM

## Elevate Your Large Language Model Applications

## [![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/) ![Version 0.0.1](https://img.shields.io/badge/version-0.0.1-blue) ![GNU AGPL 3.0](https://img.shields.io/badge/license-AGPL_3.0-green)

## Introduction

Welcome to AutoLLM, the definitive toolkit for deploying, managing, and scaling Large Language Model (LLM) applications. Built for high performance and maximum flexibility, AutoLLM provides seamless integration with multiple LLM providers, vector databases, and service contexts.

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

Instantly initialize a VectorDB instance with same API

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
    vector_store_type="in_memory", path_or_files="path/to/documents"
)
```

### AutoQueryEngine (Creates a query engine pipeline in a single line of code)

Create robust query engine pipelines with automatic cost logging. Supports fine-grained control for advanced use-cases.

#### Basic Usage:

```python
from autollm import AutoQueryEngine

# Initialize a query engine with existing vector store and service context
vector_store = AutoVectorStore.from_defaults(
    vector_store_type="in_memory", input_files="path/to/documents"
)
service_context = AutoServiceContext.from_defaults(enable_cost_calculator=True)
query_engine = AutoQueryEngine.from_instances(vector_store, service_context)
```

```python
# Initialize a query engine with default parameters
query_engine = AutoQueryEngine.from_parameters()

# Ask a question
response = query_engine.query("Why is SafeVideo AI open sourcing this project?")

print(response.response)
```

```
>> Because they are cool!
```

#### Advanced Usage:

For fine-grained control, you can initialize the `AutoQueryEngine` by explicitly passing parameters for the LLM, Vector Store, and Service Context.

```python
from autollm import AutoQueryEngine

# Initialize the query engine with explicit parameters
query_engine = AutoQueryEngine.from_parameters(
    system_prompt="Your System Prompt",
    query_wrapper_prompt="Your Query Wrapper Prompt",
    enable_cost_calculator=True,
    llm_params={"model": "gpt-3.5-turbo"},
    vector_store_params={"vector_store_type": "qdrant", "index_name": "quickstart"},
    service_context_params={"chunk_size": 1024},
    query_engine_params={"similarity_top_k": 10},
)

response = query_engine.query("Why is SafeVideo AI awesome?")

print(response.response)

>> Because they redefine the movie experience by AI!
```

### Automated Cost Calculation (Supports [80+ LLMs](https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json))

Keep track of your LLM token usage and costs in real-time.

```python
from autollm import AutoServiceContext

service_context = AutoServiceContext(enable_cost_calculation=True)

# Example calculation verbose output
"""
Embedding Token Usage: 7
LLM Prompt Token Usage: 1482
LLM Completion Token Usage: 47
LLM Total Token Cost: $0.002317
"""
```

______________________________________________________________________

## FAQ

**Q: Can I use this for commercial projects?**

A: Yes, QuickLLM is licensed under GNU Affero General Public License (AGPL 3.0), which allows for commercial use under certain conditions. [Contact](#contact) us for more information.

______________________________________________________________________

## Roadmap

Our roadmap outlines upcoming features and integrations aimed at making QuickLLM the most extensible and powerful base package for large language model applications.

- ~~\[x\] **Bedrok Integrations**:~~

  - ~~\[x\] Claude2 support~~
  - ~~\[x\] Cohere support~~
  - ~~\[x\] LLAMA2 support~~

- [ ] **VectorDB Integrations**:

  - [ ] Chroma support
  - [ ] Weviate support
  - [ ] LanceDB support

- [ ] **Pipelines**:

  - [ ] In memory PDF QA pipeline
  - [ ] DB-based documentation QA pipeline

- [ ] **FastAPI Integration**:

  - [ ] FastAPI integration for Pipelines

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
