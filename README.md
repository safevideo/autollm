<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="100%"
        src="https://github-production-user-asset-6210df.s3.amazonaws.com/44926076/278394163-6af17028-b7cc-4511-b677-7031ed31ffbc.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20231027%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231027T124157Z&X-Amz-Expires=300&X-Amz-Signature=c9618430d2631db3f45551ad062d9b940b91723636c23dcc0639e659f3b425b6&X-Amz-SignedHeaders=host&actor_id=44926076&key_id=0&repo_id=694565231"
      >
    </a>
  </p>

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
![Version 0.0.1](https://img.shields.io/badge/version-0.0.1-blue)
![GNU AGPL 3.0](https://img.shields.io/badge/license-AGPL_3.0-green)

</div>

## 🤔 why autollm?

**Simplify. Unify. Amplify.** Integrate any Large Language Model (LLM) or Vector Database with just one line of code.

| Feature                         | AutoLLM | LangChain | LlamaIndex | LiteLLM |
| ------------------------------- | :-----: | :-------: | :--------: | :-----: |
| **80+ LLMs**                    |   ✔️    |    ✔️     |     ✔️     |   ✔️    |
| **Unified API**                 |   ✔️    |     ❌     |     ❌      |   ✔️    |
| **20+ Vector Databases**        |   ✔️    |    ✔️     |     ✔️     |    ❌    |
| **Cost Calculation (80+ LLMs)** |   ✔️    |     ❌     |     ❌      |   ✔️    |
| **1-Line FastAPI**              |   ✔️    |     ❌     |     ❌      |    ❌    |
| **1-Line RAG LLM Engine**       |   ✔️    |     ❌     |     ❌      |    ❌    |

______________________________________________________________________

## 📦 Installation

Easily install autollm package with pip in [**Python>=3.8**](https://www.python.org/downloads/) environment.

```bash
pip install autollm
```

______________________________________________________________________

## 🌟 Features

### 📚 Supports [80+ LLMs](https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json)

<details>
    <summary>👉 Microsoft Azure - OpenAI example:</summary>

```python
from autollm import AutoLLM

os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""

# Dynamically initialize a llama_index llm instance with the same AutoLLM api
llm = AutoLLM(model="azure/<your_deployment_name>")
```

</details>

<details>
    <summary>👉 Google - VertexAI example</summary>

```python
from autollm import AutoLLM

## set ENV variables
os.environ["VERTEXAI_PROJECT"] = "hardy-device-38811"  # Your Project ID`
os.environ["VERTEXAI_LOCATION"] = "us-central1"  # Your Location

# Dynamically initialize a llama_index llm instance with the same AutoLLM api
llm = AutoLLM(model="text-bison@001")
```

</details>

<details>
<summary>👉 AWS Bedrock - Claude v2 example</summary>

```python
from autollm import AutoLLM

## set ENV variables
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

# Dynamically initialize a llama_index llm instance with the same AutoLLM interface
llm = AutoLLM(model="anthropic.claude-v2")
```

</details>

### 📈 Supports [20+ VectorDBs](https://docs.llamaindex.ai/en/stable/core_modules/data_modules/storage/vector_stores.html#vector-store-options-feature-support)

🌟 **Pro Tip**: AutoLLM defaults to LanceDB if no vector store is specified.

<details>
    <summary>👉 Default - LanceDB example</summary>

```python
from autollm import AutoVectorStoreIndex

vector_store_index = AutoVectorStoreIndex.from_defaults()
```

</details>

### 💰 Automated Cost Calculation for [80+ LLMs](https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json)

<details>
    <summary>👉 Keep track of your LLM costs</summary>

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

</details>

### 🚀 Create FastAPI App in 1-Line

<details>
    <summary>👉 Example</summary>

```python
from autollm import create_web_app

app = create_web_app(config_path, env_path)
```

Here, `config` and `env` should be replaced by your configuration and environment file paths.

After creating your FastAPI app, run the following command in your terminal to get it up and running:

```bash
uvicorn main:app
```

</details>

## 🎯 quickstart

### 🔥 create a query engine in one line

<details>
    <summary>👉 basic usage </summary>

```python
query_engine = AutoQueryEngine.from_parameters()

response = query_engine.query("Why is SafeVideo AI open sourcing this project?")

print(response.response)
```

```
>> Because they are cool!
```

</details>

<details>
    <summary>👉 advanced usage </summary>

```python
from autollm import AutoQueryEngine

import qdrant_client

# Initialize the query engine with explicit parameters
query_engine = AutoQueryEngine.from_parameters(
    system_prompt="You are an expert qa assistant. Provide accurate and detailed answers to queries",
    query_wrapper_prompt="The document information is the following: {context_str} | Using the document information and mostly relying on it,
answer the query. | Query {query_str} | Answer:",
    enable_cost_calculator=True,
    llm_params={"model": "gpt-3.5-turbo"},
    vector_store_params={"vector_store_type": "QdrantVectorStore", "client": qdrant_client.QdrantClient(
    url="http://<host>:<port>"
    api_key="<qdrant-api-key>",
), "collection_name": "quickstart"},
    service_context_params={"chunk_size": 1024},
    query_engine_params={"similarity_top_k": 10},
)

response = query_engine.query("Why is SafeVideo AI awesome?")

print(response.response)
```

```
>> Because they redefine the movie experience by AI!
```

</details>

### 📚 Document Reading

#### 🌐 Create documents for a VectorDB from GitHub repo in one line!

```python
from autollm.utils.document_reading import read_github_repo_as_documents

git_repo_url = "https://github.com/safevideo.git"
relative_folder_path = Path("docs/")

documents = read_github_repo_as_documents(git_repo_url, relative_folder_path)
```

#### 📂 Add Local Files into VectorDB in One Line (Supports [10+ File Types](https://github.com/run-llama/llama_index/blob/main/llama_index/readers/file/base.py#L19-L34))

```python
from autollm.utils.document_reading import read_local_files_as_documents

documents = read_local_files_as_documents(input_dir="tmp/docs")
```

______________________________________________________________________

## 🔄 Smooth Migration from LlamaIndex

Switching from LlamaIndex? We've got you covered.

<details close>
<summary> Easy Migration

```python
from autollm import AutoQueryEngine
from llama_index import StorageContext, ServiceContext, VectorStoreIndex
from llama_index.vectorstores import LanceDBVectorStore

vector_store = LanceDBVectorStore(uri="/tmp/lancedb")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents=documents)
service_context = ServiceContext.from_defaults()

query_engine = AutoQueryEngine.from_instance(index, service_context)
```

</details>

## ❓ FAQ

**Q: Can I use this for commercial projects?**

A: Yes, AutoLLM is licensed under GNU Affero General Public License (AGPL 3.0), which allows for commercial use under certain conditions. [Contact](#contact) us for more information.

______________________________________________________________________

## Roadmap

Our roadmap outlines upcoming features and integrations to make AutoLLM the most extensible and powerful base package for large language model applications.

- [ ] **Budget based email notification feature**

- [ ] **Add evaluation metrics for LLMs**:

- [ ] **Add unit tests for online vectorDB integrations**:

- [ ] **Add example code snippet to Readme on how to integrate llama-hub readers**:

______________________________________________________________________

## 📜 License

AutoLLM is available under the [GNU Affero General Public License (AGPL 3.0)](LICENSE).

______________________________________________________________________

## 📞 Contact

For more information, support, or questions, please contact:

- **Email**: [support@safevideo.ai](mailto:support@safevideo.ai)
- **Website**: [SafeVideo](https://safevideo.ai/)
- **LinkedIn**: [SafeVideo AI](https://www.linkedin.com/company/safevideo/)

______________________________________________________________________

## 🌟 Contributing

**Love AutoLLM? Star the repo or contribute and help us make it even better!** See our [contributing guidelines](CONTRIBUTING.md) for more information.

<p align="center">
    <a href="https://github.com/safevideo/autollm/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=safevideo/autollm" />
    </a>
</p>

<div align="center">
      <a href="https://www.linkedin.com/company/safevideo/">
          <img
            src="https://github-production-user-asset-6210df.s3.amazonaws.com/44926076/278662107-ee33237c-77b6-4760-91f6-a3aff9b75e56.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20231027%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231027T134854Z&X-Amz-Expires=300&X-Amz-Signature=3c0fcdc2cfd4076c31a1bc8de3274d590dd720229bfda9935dbb27c88ba57274&X-Amz-SignedHeaders=host&actor_id=44926076&key_id=0&repo_id=694565231"
            width="17%"
          />
      </a>
</div>
