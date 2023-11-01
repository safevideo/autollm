<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="1280"
        src="https://user-images.githubusercontent.com/44926076/278813325-e545319a-4652-43a7-b02b-45ec877bcfdc.png"
      >
    </a>
  </p>

[![version](https://badge.fury.io/py/autollm.svg)](https://badge.fury.io/py/autollm)
<a href="https://pepy.tech/project/autollm"><img src="https://pepy.tech/badge/autollm" alt="total autollm downloads"></a>
[![license](https://img.shields.io/pypi/l/autollm)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safevideo/autollm/blob/main/examples/quickstart.ipynb)

</div>

## 🤔 why autollm?

**Simplify. Unify. Amplify.**

| Feature                          | AutoLLM | LangChain | LlamaIndex | LiteLLM |
| -------------------------------- | :-----: | :-------: | :--------: | :-----: |
| **100+ LLMs**                    |    ✅    |     ✅     |     ✅      |    ✅    |
| **Unified API**                  |    ✅    |     ❌     |     ❌      |    ✅    |
| **20+ Vector Databases**         |    ✅    |     ✅     |     ✅      |    ❌    |
| **Cost Calculation (100+ LLMs)** |    ✅    |     ❌     |     ❌      |    ✅    |
| **1-Line RAG LLM Engine**        |    ✅    |     ❌     |     ❌      |    ❌    |
| **1-Line FastAPI**               |    ✅    |     ❌     |     ❌      |    ❌    |

______________________________________________________________________

## 📦 installation

easily install **autollm** package with pip in [**Python>=3.8**](https://www.python.org/downloads/) environment.

```bash
pip install autollm
```

______________________________________________________________________

## 🎯 quickstart

### create a query engine in seconds

```python
>>> from autollm import AutoQueryEngine, read_files_as_documents

>>> documents = read_files_as_documents(input_dir="examples/data")
>>> query_engine = AutoQueryEngine.from_parameters(documents=documents)

>>> response = query_engine.query(
...     "Why did SafeVideo AI develop this project?"
... )

>>> response.response
"Because they wanted to deploy rag based llm apis in no time!"
```

<details>
    <summary>👉 advanced usage </summary>

```python
>>> from autollm import AutoQueryEngine

>>> query_engine = AutoQueryEngine.from_parameters(
...     documents=documents,
...     system_prompt='...',
...     query_wrapper_prompt='...',
...     enable_cost_calculator=True,
...     llm_params={"model": "gpt-3.5-turbo"},
...     vector_store_params={
...       "vector_store_type": "LanceDBVectorStore",
...       "uri": "/tmp/lancedb",
...       "table_name": "lancedb",
...       "nprobs": 20
...     },
...     service_context_params={"chunk_size": 1024},
...     query_engine_params={"similarity_top_k": 10},
... )

>>> response = query_engine.query("Who is SafeVideo AI?")

>>> print(response.response)
"A startup that provides self hosted AI API's for companies!"
```

</details>

### convert it to a FastAPI app in 1-line

```python
>>> import uvicorn

>>> from autollm import AutoFastAPI

>>> app = AutoFastAPI.from_query_engine(query_engine)

>>> uvicorn.run(app, host="0.0.0.0", port=8000)
INFO:    Started server process [12345]
INFO:    Waiting for application startup.
INFO:    Application startup complete.
INFO:    Uvicorn running on http://http://0.0.0.0:8000/
```

<details>
    <summary>👉 advanced usage </summary>

```python
>>> from autollm import AutoFastAPI

>>> app = AutoFastAPI.from_query_engine(
...      query_engine,
...      api_title='...',
...      api_description='...',
...      api_version='...',
...      api_term_of_service='...',
    )

>>> uvicorn.run(app, host="0.0.0.0", port=8000)
INFO:    Started server process [12345]
INFO:    Waiting for application startup.
INFO:    Application startup complete.
INFO:    Uvicorn running on http://http://0.0.0.0:8000/
```

</details>

______________________________________________________________________

## 🌟 features

### supports [100+ LLMs](https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json)

```python
>>> from autollm import AutoQueryEngine

>>> os.environ["HUGGINGFACE_API_KEY"] = "huggingface_api_key"

>>> model = "huggingface/WizardLM/WizardCoder-Python-34B-V1.0"
>>> api_base = "https://my-endpoint.huggingface.cloud"

>>> llm_params = {
...     "model": model,
...     "api_base": api_base,
... }

>>> AutoQueryEngine.from_parameters(
...     documents='...',
...     llm_params=llm_params
... )
```

<details>
    <summary>👉 more llms:</summary>

- huggingface - ollama example:

  ```python
  >>> from autollm import AutoQueryEngine

  >>> model = "ollama/llama2"
  >>> api_base = "http://localhost:11434"

  >>> llm_params = {
  ...     "model": model,
  ...     "api_base": api_base,
  ... }

  >>> AutoQueryEngine.from_parameters(
  ...     documents='...',
  ...     llm_params=llm_params
  ... )
  ```

- microsoft azure - openai example:

  ```python
  >>> from autollm import AutoQueryEngine

  >>> os.environ["AZURE_API_KEY"] = ""
  >>> os.environ["AZURE_API_BASE"] = ""
  >>> os.environ["AZURE_API_VERSION"] = ""

  >>> model = "azure/<your_deployment_name>")
  >>> llm_params = {"model": model}

  >>> AutoQueryEngine.from_parameters(
  ...     documents='...',
  ...     llm_params=llm_params
  ... )
  ```

- google - vertexai example:

  ```python
  >>> from autollm import AutoQueryEngine

  >>> os.environ["VERTEXAI_PROJECT"] = "hardy-device-38811"  # Your Project ID`
  >>> os.environ["VERTEXAI_LOCATION"] = "us-central1"  # Your Location

  >>> model = "text-bison@001"
  >>> llm_params = {"model": model}

  >>> AutoQueryEngine.from_parameters(
  ...     documents='...',
  ...     llm_params=llm_params
  ... )
  ```

- aws bedrock - claude v2 example:

  ```python
  >>> from autollm import AutoQueryEngine

  >>> os.environ["AWS_ACCESS_KEY_ID"] = ""
  >>> os.environ["AWS_SECRET_ACCESS_KEY"] = ""
  >>> os.environ["AWS_REGION_NAME"] = ""

  >>> model = "anthropic.claude-v2"
  >>> llm_params = {"model": model}

  >>> AutoQueryEngine.from_parameters(
  ...     documents='...',
  ...     llm_params=llm_params
  ... )
  ```

</details>

### supports [20+ VectorDBs](https://docs.llamaindex.ai/en/stable/core_modules/data_modules/storage/vector_stores.html#vector-store-options-feature-support)

🌟**Pro Tip**: `autollm` defaults to `lancedb` as the vector store:
it's setup-free, serverless, and 100x more cost-effective!

<details>
    <summary>👉 more vectordbs:</summary>

- QdrantVectorStore example:
  ```python
  >>> from autollm import AutoQueryEngine
  >>> import qdrant_client

  >>> vector_store_type = "QdrantVectorStore"
  >>> client = qdrant_client.QdrantClient(
  ...     url="http://<host>:<port>",
  ...     api_key="<qdrant-api-key>"
  ... )
  >>> collection_name = "quickstart"

  >>> vector_store_params = {
  ...     "vector_store_type": vector_store_type,
  ...     "client": client,
  ...     "collection_name": collection_name,
  ... }

  >>> AutoQueryEngine.from_parameters(
  ...     documents='...',
  ...     vector_store_params=vector_store_params
  ... )
  ```

</details>

### automated cost calculation for [100+ LLMs](https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json)

```python
>>> from autollm import AutoServiceContext

>>> service_context = AutoServiceContext(enable_cost_calculation=True)

# Example verbose output after query
Embedding Token Usage: 7
LLM Prompt Token Usage: 1482
LLM Completion Token Usage: 47
LLM Total Token Cost: $0.002317
```

### create FastAPI App in 1-Line

<details>
    <summary>👉 example</summary>

```python
>>> from autollm import AutoFastAPI

>>> app = AutoFastAPI.from_config(config_path, env_path)
```

Here, `config` and `env` should be replaced by your configuration and environment file paths.

After creating your FastAPI app, run the following command in your terminal to get it up and running:

```bash
uvicorn main:app
```

</details>

______________________________________________________________________

## 🔄 migration from llama-index

switching from Llama-Index? We've got you covered.

<details>
    <summary>👉 easy migration </summary>

```python
>>> from llama_index import StorageContext, ServiceContext, VectorStoreIndex
>>> from llama_index.vectorstores import LanceDBVectorStore

>>> from autollm import AutoQueryEngine

>>> vector_store = LanceDBVectorStore(uri="/tmp/lancedb")
>>> storage_context = StorageContext.from_defaults(vector_store=vector_store)
>>> index = VectorStoreIndex.from_documents(documents=documents)
>>> service_context = ServiceContext.from_defaults()

>>> query_engine = AutoQueryEngine.from_instances(index, service_context)
```

</details>

## ❓ FAQ

**Q: Can I use this for commercial projects?**

A: Yes, AutoLLM is licensed under GNU Affero General Public License (AGPL 3.0),
which allows for commercial use under certain conditions. [Contact](#contact) us for more information.

______________________________________________________________________

## roadmap

our roadmap outlines upcoming features and integrations to make autollm the most extensible and powerful base package for large language model applications.

- [ ] **1-line [Gradio](https://www.gradio.app/) app creation and deployment**

- [ ] **Budget based email notification**

- [ ] **Automated LLM evaluation**

- [ ] **Add more quickstart apps on pdf-chat, documentation-chat, academic-paper-analysis, patent-analysis and more!**

______________________________________________________________________

## 📜 license

autollm is available under the [GNU Affero General Public License (AGPL 3.0)](LICENSE).

______________________________________________________________________

## 📞 contact

for more information, support, or questions, please contact:

- **Email**: [support@safevideo.ai](mailto:support@safevideo.ai)
- **Website**: [SafeVideo](https://safevideo.ai/)
- **LinkedIn**: [SafeVideo AI](https://www.linkedin.com/company/safevideo/)

______________________________________________________________________

## 🏆 contributing

**love autollm? star the repo or contribute and help us make it even better!** see our [contributing guidelines](CONTRIBUTING.md) for more information.

<p align="center">
    <a href="https://github.com/safevideo/autollm/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=safevideo/autollm" />
    </a>
</p>

______________________________________________________________________

<div align="center">
  <b>follow us for more!</b>
  <br>
  <a href="https://www.linkedin.com/company/safevideo/">
      <img
        src="https://user-images.githubusercontent.com/44926076/278822352-30e06f9b-1915-4aed-8081-6796432daa7a.png"
        height="32px"
      />
  </a>
  <a href="https://huggingface.co/safevideo">
      <img
        src="https://user-images.githubusercontent.com/34196005/278877706-ed074c9c-0938-48a1-98e8-39a322faf01d.png"
        height="32px"
      />
  </a>
  <a href="https://twitter.com/safevideo_ai">
      <img
        src="https://user-images.githubusercontent.com/34196005/278877049-141925a9-aa1b-4730-829e-74f6d08ee8ca.png"
        height="32px"
      />
  </a>
</div>
