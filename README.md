<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="1280"
        src="https://user-images.githubusercontent.com/44926076/278813325-e545319a-4652-43a7-b02b-45ec877bcfdc.png"
      >
    </a>
  </p>

<br>

[questions](https://github.com/safevideo/autollm/discussions/categories/q-a) | [feature requests](https://github.com/safevideo/autollm/discussions/categories/feature-requests)

<br>

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

for built-in data readers (github, pdf, docx, ipynb, epub, mbox, websites..), install with:

```bash
pip install autollm[readers]
```

______________________________________________________________________

## 🎯 quickstart

### tutorials

- **video tutorials**:

  - [Make RAG with API Just in a few Seconds!!!](https://www.youtube.com/watch?v=sgKpBMGC6M0&list=LL&index=2)
  - [Autollm automagically create llm apps in seconds](https://www.youtube.com/watch?v=HZik9309wQc&list=LL&index=3)
  - [AutoLLM: Create RAG Based LLM Web Apps in SECONDS!](https://www.youtube.com/watch?v=kPaiZe_qD34)
  - [AutoLLM: Ship RAG based LLM Apps and API in Seconds](https://www.youtube.com/watch?v=iTGbwD-sSxM)
  - [🚀 AutoLLM: Unlock the Power of 100+ Language Models! Step-by-Step Tutorial](https://www.youtube.com/watch?v=Dh9pIWO1VJw)

- **blog posts**:

  - [Introduction to AutoLLM](https://abrahimzaman360.medium.com/introduction-to-autollm-c8cd31be2a5f)

- **colab notebooks**:

  - quickstart: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safevideo/autollm/blob/main/examples/quickstart.ipynb)

### create a query engine in seconds

```python
>>> from autollm import AutoQueryEngine, read_files_as_documents

>>> documents = read_files_as_documents(input_dir="examples/data")
>>> query_engine = AutoQueryEngine.from_defaults(documents)

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

>>> query_engine = AutoQueryEngine.from_defaults(
...     documents=documents,
...     system_prompt='...',
...     query_wrapper_prompt='...',
...     enable_cost_calculator=True,
...     embed_model="local",
...     chunk_size=1024,
...     context_window=4096,
...     llm_model="gpt-3.5-turbo",
...     llm_max_tokens="256",
...     llm_temperature="0.1",
...     vector_store_type="LanceDBVectorStore",
...     lancedb_uri="./.lancedb",
...     lancedb_table_name="vectors",
...     enable_metadata_extraction=False,
...     similarity_top_k=3,
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

>>> llm_model = "huggingface/WizardLM/WizardCoder-Python-34B-V1.0"
>>> llm_api_base = "https://my-endpoint.huggingface.cloud"

>>> AutoQueryEngine.from_defaults(
...     documents='...',
...     llm_model=llm_model,
...     llm_api_base=llm_api_base,
... )
```

<details>
    <summary>👉 more llms:</summary>

- huggingface - ollama example:

  ```python
  >>> from autollm import AutoQueryEngine

  >>> llm_model = "ollama/llama2"
  >>> llm_api_base = "http://localhost:11434"

  >>> AutoQueryEngine.from_defaults(
  ...     documents='...',
  ...     llm_model=llm_model,
  ...     llm_api_base=llm_api_base,
  ... )
  ```

- microsoft azure - openai example:

  ```python
  >>> from autollm import AutoQueryEngine

  >>> os.environ["AZURE_API_KEY"] = ""
  >>> os.environ["AZURE_API_BASE"] = ""
  >>> os.environ["AZURE_API_VERSION"] = ""

  >>> llm_model = "azure/<your_deployment_name>")

  >>> AutoQueryEngine.from_defaults(
  ...     documents='...',
  ...     llm_model=llm_model
  ... )
  ```

- google - vertexai example:

  ```python
  >>> from autollm import AutoQueryEngine

  >>> os.environ["VERTEXAI_PROJECT"] = "hardy-device-38811"  # Your Project ID`
  >>> os.environ["VERTEXAI_LOCATION"] = "us-central1"  # Your Location

  >>> llm_model = "text-bison@001"

  >>> AutoQueryEngine.from_defaults(
  ...     documents='...',
  ...     llm_model=llm_model
  ... )
  ```

- aws bedrock - claude v2 example:

  ```python
  >>> from autollm import AutoQueryEngine

  >>> os.environ["AWS_ACCESS_KEY_ID"] = ""
  >>> os.environ["AWS_SECRET_ACCESS_KEY"] = ""
  >>> os.environ["AWS_REGION_NAME"] = ""

  >>> llm_model = "anthropic.claude-v2"

  >>> AutoQueryEngine.from_defaults(
  ...     documents='...',
  ...     llm_model=llm_model
  ... )
  ```

</details>

### supports [20+ VectorDBs](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores.html#vector-store-options-feature-support)

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

  >>> AutoQueryEngine.from_defaults(
  ...     documents='...',
  ...     vector_store_type=vector_store_type,
  ...     client=client,
  ...     collection_name=collection_name,
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

>>> vector_store = LanceDBVectorStore(uri="./.lancedb")
>>> storage_context = StorageContext.from_defaults(vector_store=vector_store)
>>> service_context = ServiceContext.from_defaults()
>>> index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_contex,
        service_context=service_context,
    )

>>> query_engine = AutoQueryEngine.from_instances(index)
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
