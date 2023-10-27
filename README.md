<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="1280"
        src="https://github.com/safevideo/autollm/assets/44926076/6af17028-b7cc-4511-b677-7031ed31ffbc"
      >
    </a>
  </p>

[![version](https://badge.fury.io/py/autollm.svg)](https://badge.fury.io/py/autollm)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![GNU AGPL 3.0](https://img.shields.io/badge/license-AGPL_3.0-green)](LICENSE)

</div>

## 🤔 why autollm?

**Simplify. Unify. Amplify.**

| Feature                         | AutoLLM | LangChain | LlamaIndex | LiteLLM |
| ------------------------------- | :-----: | :-------: | :--------: | :-----: |
| **80+ LLMs**                    |    ✅    |     ✅     |     ✅      |    ✅    |
| **Unified API**                 |    ✅    |     ❌     |     ❌      |    ✅    |
| **20+ Vector Databases**        |    ✅    |     ✅     |     ✅      |    ❌    |
| **Cost Calculation (80+ LLMs)** |    ✅    |     ❌     |     ❌      |    ✅    |
| **1-Line RAG LLM Engine**       |    ✅    |     ❌     |     ❌      |    ❌    |
| **1-Line FastAPI**              |    ✅    |     ❌     |     ❌      |    ❌    |

______________________________________________________________________

## 📦 installation

Easily install autollm package with pip in [**Python>=3.8**](https://www.python.org/downloads/) environment.

```bash
pip install autollm
```

______________________________________________________________________

## 🎯 quickstart

### create a query engine in seconds

```python
>>> from autollm.utils.document_reading import read_local_files_as_documents
>>> from autollm import AutoQueryEngine

>>> query_engine = AutoQueryEngine.from_parameters(
>>>   documents: List[llama_index.Documents]
>>> )

>>> response = query_engine.query(
>>>   "Why did SafeVideo AI develop this project?"
>>> )

>>> response.response
"Because they wanted to deploy rag based llm apis in no time!"
```

### convert it to a FastAPI app in 1-line

```python
>>> import uvicorn

>>> from autollm import AutoFastAPI

>>> app = AutoFastAPI.from_query_engine(query_engine)

>>> uvicorn.run(app, host="0.0.0.0", port=8000)
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://http://0.0.0.0:8000/
```

</details>

<details>
    <summary>👉 advanced usage </summary>

```python
>>> from autollm import AutoQueryEngine

>>> query_engine = AutoQueryEngine.from_parameters(
>>>   documents=documents,
>>>   system_prompt= ...
>>>   query_wrapper_prompt= ...
>>>   enable_cost_calculator=True,
>>>   llm_params={"model": "gpt-3.5-turbo"},
>>>   vector_store_params={
>>>     "vector_store_type": "LanceDBVectorStore",
>>>     "uri": "/tmp/lancedb",
>>>     "table_name": "lancedb",
>>>     "nprobs": 20
>>>   },
>>>   service_context_params={"chunk_size": 1024},
>>>   query_engine_params={"similarity_top_k": 10},
>>> )

>>> response = query_engine.query("Who is SafeVideo AI?")

>>> print(response.response)
"A startup that provides self hosted AI API's for companies!"
```

```python
>>> from autollm import AutoFastAPI

>>> app = AutoFastAPI.from_query_engine(
      query_engine,
      api_title= ...,
      api_description= ...,
      api_version= ...,
      api_term_of_service= ...,
    )

>>> uvicorn.run(app, host="0.0.0.0", port=8000)
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://http://0.0.0.0:8000/
```

</details>

______________________________________________________________________

## 🌟 features

### supports [80+ LLMs](https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json)

<details>
    <summary>👉 microsoft azure - openai example:</summary>

```python
>>> from autollm import AutoLLM

>>> os.environ["AZURE_API_KEY"] = ""
>>> os.environ["AZURE_API_BASE"] = ""
>>> os.environ["AZURE_API_VERSION"] = ""

>>> llm = AutoLLM(model="azure/<your_deployment_name>")
```

</details>

<details>
    <summary>👉 google - vertexai example</summary>

```python
>>> from autollm import AutoLLM

>>> os.environ["VERTEXAI_PROJECT"] = "hardy-device-38811"  # Your Project ID`
>>> os.environ["VERTEXAI_LOCATION"] = "us-central1"  # Your Location

>>> llm = AutoLLM(model="text-bison@001")
```

</details>

<details>
<summary>👉 aws bedrock - claude v2 example</summary>

```python
>>> from autollm import AutoLLM

>>> os.environ["AWS_ACCESS_KEY_ID"] = ""
>>> os.environ["AWS_SECRET_ACCESS_KEY"] = ""
>>> os.environ["AWS_REGION_NAME"] = ""

>>> llm = AutoLLM(model="anthropic.claude-v2")
```

</details>

### supports [20+ VectorDBs](https://docs.llamaindex.ai/en/stable/core_modules/data_modules/storage/vector_stores.html#vector-store-options-feature-support)

🌟 **pro tip**: `autollm` defaults to `lancedb` as the vector store since it is lightweight,
scales from development to production and is 100x cheaper than alternatives!

<details>
    <summary>👉 default - lancedb example</summary>

```python
>>> from autollm import AutoVectorStoreIndex

>>> vector_store_index = AutoVectorStoreIndex.from_defaults(
>>>     documents=documents
>>> )
```

</details>

### automated cost calculation for [80+ LLMs](https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json)

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

Switching from Llama-Index? We've got you covered.

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

Our roadmap outlines upcoming features and integrations to make autollm the most extensible and powerful base package for large language model applications.

- [ ] **1-line [Gradio](https://www.gradio.app/) app creation and deployment**

- [ ] **Budget based email notification**

- [ ] **Built-in LLM evaluation**

- [ ] **Unit tests for online vectorDB integrations**

- [ ] **Example code snippet to Readme on how to integrate llama-hub readers**

______________________________________________________________________

## 📜 license

autollm is available under the [GNU Affero General Public License (AGPL 3.0)](LICENSE).

______________________________________________________________________

## 📞 contact

For more information, support, or questions, please contact:

- **Email**: [support@safevideo.ai](mailto:support@safevideo.ai)
- **Website**: [SafeVideo](https://safevideo.ai/)
- **LinkedIn**: [SafeVideo AI](https://www.linkedin.com/company/safevideo/)

______________________________________________________________________

## 🌟 contributing

**Love AutoLLM? Star the repo or contribute and help us make it even better!** See our [contributing guidelines](CONTRIBUTING.md) for more information.

<p align="center">
    <a href="https://github.com/safevideo/autollm/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=safevideo/autollm" />
    </a>
</p>

<div align="center">
      <a href="https://www.linkedin.com/company/safevideo/">
          <img
            src="https://github.com/safevideo/autollm/assets/44926076/ee33237c-77b6-4760-91f6-a3aff9b75e56"
            width="17%"
          />
      </a>
</div>
