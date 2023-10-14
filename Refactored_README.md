# AutoLLM

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![Version 0.0.1](https://img.shields.io/badge/version-0.0.1-blue)
![GNU AGPL 3.0](https://img.shields.io/badge/license-AGPL_3.0-green)

## Elevate Your Large Language Model Applications

Welcome to AutoLLM, the definitive toolkit for deploying, managing, and scaling Large Language Model (LLM) applications. Built for high performance and maximum flexibility, AutoLLM provides seamless integration with multiple LLM providers, vector databases, and service contexts.

______________________________________________________________________

## Quick Start

Get up and running with just a few lines of code:

```python
from autollm import AutoLLM, AutoVectorStore, AutoQueryEngine

# Initialize and query
llm = AutoLLM(model="gpt-3.5-turbo")
vector_store = AutoVectorStore.from_defaults(vector_store_type="qdrant")
query_engine = AutoQueryEngine.from_instances(vector_store, llm.service_context)
response = query_engine.query("What is the meaning of life?")
```

______________________________________________________________________

## Features

### Comprehensive LLM Support

AutoLLM seamlessly integrates with [80+ Large Language Models](https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json), including but not limited to:

- **Microsoft Azure - OpenAI**
- **Google - VertexAI**
- **AWS Bedrock - Claude v2**

[See detailed usage](#detailed-usage)

### AutoVectorStore

Instantly initialize a VectorDB instance with one of the supported vector databases:

- **Pinecone**
- **Qdrant**
- **InMemory**

### AutoQueryEngine

Create robust query engine pipelines with automatic cost logging. Supports fine-grained control for advanced use-cases.

### Automated Cost Calculation

Keep track of your LLM token usage and costs in real-time.

______________________________________________________________________

## Installation

AutoLLM is a Python package compatible with Python >= 3.8. Install it via pip:

```bash
pip install autollm
```

______________________________________________________________________

## Advanced Usage

For those looking for more control and customization, AutoLLM allows you to...

______________________________________________________________________

## FAQ

**Q: Can I use this for commercial projects?**
A: Yes, AutoLLM is licensed under GNU AGPL 3.0, allowing for commercial usage under certain conditions.

______________________________________________________________________

## Roadmap

- [ ] Future support for additional VectorDBs
- [ ] Additional query optimization features

______________________________________________________________________

## Contributing

Contributions are welcome! [See guidelines](CONTRIBUTING.md)

______________________________________________________________________

## License

AutoLLM is licensed under the [GNU Affero General Public License (AGPL 3.0)](LICENSE.txt).

______________________________________________________________________

## Contact

For more information, contact us:

- **Email**: [support@safevideo.ai](mailto:support@safevideo.ai)
- **Website**: [SafeVideo](https://safevideo.ai/)
- **LinkedIn**: [SafeVideo AI](https://www.linkedin.com/company/safevideo/)
