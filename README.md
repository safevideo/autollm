
# QuickLLM

## Base Package for Large Language Model Applications

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
![Version 0.0.1](https://img.shields.io/badge/version-0.0.1-blue)
![GNU AGPL 3.0](https://img.shields.io/badge/license-AGPL_3.0-green)
---

## Introduction

![QuickLLM Logo](logo.png)

Welcome to QuickLLM, the foundational package for building large language model applications. Designed for extensibility and high performance, QuickLLM serves as a base package for more specialized applications in querying, document processing, and more.

---

## Features

- **Automated LLM Integrations**: Seamlessly integrate with leading large language models.
    - OpenAI GPT3.5 and GPT4
    - Google PALM
    - Anyscale LLAMA2

- **Automated VectorDB Integrations**: Quickly connect to vector databases without the manual hassle.
    - Pinecone
    - Qdrant

- **Utility Functions**: Additional utility functions to aid in data manipulation, query optimization, and more.
    - Query Cost Estimation
    - And more!

---

### TODO: Add screenshots/gif or not?
## Screenshots

![Query Example](query_example.png)
![Response Example](response_example.png)

---

## Installation

QuickLLM is available as a Python package for Python>=3.8 environments. Install it using pip:

```bash
pip install quickllm
```
---

### TODO: Add code examples from AutoVectorstore, AutoLLM, AutoQueryEngine.
## Code Examples

QuickLLM is designed to be easy to use. Here's a simple example of how to make a query:

```python
from quickllm import QueryEngine

# Initialize the query engine
engine = QueryEngine()

# Making a query
response = engine.query("What is AI?")
print(response)
```

---

## FAQ

**Q: Can I use this for commercial projects?**  
A: Yes, QuickLLM is licensed under GNU Affero General Public License (AGPL 3.0), which allows for commercial use under certain conditions. [Contact](#contact) us for more information.


**Q: How do I contribute?**  
A: Please see the [Contributing Guidelines](LINK_TO_CONTRIBUTING_GUIDELINES).

---

## Contributing

We welcome contributions to QuickLLM! Please see our [Contributing Guidelines](LINK_TO_CONTRIBUTING_GUIDELINES) for more details.

---

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


---
### TODO: Add testimonials or not?
## Testimonials

> "QuickLLM transformed the way we search through our document base. It's fast, accurate, and a joy to use!"  
> — John Doe, CEO of Ultralytics (hopefully :)

---

## License

QuickLLM is available under the [GNU Affero General Public License (AGPL 3.0)](LICENSE.txt).

---

## Contact

For more information, support, or questions, please contact:

- **Email**: [support@safevideo.ai](mailto:support@quickllm.com)
- **Website**: [SafeVideo](https://safevideo.ai/)
- **LinkedIn**: [SafeVideo AI](https://www.linkedin.com/company/safevideo/)
---
