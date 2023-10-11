
# QuickLLM

## Natural Language Query Engine for Large Document Sets

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version 1.0.0](https://img.shields.io/badge/version-1.0.0-blue)
![License CC NC](https://img.shields.io/badge/license-CC_NC-green)
---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Web-Based (FastAPI Swagger UI)](#web-based-fastapi-swagger-ui)
  - [Python Package](#python-package)
- [FAQ](#faq)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [Testimonials](#testimonials)
- [License](#license)
- [Contact](#contact)

---

## Introduction

![QuickLLM Logo](logo.png)

Welcome to QuickLLM, your go-to service for querying large sets of documents using natural language queries. Engineered for both single-user and enterprise-level applications, QuickLLM delivers high-performance, low-latency responses tailored to your specific needs.

---

## Features

- **Natural Language Queries**: Ask questions and get precise answers from your documents.
- **Scalable**: Designed to handle large sets of documents.
- **Fast**: High-performance, low-latency responses.
- **Easy to Use**: Simple setup and query process.
- **Open Source**: Developed under a Creative Commons NonCommercial (CC NC) license.

---

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

## Configuration

A sample `.env` file is provided in the repository as `.env.sample`. To configure the application to suit your specific needs, make a copy of this file, rename it to `.env`, and update the variables.

If you don't include a `.env` file, the application will run with default settings.

---

## Usage


### Web-Based (FastAPI Swagger UI)

The easiest way to use QuickLLM is through the web-based interface. To run the FastAPI server, use the following command:

```bash
uvicorn main:app --reload
```

Then navigate to http://127.0.0.1:8000/docs for the Swagger UI.

### Python Package

QuickLLM can also be used in a Python environment. Here's a quick example:

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
A: No, QuickLLM is licensed under Creative Commons NonCommercial (CC NC), making it unsuitable for commercial use.

**Q: How do I contribute?**  
A: Please see the [Contributing Guidelines](LINK_TO_CONTRIBUTING_GUIDELINES).

---

## Contributing

We welcome contributions to QuickLLM! Please see our [Contributing Guidelines](LINK_TO_CONTRIBUTING_GUIDELINES) for more details.

---

## Roadmap

- Q1 2023: Support for more document formats
- Q2 2023: Advanced analytics dashboard
- Q3 2023: Machine learning-based query optimization

---

## Testimonials

> "QuickLLM transformed the way we search through our document base. It's fast, accurate, and a joy to use!"  
> — John Doe, CEO of Ultralytics (hopefully :)

---

## License

QuickLLM is available under the [Creative Commons NonCommercial (CC NC) License](LICENSE.txt).

---

## Contact

For more information, support, or questions, please contact:

- **Email**: [support@safevideo.ai](mailto:support@quickllm.com)
- **Website**: [SafeVideo](https://safevideo.ai/)
- **LinkedIn**: [SafeVideo AI](https://www.linkedin.com/company/safevideo/)
---
