import io
import os
import re

import setuptools


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_requirements():
    with open("requirements.txt", encoding="utf8") as f:
        return f.read().splitlines()


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "sahi", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


setuptools.setup(
    name="quickllm",
    version=get_version(),
    author="safevideo",
    author_email="support@safevideo.ai",
    license="AGPL-3.0",
    description="Base Package for Large Language Model Applications",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=get_requirements(),
)