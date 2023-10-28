import io
import os
import re

import setuptools

DEV_REQUIREMETNS = [
    'pre-commit==3.4.0',
    'pytest==7.4.2',
]


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_dir, 'README.md'), encoding='utf-8') as f:
        return f.read()


def get_requirements():
    with open('requirements.txt', encoding='utf8') as f:
        return f.read().splitlines()


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, 'autollm', '__init__.py')
    with open(version_file, encoding='utf-8') as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


def get_author():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    init_file = os.path.join(current_dir, 'autollm', '__init__.py')
    with open(init_file, encoding='utf-8') as f:
        return re.search(r'^__author__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


def get_license():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    init_file = os.path.join(current_dir, 'autollm', '__init__.py')
    with open(init_file, encoding='utf-8') as f:
        return re.search(r'^__license__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


setuptools.setup(
    name='autollm',
    version=get_version(),
    author=get_author(),
    author_email='support@safevideo.ai',
    license=get_license(),
    description="Ship RAG based LLM Web API's, in seconds.",
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/safevideo/autollm',
    packages=setuptools.find_packages(exclude=["tests", "examples"]),
    install_requires=get_requirements(),
    extras_require={'dev': DEV_REQUIREMETNS},
    python_requires='>=3.8',
    classifiers=[
        'Intended Audience :: Developers', 'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Operating System :: MacOS :: MacOS X', 'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX', 'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9', 'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11', 'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Internet :: WWW/HTTP :: HTTP Servers'
    ],
)
