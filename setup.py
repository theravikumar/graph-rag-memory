import os
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="graphmemo",
    version="0.1.2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "faiss-cpu>=1.7.4",
        "sqlalchemy>=2.0.0",
        "langchain-core>=0.1.0",
        "openai>=1.0.0",
        "groq>=0.4.0"
    ],
    extras_require={
        "local": ["sentence-transformers>=2.2.2"],
        "all": ["sentence-transformers>=2.2.2"]
    },
    author="Ravi",
    author_email="your.email@example.com",
    description="A BYOK Hierarchical Graph Memory library for AI agents.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/graphmemo",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
