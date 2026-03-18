"""Setup script for Open Swarm"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="open-swarm",
    version="0.4.1",
    author="Open Swarm Contributors",
    description="A lightweight multi-agent rollout framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/open-swarm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.0.0",
    ],
    extras_require={
        "search": ["duckduckgo-search>=4.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
)
