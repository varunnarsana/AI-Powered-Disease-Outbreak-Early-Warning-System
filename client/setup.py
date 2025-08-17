from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="disease-outbreak-client",
    version="0.1.0",
    author="Disease Outbreak Team",
    author_email="ml-team@disease-outbreak.example.com",
    description="Python client for the Disease Outbreak Early Warning System API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/disease-outbreak-mlops",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    install_requires=[
        "httpx>=0.23.0",
        "pydantic>=1.9.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.15.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "mypy>=0.910",
            "types-python-dateutil>=2.8.0",
        ]
    },
    project_urls={
        "Bug Reports": "https://github.com/your-org/disease-outbreak-mlops/issues",
        "Source": "https://github.com/your-org/disease-outbreak-mlops",
    },
)
