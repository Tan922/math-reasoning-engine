from setuptools import setup, find_packages

setup(
    name="math-reasoning-engine",
    version="0.1.0",
    description="Evolvable Mathematical Reasoning Engine (MRE)",
    author="MRE Contributors",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "sympy>=1.12",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "mwparserfromhell>=0.6.5",
        "SPARQLWrapper>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "embed": [
            "sentence-transformers>=2.2.0",
        ],
    },
)
