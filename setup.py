"""
Setup script for GPU-Gillespie package
"""

from setuptools import setup, find_packages


# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]


setup(
    name="gpu-gillespie",
    version="1.0.1",
    author="GPU-Gillespie Development Team",
    author_email="contact@gpu-gillespie.org",
    description="High-performance GPU-accelerated Gillespie stochastic simulation algorithms",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/SiyuanMa42/gpu-gillespie",
    project_urls={
        "Bug Tracker": "https://github.com/SiyuanMa42/gpu-gillespie/issues",
        "Documentation": "https://gpu-gillespie.readthedocs.io/",
        "Source": "https://github.com/SiyuanMa42/gpu-gillespie",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "nbsphinx>=0.8",
            "jupyter>=1.0",
            "matplotlib>=3.3",
            "seaborn>=0.11",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "nbsphinx>=0.8",
            "myst-parser>=0.15",
        ],
        "examples": [
            "jupyter>=1.0",
            "matplotlib>=3.3",
            "seaborn>=0.11",
            "plotly>=5.0",
            "ipywidgets>=7.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "gpu-gillespie=gpu_gillespie.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "gpu_gillespie": [
            "examples/*.py",
            "examples/*.ipynb",
            "examples/data/*.csv",
        ],
    },
    zip_safe=False,
    keywords=[
        "gillespie",
        "stochastic simulation",
        "gpu",
        "cuda",
        "parallel computing",
        "biochemical networks",
        "systems biology",
        "computational biology",
    ],
)
