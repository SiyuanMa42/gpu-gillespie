# GPU-Gillespie Makefile
# Development and build automation

.PHONY: help install install-dev test lint format clean build upload docs benchmark

# Default target
help:
	@echo "GPU-Gillespie Development Commands"
	@echo "=================================="
	@echo ""
	@echo "Installation:"
	@echo "  make install      - Install package"
	@echo "  make install-dev  - Install with development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linting"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean build artifacts"
	@echo ""
	@echo "Building:"
	@echo "  make build        - Build package"
	@echo "  make upload       - Upload to PyPI"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs         - Build documentation"
	@echo ""
	@echo "Testing:"
	@echo "  make benchmark    - Run performance benchmarks"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,examples]"
	pre-commit install

# Development targets
test:
	pytest tests/ -v --cov=gpu_gillespie --cov-report=html --cov-report=term

lint:
	flake8 gpu_gillespie/ tests/
	mypy gpu_gillespie/
	black --check gpu_gillespie/ tests/

format:
	black gpu_gillespie/ tests/
	isort gpu_gillespie/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

# Building targets
build:
	python -m build

upload:
	python -m twine upload dist/*

# Documentation targets
docs:
	cd docs && make html

# Benchmarking targets
benchmark:
	python -m gpu_gillespie.examples.performance_comparison

# Run specific examples
example-basic:
	python gpu_gillespie/examples/basic_usage.py

example-sweep:
	python gpu_gillespie/examples/parameter_sweep_example.py

example-analysis:
	python gpu_gillespie/examples/advanced_analysis.py

# Development server
dev-server:
	python -m http.server 8000 -d output/

# All-in-one development setup
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make benchmark' to run benchmarks"
	@echo "Run 'make example-basic' to test basic functionality"

# CI/CD targets
ci-test:
	pytest tests/ -v --cov=gpu_gillespie --cov-report=xml
	flake8 gpu_gillespie/ tests/
	mypy gpu_gillespie/

ci-build:
	make clean
	make build
	make test

# Release targets
release-check:
	@echo "Checking release readiness..."
	make lint
	make test
	@echo "All checks passed! Ready for release."

release: release-check build
	@echo "Release package built successfully!"
	@echo "Run 'make upload' to publish to PyPI"