.PHONY: help install install-dev test test-cov lint format clean build docs serve-docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install the package in development mode"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build the package"
	@echo "  docs         - Build documentation"
	@echo "  serve-docs   - Serve documentation locally"
	@echo "  run          - Run the multi-mode agent"
	@echo "  backup       - Create a backup of the project"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs,test]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=multimodeagents --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 multi_mode_agents.py config.py
	mypy multi_mode_agents.py config.py

format:
	black multi_mode_agents.py config.py
	isort multi_mode_agents.py config.py

# Build and clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

build:
	python setup.py sdist bdist_wheel

# Documentation
docs:
	cd docs && make html

serve-docs:
	cd docs/_build/html && python -m http.server 8000

# Run the application
run:
	python multi_mode_agents.py

# Backup
backup:
	@echo "Creating backup..."
	@mkdir -p backups
	@tar -czf backups/multimodeagents-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		--exclude='__pycache__' \
		--exclude='*.pyc' \
		--exclude='.git' \
		--exclude='venv' \
		--exclude='env' \
		--exclude='backups' \
		--exclude='*.db' \
		--exclude='*.log' \
		--exclude='backup_*' \
		.
	@echo "Backup created in backups/"

# Development setup
setup-dev: install-dev
	pre-commit install

# Quick start
quick-start: install
	@echo "Multi-Mode Agents installed successfully!"
	@echo "Run 'make run' to start the agent"
	@echo "Run 'make help' to see all available commands"

# Docker commands (if using Docker)
docker-build:
	docker build -t multimodeagents .

docker-run:
	docker run -it --rm multimodeagents

# Environment setup
setup-env:
	@echo "Setting up development environment..."
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  source venv/bin/activate  # On Unix/macOS"
	@echo "  venv\\Scripts\\activate     # On Windows"

# Check dependencies
check-deps:
	pip list --outdated

update-deps:
	pip install --upgrade -r requirements.txt

# Security checks
security-check:
	bandit -r multi_mode_agents.py config.py

# Performance profiling
profile:
	python -m cProfile -o profile.stats multi_mode_agents.py

# Type checking
type-check:
	mypy multi_mode_agents.py config.py --strict

# All checks
check-all: lint type-check test
	@echo "All checks passed!"

# Release preparation
pre-release: clean check-all build
	@echo "Release preparation complete!"

# Install pre-commit hooks
install-hooks:
	pre-commit install
	pre-commit install --hook-type commit-msg

# Run pre-commit on all files
pre-commit-all:
	pre-commit run --all-files
