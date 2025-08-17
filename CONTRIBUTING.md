# Contributing to RightNow CLI

Thank you for your interest in contributing to RightNow CLI! We welcome contributions from the community.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/RightNow-AI/rightnow-cli.git
   cd rightnow-cli
   ```

3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

## Development Setup

### Prerequisites
- Python 3.8 or higher
- CUDA Toolkit 11.0+
- NVIDIA GPU (for testing)
- OpenRouter API key

### Running Tests
```bash
pytest tests/
```

### Code Style
We use Black and isort for code formatting:
```bash
black rightnow_cli/
isort rightnow_cli/
```

### Type Checking
```bash
mypy rightnow_cli/
```

## Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Format your code
6. Commit your changes with clear messages

## Submitting Pull Requests

1. Push to your fork
2. Create a Pull Request with:
   - Clear description of changes
   - Any related issue numbers
   - Screenshots if applicable

## Code Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions/classes
- Keep functions focused and small
- Write tests for new features
- Handle errors gracefully

## Reporting Issues

Please use GitHub Issues to report bugs or request features. Include:
- Clear description
- Steps to reproduce
- Expected vs actual behavior
- System information

## Questions?

Feel free to open an issue or reach out to jaber@rightnowai.co

Thank you for contributing!