# Contributing to TGI Memory Profiler

Thank you for your interest in contributing to TGI Memory Profiler! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct (see CODE_OF_CONDUCT.md). Please read it before contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/tgi-profiler.git
   cd tgi-profiler
   ```
3. Set up your development environment:
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   
   # Install development dependencies
   pip install -e ".[dev]"
   ```

## Development Process

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bugfix-name
   ```

2. Make your changes, following our coding standards:
   - Follow PEP 8 style guidelines
   - Write descriptive commit messages
   - Include docstrings for new functions and classes
   - Add type hints to function signatures
   - Update documentation as needed

3. Add tests for any new functionality

4. Run the test suite:
   ```bash
   pytest tests/
   ```

5. Run the linter:
   ```bash
   flake8 tgi_profiler tests
   ```

## Submitting Changes

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Submit a Pull Request (PR) on GitHub:
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe your changes in detail
   - Include any relevant screenshots or output examples

3. Wait for review:
   - Maintainers will review your PR
   - Address any requested changes
   - Once approved, your PR will be merged

## Testing Guidelines

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Include integration tests for complex features
- Test edge cases and error conditions
- Maintain test coverage above 80% if possible

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions and classes
- Update API documentation if needed
- Include examples for new features
- Document any breaking changes

## Reporting Issues

- Use the GitHub issue tracker
- Check for existing issues before creating new ones
- Include system information and Python version
- Provide clear steps to reproduce the issue
- Include relevant logs or error messages

## Feature Requests

- Use the GitHub issue tracker with the "enhancement" label
- Clearly describe the feature and its use case
- Provide examples of how the feature would be used
- Discuss potential implementation approaches

## Code Review Process

All submissions require review. We use GitHub pull requests for this purpose:

1. Code quality and style
2. Test coverage and correctness
3. Documentation completeness
4. Performance implications
5. Compatibility considerations

## Development Setup Tips

- Use pre-commit hooks for consistent code formatting:
  ```bash
  pre-commit install
  ```
- Set up your IDE with:
  - flake8 linting
  - yapf formatting
  - mypy type checking

## Community

- Join our discussions on GitHub Discussions
- Ask questions in the "Discussions" tab
- Help others in the community
- Share your success stories and use cases

## License

By contributing, you agree that your contributions will be licensed under the project's GNU General Public License v3.

## Questions?

If you have questions, please:
1. Check the documentation
2. Search existing issues and discussions
3. Open a new discussion if needed

Thank you for contributing to TGI Memory Profiler!