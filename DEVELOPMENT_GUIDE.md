# Development Guide for canml

This guide outlines the process for contributing to the `canml` library, a Python toolkit for preparing CAN bus data for machine learning. It covers setting up the development environment, coding standards, and the contribution workflow.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Setting Up the Development Environment](#setting-up-the-development-environment)
3. [Coding Standards](#coding-standards)
4. [Contribution Workflow](#contribution-workflow)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Contact](#contact)

## Project Overview

`canml` is a Python library designed to parse CAN bus data (e.g., BLF files) using CAN.DBC files, preprocess it, and prepare it for machine learning applications. The initial module, `canmlio`, handles BLF parsing and CSV export. Future modules will add preprocessing, feature extraction, and visualization.

The library aims to be:
- **User-Friendly**: Accessible to non-experts in CAN bus systems.
- **Modular**: Easy to extend with new features.
- **Robust**: Handles errors gracefully and supports large datasets.

## Setting Up the Development Environment

To contribute to `canml`, set up a local development environment as follows:

### Prerequisites
- **Python**: Version 3.7 or higher.
- **Git**: For version control.
- **Virtualenv** (optional): For creating isolated Python environments.
- A BLF file and corresponding CAN.DBC file for testing (e.g., from Vector CANoe/CANalyzer).

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/canml/canml.git
   cd canml
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv canmlenv
   source canmlenv/bin/activate
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` file includes:
   - `cantools`: For DBC parsing and decoding.
   - `python-can`: For BLF file reading.
   - `pandas`: For DataFrame and CSV handling.
   - `numpy`: Dependency for pandas.

4. **Install Development Tools** (optional):
   ```bash
   pip install pytest pytest-cov black flake8
   ```
   - `pytest`: For running tests.
   - `pytest-cov`: For test coverage reports.
   - `black`: For code formatting.
   - `flake8`: For linting.

5. **Verify Setup**:
   Run the example script to ensure the environment is set up correctly:
   ```bash
   python examples/example_usage.py
   ```
   This requires sample `data.blf` and `config.dbc` files in the project directory.

### Notes
- **Vector BLF Support**: `python-can` requires Vector drivers for BLF files on Windows. Install Vector CANoe/CANalyzer or the Vector XL Driver Library.
- **Large Files**: Ensure sufficient memory for large BLF files during testing.

## Coding Standards

To maintain code quality, follow these guidelines:
- **Style**: Use [PEP 8](https://www.python.org/dev/peps/pep-0008/) for code style.
  - Run `black` to format code: `black .`
  - Run `flake8` to check for style issues: `flake8 .`
- **Type Hints**: Use type hints (PEP 484) for function signatures.
- **Docstrings**: Use Google-style docstrings for all functions and modules.
  ```python
  def example_function(arg: str) -> bool:
      """Short description.

      Args:
          arg: Description of arg.

      Returns:
          Description of return value.
      """
  ```
- **Logging**: Use the `logging` module instead of `print` for debugging and user feedback.
- **Modularity**: Keep functions small and focused, following the single responsibility principle.
- **Error Handling**: Handle exceptions gracefully and provide informative error messages.

## Contribution Workflow

To contribute a feature or bug fix:
1. **Create an Issue**:
   - Open an issue on the [GitHub repository](https://github.com/canml/canml) to discuss your proposal.
   - Describe the feature, bug, or improvement clearly.

2. **Fork and Branch**:
   - Fork the repository and clone it locally.
   - Create a new branch for your changes:
     ```bash
     git checkout -b feature/your-feature-name
     ```

3. **Develop Your Changes**:
   - Write code following the [Coding Standards](#coding-standards).
   - Add tests in the `tests/` directory to cover your changes.
   - Update documentation if needed (e.g., README, docstrings).

4. **Test Your Changes**:
   - Run tests with:
     ```bash
     pytest --cov=canml

     python -m unittest tests.test_integration_canmlio -v
     python -m unittest tests.test_canmlio -v
     ```
   - Ensure 100% test coverage for new code.
   - Fix any linting issues with `black` and `flake8`.

5. **Commit and Push**:
   - Write clear commit messages:
     ```bash
     git commit -m "Add feature X to canmlio module"
     ```
   - Push your branch:
     ```bash
     git push origin feature/your-feature-name
     ```

6. **Submit a Pull Request**:
   - Open a pull request (PR) on GitHub.
   - Reference the related issue in the PR description.
   - Ensure the PR passes all CI checks (tests, linting).

7. **Code Review**:
   - Respond to feedback from maintainers.
   - Make necessary changes and push updates to the same branch.

## Testing

Tests are located in the `tests/` directory and use `pytest`. To run tests:
```bash
pytest --cov=canml --cov-report=html
```
- Write unit tests for all new functions.
- Use mock BLF and DBC files for testing (avoid real data in the repository).
- Test edge cases, such as:
  - Empty BLF files.
  - Invalid DBC files.
  - Missing signals in the DBC.
- Aim for 100% code coverage, reported via `pytest-cov`.

## Documentation

- **Docstrings**: Update docstrings for all new or modified functions.
- **README**: Update `README.md` with new features or usage examples.
- **Examples**: Add example scripts to the `examples/` directory.
- **API Documentation**: Future versions will use Sphinx for API docs. For now, ensure docstrings are clear and comprehensive.

## Contact

For questions or support:
- Open an issue on the [GitHub repository](https://github.com/canml/canml).
- Join the community discussion (TBD as the project grows).

Thank you for contributing to `canml`! Your efforts help make CAN bus data accessible for machine learning applications.