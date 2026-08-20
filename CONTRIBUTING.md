# Contributing to StellarX-StarNav-AI

Thank you for your interest in contributing to this project. This document describes the process and expectations for contributions.

---

## Getting Started

1. Fork the repository and create a feature branch from `main`.
2. Install dependencies into a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Make your changes, following the code style guidelines below.
4. Run the test suite before submitting:
   ```bash
   pytest tests/
   ```
5. Open a pull request with a clear description of the change and its motivation.

---

## Branch Naming

| Type | Pattern | Example |
|---|---|---|
| Feature | `feature/<short-description>` | `feature/star-detection-threshold` |
| Bug fix | `fix/<short-description>` | `fix/catalog-loader-key-error` |
| Documentation | `docs/<short-description>` | `docs/update-methodology` |
| Experiment | `experiment/<short-description>` | `experiment/cnn-baseline` |

---

## Code Style

- Follow **PEP 8** for all Python code.
- Use **type hints** for all public function signatures.
- Write **docstrings** for all modules, classes, and public functions (Google style preferred).
- Keep functions focused — one responsibility per function.
- Use `config.yaml` for all configurable parameters. Do not hard-code paths or hyperparameters.
- Do not commit large binary files (datasets, model weights) to the repository.

---

## Commit Messages

Use the conventional-commit format:

```
<type>(<scope>): <short summary>

[optional body]
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

Example:
```
feat(preprocessing): add background subtraction step
```

---

## Pull Request Guidelines

- Keep PRs focused. One concern per PR.
- Reference any relevant issue numbers in the PR description.
- Ensure all tests pass and no new linting errors are introduced.
- Include a brief description of what was changed and why.
- Add or update tests for any new functionality.

---

## Reporting Issues

Use GitHub Issues to report bugs or request features. When reporting a bug, include:

- A clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python version, relevant package versions)

---

## Code of Conduct

All contributors are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
