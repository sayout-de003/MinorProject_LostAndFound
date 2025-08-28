# 🤝 Contributing to Lost and Found System

First off, thank you for considering contributing to the Lost and Found System! It's people like you that make this project such a great tool for helping people recover their lost items.

## 📋 Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## 🎯 How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates.

**How to Submit a Good Bug Report:**

1. **Use the GitHub issue search** — check if the issue has already been reported.
2. **Check if the issue has been fixed** — try to reproduce it using the latest `main` branch.
3. **Isolate the problem** — create a reduced test case.

**A good bug report shouldn't leave others needing to chase you up for more information.** Include:

- **Quick summary** of what the bug is
- **Steps to reproduce**
  - Be specific!
  - Give sample code if you can
- **What you expected would happen**
- **What actually happens**
- **Screenshots** (if applicable)
- **Environment details** (OS, Python version, Django version, etc.)

### 💡 Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

1. **Use a clear and descriptive title**
2. **Provide a step-by-step description** of the suggested enhancement
3. **Provide specific examples** to demonstrate the steps
4. **Describe the current behavior** and **explain which behavior you expected to see instead**
5. **Explain why this enhancement would be useful** to most users

### 🛠️ Your First Code Contribution

Unsure where to begin contributing? You can start by looking through these `good first issue` and `help wanted` issues:

- **Good first issues** - issues which should only require a few lines of code, and a test or two.
- **Help wanted issues** - issues which should be a bit more involved than `good first issues`.

## 🏗️ Development Setup

### Prerequisites

- Python 3.8+
- PostgreSQL or SQLite
- Redis (for real-time features)
- Git

### Setup Steps

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/MinorProject_LostAndFound.git
   cd MinorProject_LostAndFound
   ```

3. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # OR
   venv\Scripts\activate    # Windows
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

6. **Run migrations**:
   ```bash
   python manage.py migrate
   ```

7. **Start development server**:
   ```bash
   python manage.py runserver
   ```

## 📝 Pull Request Process

1. **Ensure any install or build dependencies are removed** before the end of the layer when doing a build.
2. **Update the README.md** with details of changes to the interface, if applicable.
3. **Increase the version numbers** in any examples files and the README.md to the new version that this Pull Request would represent.
4. **The PR will be merged** once you have the sign-off of at least one other developer.

### PR Guidelines

- **Keep it focused**: Each PR should address a single issue or add a single feature
- **Write tests**: Include tests for new functionality
- **Follow style guidelines**: Use consistent coding style
- **Update documentation**: Keep documentation in sync with code changes
- **Squash commits**: Keep commit history clean

## 🎨 Style Guidelines

### Python Code Style

- Follow **PEP 8** guidelines
- Use **type hints** for function signatures
- Write **comprehensive docstrings** using Google style
- Maximum line length: **88 characters**

### Django Specific

- Use **class-based views** where appropriate
- Follow Django's **naming conventions**
- Use **model managers** for complex queries
- Implement **proper error handling**

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally

## 🧪 Testing

- Write tests for new functionality
- Ensure all tests pass before submitting PR
- Include both unit tests and integration tests
- Test edge cases and error conditions

### Running Tests

```bash
python manage.py test
```

## 📚 Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions and classes
- Update API documentation if endpoints change
- Include examples for complex functionality

## 🏷️ Issue and Pull Request Labels

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Improvements or additions to documentation
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention is needed
- `question` - Further information is requested

## ❓ Questions?

If you have any questions about contributing, please:

1. Check the existing documentation
2. Search existing issues
3. Create a new issue with the `question` label

Thank you for contributing to making the Lost and Found System better! 🎉

---

*This contributing guide was adapted from the [Atom contributing guide](https://github.com/atom/atom/blob/master/CONTRIBUTING.md).*
