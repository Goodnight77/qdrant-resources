# Contributing to Qdrant Resources

Thank you for your interest in contributing to Qdrant Resources! 🎉

We're passionate about contributing to the community and sharing knowledge with developers exploring vector databases and Qdrant. Your contributions help make these resources better for everyone.

## Table of Contents

- [How to Contribute](#how-to-contribute)
- [Creating New Tutorials](#creating-new-tutorials)
- [Creating New Guides](#creating-new-guides)
- [Creating New Projects](#creating-new-projects)
- [Existing Folder Structure](#existing-folder-structure)
- [Reporting Issues](#reporting-issues)
- [Pull Request Process](#pull-request-process)
- [Code of Conduct](#code-of-conduct)

## How to Contribute

We welcome contributions in the following forms:

- **Tutorials**: Step-by-step implementations teaching specific Qdrant concepts (can include guides)
- **Projects**: Complete working examples with all necessary scripts and dependencies
- **Bug fixes**: Corrections to existing tutorials or code
- **Improvements**: Enhancements to existing content

## Creating New Tutorials

Tutorials teach specific Qdrant concepts or features. They can be:
- Jupyter notebooks (`.ipynb`) - preferred for interactive, step-by-step learning
- Python scripts (`.py`) - for standalone demonstrations
- Any format that effectively teaches the concept

### Guidelines for Tutorials

1. **Location**: Place tutorials in the appropriate existing folder or create a new one if needed:
   - `Qdrant-101/` - Basic Qdrant concepts and getting started
   - `RAG/` - Retrieval Augmented Generation examples
   - `quantization/` - Vector quantization techniques
   - `Agents/` - Agent-based implementations
   - `image-recommendation/` - Image recommendation systems

2. **Format**: Choose the format that best suits your tutorial (Jupyter notebooks are preferred for their interactive nature)

3. **Structure**:
   - Start with a clear title and description
   - Include prerequisites and dependencies
   - Add installation instructions if needed
   - Provide step-by-step explanations
   - Include code comments
   - End with a summary or next steps

4. **Requirements**: If your tutorial needs specific dependencies, include a `requirements.txt` file in the folder

5. **Guides** (optional but recommended): Create accompanying markdown guides (`.md` files) in the tutorial folder to:
   - Explain complex concepts covered in the tutorial
   - Provide setup instructions (e.g., how to obtain and set API keys)
   - Document configuration steps
   - Add troubleshooting tips
   - Include additional context or theory
   - Common naming: `README.md`, `setup_guide.md`, `colab_checklist.md`, etc.

### Example Tutorial Structure

```markdown
# Tutorial Title

## Overview
Brief description of what this tutorial covers

## Prerequisites
- Python 3.10+
- Basic knowledge of vector databases

## Installation
```python
!pip install qdrant-client
```

## Step 1: Setup
...
```

### Example Tutorial Folder with Guides

```
image-recommendation/
├── image_rec_qdrant.ipynb    # Main tutorial notebook
├── launch.py          # Launch script
├── requirements.txt            # Dependencies
└── README.md                   # Overview and usage guide
```

## Creating New Projects

Projects are complete, self-contained implementations with all necessary scripts and resources.

### Guidelines for Projects

1. **Location**: Create a new folder in the root directory for each project

2. **Required Files**:
   - `README.md` - Project overview, setup instructions, and usage
   - `requirements.txt` - All Python dependencies
   - Main scripts and modules
   - Sample data or instructions to obtain data (if applicable)

3. **Folder Structure Example**:
   ```
   my_project/
   ├── README.md
   ├── requirements.txt
   ├── main.py
   ├── config.py
   ├── utils/
   │   └── helper.py
   └── data/
       └── sample_data.json
   ```

4. **README Requirements**:
   - Project description
   - Features
   - Installation instructions
   - Usage examples
   - Configuration options
   - License information

## Existing Folder Structure

Our current repository structure:

- `Agents/` - LangGraph and agent-based implementations
- `Qdrant-101/` - Foundational Qdrant tutorials
- `quantization/` - Vector quantization tutorials
- `RAG/` - Retrieval Augmented Generation examples (LangChain, LlamaIndex)

If your contribution doesn't fit into an existing folder, feel free to create a new one with a descriptive name.

## Reporting Issues

Found a bug or error in an existing tutorial? We appreciate your help! 🐛

### How to Report Issues

1. Go to the [Issues](https://github.com/Goodnight77/qdrant-resources/issues) page
2. Click "New Issue"
3. Provide a clear title describing the problem
4. Include:
   - The affected file(s)
   - Description of the issue
   - Steps to reproduce (if applicable)
   - Expected vs. actual behavior
   - Error messages or screenshots
   - Your environment (Python version, OS, etc.)

### Issue Labels

- `bug` - Something isn't working
- `documentation` - Improvements or additions to documentation
- `enhancement` - New feature or request
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed

## Pull Request Process

Ready to submit your contribution? Follow these steps:

### 1. Fork and Clone

```bash
git clone https://github.com/Goodnight77/qdrant-resources.git
cd qdrant-resources
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 3. Make Your Changes

- Follow the guidelines above for your contribution type
- Test your code/tutorial to ensure it works
- Update documentation as needed

### 4. Commit Your Changes

```bash
git add .
git commit -m "Add: descriptive commit message"
```

Use clear commit messages:
- `Add: ...` for new features/content
- `Fix: ...` for bug fixes
- `Update: ...` for improvements to existing content
- `Docs: ...` for documentation changes

### 5. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 6. Create a Pull Request

1. Go to the [original repository](https://github.com/Goodnight77/qdrant-resources)
2. Click "New Pull Request"
3. Select your branch
4. Provide a clear title and description:
   - What does this PR do?
   - Why is this change needed?
   - Any related issues?
   - Screenshots (if applicable)

### 7. Review Process

- We'll review your PR as soon as possible
- We may suggest changes or improvements
- Once approved, your contribution will be merged!

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Guidelines

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what's best for the community
- Show empathy towards others

## Questions?

If you have questions about contributing, feel free to:

- Open an issue with the `question` label
- Reach out to the maintainers

---

Thank you for contributing to Qdrant Resources and helping us share knowledge with the community! 🚀

