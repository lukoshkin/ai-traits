# AI-Traits

A collection of AI-driven developer tools, featuring AI-Commit for generating
intelligent commit messages using LLMs based on your git changes.

## Overview

AI-Commit is a command-line tool that analyzes your staged git changes and uses
Large Language Models (LLMs) to generate meaningful, well-structured commit
messages. It supports multiple LLM providers and can learn from your previous
commit messages to maintain a consistent style.

## Features

- 📝 **Context-Aware**: Can analyze previous commit messages to maintain consistent formatting and style
- ⚙️ **Highly Customizable**: Configure your preferred LLM provider, model, temperature, and system prompt
- ✏️ **Editor Integration**: Option to edit the generated message before committing
- 🔄 **Multi-Provider Support**:
  - OpenAI
  - Anthropic (Claude)
  - Google (Gemini)
  - AWS Bedrock models (requires boto3)

## Installation

### Prerequisites

- uv (Python package manager)
- Python 3.11+
- Git

### Install as a Python package (limited to the Python's virtual environment)

```bash
# Install from PyPI (not yet available)
# pip install ai-traits

# Install from source
git clone https://github.com/lukoshkin/ai-traits.git
cd ai-traits
pip install .

# To use AWS Bedrock models, install with the aws extras:
pip install ".[aws]"

# The command 'ai-commit' will be available within your virtual environment
```

### Manual installation (globally accessible thanks to the uv)

```bash
# [! As a prerequisite, install uv !]
# Clone the repository
git clone https://github.com/lukoshkin/ai-traits.git
cd ai-traits

# Optional: if need be, make the script executable
chmod +x ai-traits/ai-commit.py

# Optional: install dependencies required for AWS Bedrock models
uv sync --extra aws

# Create a symlink to make it available system-wide
ln -s $(pwd)/ai-traits/ai-commit.py /usr/local/bin/ai-commit
```

## Configuration

On first run, AI-Commit will create a default configuration file at
`~/.config/ai-commit/config.ini` (or `%APPDATA%\ai-commit\config.ini` on
Windows). You can edit this file directly or use the `ai-commit config`
command.

You'll need to add your API keys to the configuration:

```ini
[DEFAULT]
provider = openai
model = gpt-4o
temperature = 0
max_tokens = 500
template_commit_count = 10
editor =

[API_KEYS]
openai = your_openai_api_key
anthropic = your_anthropic_api_key
google = your_google_api_key
aws = your_aws_access_key_id,aws_secret_access_key,aws_region

[PROJECT_TEMPLATES]
* = Default template text...
project1 = Project-specific template...
```

The configuration allows you to:

- Set default provider and model
- Configure temperature and token limits
- Store API keys for different providers
- Set project-specific commit templates

## Usage

### Basic Usage

```bash
# Generate a commit message for staged changes
ai-commit commit

# Include unstaged modifications
ai-commit commit --all

# Generate and automatically commit
ai-commit commit --commit

# Use a different provider or model
ai-commit commit --provider anthropic --model claude-3-opus

# Create a commit style template from previous commits
ai-commit mimic

# Create a template for a specific project
ai-commit mimic -t project-name

# Create a global template
ai-commit mimic -t global

# Edit configuration file
ai-commit config
```

### Command-line Options

#### Global Options

| Option         | Description                           |
| -------------- | ------------------------------------- |
| `--help`, `-h` | Show help information for the command |

#### Commit Command

| Option                | Description                                              |
| --------------------- | -------------------------------------------------------- |
| `--provider PROVIDER` | LLM provider to use (openai, anthropic, google, bedrock) |
| `--model MODEL`       | Model to use for generating commit messages              |
| `--temperature FLOAT` | Temperature for LLM generation (0.0-1.0)                 |
| `--commit`            | Automatically commit with the generated message          |
| `--all`               | Use all modified files instead of just staged            |

#### Mimic Command

| Option                | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| `-i`, `--idol PATH`   | Source repository to extract commits from (default: current)   |
| `-t`, `--target NAME` | Target project to save template for (default: current project) |

## Examples

### Generate a commit message

```bash
ai-commit commit

Generated commit message:
--------------------------------------------------
feat(auth): implement JWT authentication

Add JWT token generation and validation for secure API access.
Includes token refresh mechanism and expiration handling.
--------------------------------------------------

To use this message, run:
git commit -m 'feat(auth): implement JWT authentication'
```

### Create a commit style template

```bash
ai-commit mimic
```

### Use a different provider and model

```bash
ai-commit commit --provider anthropic --model claude-3-opus

Generated commit message:
--------------------------------------------------
refactor(core): optimize database query performance

Reduce query execution time by 40% through index optimization
and connection pooling improvements.
--------------------------------------------------
```

## Customizing the System Prompt

You can customize the system prompt in the configuration file to guide the LLM
in generating commit messages that match your team's style and conventions.

## Project Structure

AI-Traits is organized with the following structure:

```text
ai-traits/
├── ai_commit.py             # AI-Commit implementation
├── pyproject.toml           # Project metadata and dependencies
└── README.md                # This file
```

When installed as a package, the `ai-commit` command will be available
globally. If you're using the manual installation method, you'll need to
execute the script directly with `./ai-commit.py` or create a symlink as
described in the installation section.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.
To contribute to AI-Traits development:

1. Clone the repository
1. Make your changes
1. Do some testing (no automated tests yet - feel free to add them!)
1. Update the documentation (README.md)
1. Submit a pull request

## TODO

The following improvements are planned for future development:

### Core Functionality Expansion

- [ ] Expand from AI-Commit to AI-Traits with repository code style analysis
  - [ ] Analyze coding patterns in repository
  - [ ] Extract coding style preferences and conventions
  - [ ] Generate a summary of main coding style points

### Code Organization

- [ ] Restructure ai-commit.py into multiple files
  - [ ] Create a proper package structure
  - [ ] Separate concerns (CLI, LLM integration, git operations)
  - [ ] Implement proper module organization

### Performance Improvements

- [ ] Optimize `ai-commit config` startup time
  - [ ] Avoid loading unnecessary libraries (e.g., litellm)
  - [ ] Implement lazy loading for heavy dependencies
  - [ ] Create a lightweight configuration editor module

### User Experience Enhancements

- [ ] Add visual feedback for LLM operations
  - [ ] Implement spinner for loading states
  - [ ] Add thought rendering for reasoning models
  - [ ] Improve progress indication during operations
  - [ ] Add examples folder with configuration examples

## License

This project is licensed under the MIT License - see the LICENSE file for details.
