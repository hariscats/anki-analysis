# Azure Flashcard Generator

Python tool to generate high-quality, ultra-concise flashcards for technical learning.

## Features

- **Multiple content sources**: Auto-generate content, import from files, paste directly, use Wikipedia
- **Automated refinement**: AI-powered iterative improvement process
- **Quality assessment**: Evaluates cards on atomicity, conciseness, technical precision and more
- **CSV export**: Export flashcards for import into spaced repetition systems

## Setup

1. Set Azure OpenAI environment variables:
   ```
   AZURE_OPENAI_ENDPOINT
   AZURE_OPENAI_API_KEY
   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME (default: o4-mini)
   ```

2. Install dependencies:
   ```bash
   pip install openai azure-identity wikipedia
   ```

## Usage

```bash
# Interactive mode (recommended)
python src/flashcard-generator.py

# Auto-generate content
python src/flashcard-generator.py --auto

# Create from file
python src/flashcard-generator.py --file

# Check configuration
python src/flashcard-generator.py --check
```

## Card Quality Principles

- **Atomicity**: One concept per card
- **Conciseness**: Questions 4-10 words, answers 2-7 words
- **Technical precision**: Exact terminology, no ambiguity
- **Bidirectional design**: Answer works as standalone question if reversed