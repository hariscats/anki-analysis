# Azure Flashcard Generator

A simple Python tool that creates flashcards from your text using Azure OpenAI. Perfect for studying technical topics!

## Quick Start

1. **Set up Azure OpenAI** (replace with your actual values):
   ```powershell
   $env:AZURE_OPENAI_ENDPOINT = "https://your-resource-name.openai.azure.com/"
   $env:AZURE_OPENAI_API_KEY = "your-api-key-here"
   $env:AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = "gpt-4o"
   ```

2. **Install dependencies**:
   ```bash
   pip install openai
   ```

3. **Generate flashcards**:
   ```bash
   python src/simple_flashcard_generator.py
   ```

That's it! Paste your content, and you'll get a CSV file with flashcards ready for any study app.

## What You Get

- **Smart flashcards**: 5-8 Q&A pairs focused on key concepts
- **CSV export**: Works with Anki, Quizlet, or any flashcard app
- **Simple workflow**: Text in → flashcards out

## Files

- `src/simple_flashcard_generator.py` - Main generator
- `src/test_simple.py` - Test your setup
- `src/demo.py` - See examples

## Example

**Input**: Text about Azure Functions  
**Output**: 
- Q: "What does Azure Functions provide?" 
- A: "Serverless compute for event-triggered code"

The CSV file gets saved with a timestamp, ready to import anywhere!

## Troubleshooting

- Missing environment variables? Run `python src/test_simple.py` to check
- Need help? Check `SIMPLIFIED_GUIDE.md` for details

---

*Built for developers who want to turn documentation into study material quickly and easily.*