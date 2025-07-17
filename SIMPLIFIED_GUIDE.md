# Simplified Flashcard Generator

I've created a simplified version of your flashcard generator that focuses on core functionality:

## Files Created

1. **`simple_flashcard_generator.py`** - Main simplified generator
2. **`test_simple.py`** - Test script to verify it works
3. **`demo.py`** - Demo script showing usage examples

## Key Simplifications

### What was simplified:
- ✅ Removed complex iterative refinement
- ✅ Removed multiple content sources (Wikipedia, etc.)
- ✅ Removed quality assessment loops
- ✅ Simplified error handling
- ✅ Reduced complexity from ~1100 lines to ~200 lines

### Core functionality preserved:
- ✅ Azure OpenAI integration
- ✅ Flashcard generation from text content
- ✅ CSV export
- ✅ Environment variable configuration
- ✅ Basic error handling

## How to Use

### 1. Set Environment Variables (PowerShell)
```powershell
$env:AZURE_OPENAI_ENDPOINT = "https://your-resource-name.openai.azure.com/"
$env:AZURE_OPENAI_API_KEY = "your-api-key-here"
$env:AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = "o4-mini"
```

### 2. Run the Simple Generator
```powershell
cd src
python simple_flashcard_generator.py
```

### 3. Or Use Programmatically
```python
from simple_flashcard_generator import SimpleFlashcardGenerator

async def generate_cards():
    generator = SimpleFlashcardGenerator()
    flashcards = await generator.generate_flashcards(
        topic="Your Topic",
        content="Your content here...",
        difficulty="intermediate"
    )
    generator.print_flashcards(flashcards)
    generator.export_to_csv(flashcards)
```

## Features

- **Simple Input**: Just provide topic, content, and difficulty
- **Clean Output**: Generates 5-8 focused flashcards
- **CSV Export**: Exports to timestamped CSV files
- **Error Handling**: Clear error messages for common issues
- **Logging**: Shows what's happening during generation

## What the Generator Does

1. Takes your text content
2. Sends it to Azure OpenAI with a focused prompt
3. Gets back JSON with flashcard data
4. Converts to Python objects
5. Exports to CSV with timestamp

## CSV Output Format

The exported CSV includes:
- `question` - The flashcard question
- `answer` - The flashcard answer  
- `topic` - The topic you specified
- `difficulty` - The difficulty level
- `concept` - The specific concept covered
- `created_date` - When the flashcard was created

## Next Steps

1. **Set your environment variables** using the PowerShell commands above
2. **Test the generator** with `python test_simple.py`
3. **Run interactively** with `python simple_flashcard_generator.py`
4. **Try the demos** with `python demo.py`

The simplified version should generate flashcards successfully and export them to CSV without the complexity of the original version!
