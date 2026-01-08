# High-Quality Anki Flashcard Generator

An AI-powered flashcard generator that creates **effective, science-backed study cards** based on SuperMemo's 20 Rules of Formulating Knowledge.

Transform any text into optimized Anki flashcards that actually help you learn and retain information.

## Why This Tool?

Most flashcard generators create cards that are:
- Too vague ("What is X?")
- Testing multiple facts at once
- Missing context
- Not optimized for long-term retention

This generator applies **proven learning science principles** to create cards that work with your memory, not against it.

## Key Features

### Based on Learning Science

Implements [SuperMemo's 20 Rules of Formulating Knowledge](https://www.supermemo.com/en/blog/twenty-rules-of-formulating-knowledge):

| Principle | How We Apply It |
|-----------|-----------------|
| **Minimum Information** | Each card tests exactly ONE fact |
| **Avoid Sets** | Lists are broken into individual cloze cards |
| **Cloze Deletions** | Facts use fill-in-the-blank format for better recall |
| **Context in Questions** | Questions are self-contained and unambiguous |
| **Mnemonic Hooks** | Memory aids for difficult concepts |
| **Prerequisites** | Cards identify what you should learn first |
| **Source References** | Track where information came from |

### Multiple Card Types

| Type | Best For | Example |
|------|----------|---------|
| **Basic** | "Why" and "how" questions | Q: Why is Docker useful? A: Provides consistent environments |
| **Cloze** | Facts and definitions | "Docker uses {{c1::containers}} to isolate applications" |
| **Reverse** | Terminology (creates 2 cards) | Term→Definition AND Definition→Term |
| **Concept** | Abstract ideas with examples | Concept + concrete example to anchor understanding |

### Quality Analysis

Every card gets a quality score based on:
- **Atomicity** (0-1): Is it testing one thing?
- **Clarity** (0-1): Is the question unambiguous?
- **Context** (0-1): Does it include necessary context?
- **Interference Risk** (0-1): Could it be confused with similar cards?

Cards with issues include specific improvement suggestions.

### Anki-Optimized Export

- Separate files for Basic and Cloze note types
- Proper tag formatting
- Ready-to-import CSV format

## Quick Start

### 1. Set Up Azure OpenAI

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="gpt-4o"  # optional
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Flashcards

```bash
python src/simple_flashcard_generator.py
```

Follow the interactive prompts to:
1. Enter your topic
2. Select difficulty level
3. Paste content or fetch from Wikipedia
4. Export to Anki format

## Usage Examples

### Interactive Mode

```bash
python src/simple_flashcard_generator.py
```

### Programmatic Usage

```python
import asyncio
from simple_flashcard_generator import FlashcardGenerator, CardType, Difficulty

async def main():
    generator = FlashcardGenerator()

    flashcards = await generator.generate_flashcards(
        content="Your learning material here...",
        topic="Machine Learning",
        difficulty=Difficulty.INTERMEDIATE,
        card_types=[CardType.BASIC, CardType.CLOZE, CardType.REVERSE]
    )

    # Display with quality metrics
    generator.print_flashcards(flashcards)

    # Export for Anki
    generator.export_to_anki_csv(flashcards)

    # Show statistics
    generator.print_statistics()

asyncio.run(main())
```

### From Wikipedia

```python
flashcards = await generator.generate_from_wikipedia(
    topic="Spaced repetition",
    difficulty=Difficulty.BEGINNER,
    sentences=15
)
```

## Difficulty Levels

| Level | Focus | Card Style |
|-------|-------|------------|
| **Beginner** | Foundations, terminology | More cloze, more mnemonics, simpler language |
| **Intermediate** | Concepts + applications | Balanced, connects ideas |
| **Advanced** | Nuances, edge cases | Comparisons, misconceptions, practical scenarios |

## Best Practices for Input Content

### Good Content

```
Docker is a platform for developing, shipping, and running
applications in containers. Containers are lightweight,
isolated environments that package applications with their
dependencies.

Key Components:
- Docker Engine: The runtime that runs containers
- Docker Image: Read-only template for creating containers
- Dockerfile: Script defining how to build an image
```

The generator will:
1. Create atomic cards for each component
2. Use cloze for definitions
3. Generate reverse cards for terminology
4. Add context to all questions

### Avoid

- Very short content (< 50 characters)
- Content without clear concepts
- Already-formatted Q&A (let the AI optimize)

## Output Files

### Basic Cards (`*_basic.csv`)

```csv
Front,Back,Tags,Concept,Mnemonic,Example,Source
"What is Docker's primary purpose?","Containerization of applications","Docker containers","Docker purpose","","Packaging apps with dependencies",""
```

Import into Anki's **Basic** note type.

### Cloze Cards (`*_cloze.csv`)

```csv
Text,Extra,Tags
"Docker uses {{c1::containers}} to provide isolated environments for applications","Lightweight alternative to VMs","Docker containers"
```

Import into Anki's **Cloze** note type.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Required | Your Azure OpenAI resource URL |
| `AZURE_OPENAI_API_KEY` | Required | API authentication key |
| `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME` | `gpt-4o` | Model deployment name |
| `AZURE_OPENAI_MAX_TOKENS` | `4000` | Maximum response tokens |
| `AZURE_OPENAI_TEMPERATURE` | `0.2` | Generation randomness (lower = more focused) |
| `AZURE_OPENAI_API_VERSION` | `2024-12-01-preview` | API version |

## Project Structure

```
anki-analysis/
├── src/
│   ├── simple_flashcard_generator.py  # Main generator
│   ├── demo.py                        # Usage demonstrations
│   └── test_simple.py                 # Test suite
├── requirements.txt
└── README.md
```

## How It Works

1. **Content Analysis**: The AI analyzes your text for key concepts, terminology, and relationships

2. **Principle Application**: Cards are generated following learning science rules:
   - Break lists into atomic facts
   - Create cloze deletions for definitions
   - Add context to questions
   - Identify prerequisites and interference risks

3. **Quality Scoring**: Each card is evaluated for:
   - Atomicity (one fact per card)
   - Clarity (unambiguous questions)
   - Context (self-contained)
   - Interference (confusion with similar concepts)

4. **Export**: Cards are formatted for direct Anki import with proper note type separation

## Testing

```bash
# Run test suite
python src/test_simple.py

# Run demonstrations
python src/demo.py
```

## Why These Principles Matter

Research shows that **how you formulate knowledge** directly impacts retention:

- **Atomic cards** activate focused memory pathways
- **Cloze deletions** force active recall (more effective than recognition)
- **Context** prevents interference between similar facts
- **Mnemonics** create additional retrieval pathways
- **Breaking sets** prevents "list amnesia" where you forget middle items

Cards that violate these principles may feel productive but lead to:
- Faster forgetting
- Confusion between similar concepts
- Difficulty recalling during actual use

## Troubleshooting

### "Missing Azure OpenAI configuration"

Set the required environment variables:
```bash
export AZURE_OPENAI_ENDPOINT="https://..."
export AZURE_OPENAI_API_KEY="..."
```

### "Content too short"

Provide at least 50 characters of meaningful content. The generator needs substance to create good cards.

### Low quality scores

Review the suggestions provided with each card. Common issues:
- Questions starting with "it" or "this" (add context)
- Answers with multiple comma-separated items (split into cards)
- Questions under 5 words (add more context)

### Wikipedia not working

Install the optional dependency:
```bash
pip install wikipedia
```

## References

- [SuperMemo's 20 Rules of Formulating Knowledge](https://www.supermemo.com/en/blog/twenty-rules-of-formulating-knowledge)
- [Effective Learning: Twenty Rules of Formulating Knowledge](https://www.supermemo.com/en/archives1990-2015/articles/20rules)
- [Anki Manual - Cloze Deletion](https://docs.ankiweb.net/editing.html#cloze-deletion)

---

*Built for learners who want flashcards that actually work.*
