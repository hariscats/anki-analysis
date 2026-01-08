"""
High-Quality Anki Flashcard Generator

Implements SuperMemo's 20 Rules of Formulating Knowledge and Anki best practices:
- Atomic cards (minimum information principle)
- Multiple card types (basic, cloze, reverse, concept-example)
- Avoids sets and enumerations (breaks them down)
- Quality analysis and scoring
- Proper Anki import format with tags and cloze syntax

Reference: https://www.supermemo.com/en/blog/twenty-rules-of-formulating-knowledge
"""

import os
import csv
import json
import asyncio
import logging
import re
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from openai import AsyncAzureOpenAI

try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False


class CardType(Enum):
    """Types of flashcards optimized for different learning scenarios"""
    BASIC = "basic"              # Simple Q&A
    CLOZE = "cloze"              # Fill-in-the-blank (best for facts)
    REVERSE = "reverse"          # Creates both Q→A and A→Q cards
    CONCEPT_EXAMPLE = "concept"  # Concept with concrete example
    ENUMERATION = "enum"         # For lists (broken into atomic cards)


class Difficulty(Enum):
    """Learning difficulty levels affecting card complexity"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class QualityMetrics:
    """Metrics for evaluating flashcard quality based on learning science"""
    atomicity_score: float      # 0-1: Is it one fact per card?
    clarity_score: float        # 0-1: Is the question unambiguous?
    context_score: float        # 0-1: Does it include necessary context?
    interference_risk: float    # 0-1: Risk of confusion with similar cards
    overall_score: float        # Weighted average
    suggestions: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.overall_score = (
            self.atomicity_score * 0.3 +
            self.clarity_score * 0.3 +
            self.context_score * 0.25 +
            (1 - self.interference_risk) * 0.15
        )


@dataclass
class Flashcard:
    """
    Enhanced flashcard supporting multiple types and Anki-specific features.

    Follows minimum information principle: each card tests ONE piece of knowledge.
    """
    question: str
    answer: str
    card_type: CardType
    topic: str
    tags: List[str]
    difficulty: Difficulty
    concept: str
    mnemonic_hint: Optional[str] = None     # Memory aid suggestion
    example: Optional[str] = None           # Concrete example for abstraction
    source: Optional[str] = None            # Where the information came from
    prerequisites: List[str] = field(default_factory=list)  # What to learn first
    quality: Optional[QualityMetrics] = None
    cloze_text: Optional[str] = None        # For cloze deletions: "{{c1::answer}}"

    def to_anki_format(self) -> Dict:
        """
        Convert to Anki-compatible format.

        For cloze cards, the question field contains the cloze text.
        Tags are space-separated for Anki import.
        """
        base = {
            'Front': self.question,
            'Back': self.answer,
            'Tags': ' '.join(self.tags),
            'Type': self.card_type.value,
            'Topic': self.topic,
            'Concept': self.concept,
            'Difficulty': self.difficulty.value,
            'Source': self.source or '',
            'Created': datetime.now().strftime('%Y-%m-%d'),
        }

        if self.card_type == CardType.CLOZE and self.cloze_text:
            base['Front'] = self.cloze_text
            base['Back'] = ''  # Anki generates back from cloze

        if self.mnemonic_hint:
            base['Mnemonic'] = self.mnemonic_hint

        if self.example:
            base['Example'] = self.example

        return base

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export (legacy format)"""
        return {
            'question': self.question,
            'answer': self.answer,
            'card_type': self.card_type.value,
            'topic': self.topic,
            'tags': ';'.join(self.tags),
            'difficulty': self.difficulty.value,
            'concept': self.concept,
            'mnemonic': self.mnemonic_hint or '',
            'example': self.example or '',
            'source': self.source or '',
            'quality_score': f"{self.quality.overall_score:.2f}" if self.quality else '',
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


class FlashcardGenerator:
    """
    High-quality flashcard generator implementing learning science best practices.

    Key Principles (from SuperMemo's 20 Rules):
    1. Minimum information principle - one fact per card
    2. Cloze deletions for factual recall
    3. Avoid sets - break down lists into atomic facts
    4. Include context in questions
    5. Use mnemonic hooks for difficult material
    6. Build prerequisite chains
    7. Add source references
    """

    # Configurable via environment variables
    DEFAULT_API_VERSION = "2024-12-01-preview"
    DEFAULT_MAX_TOKENS = 4000
    DEFAULT_TEMPERATURE = 0.2
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 2  # Exponential backoff base in seconds

    def __init__(self, log_level: int = logging.INFO):
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.client = self._initialize_azure_client()
        self.token_usage = {'prompt': 0, 'completion': 0, 'total': 0}
        self.generation_stats = {'cards_generated': 0, 'api_calls': 0, 'total_time': 0}

    def _initialize_azure_client(self) -> AsyncAzureOpenAI:
        """Initialize Azure OpenAI client with validation"""
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        if not endpoint or not api_key:
            raise ValueError(
                "Missing Azure OpenAI configuration. Set environment variables:\n"
                "  AZURE_OPENAI_ENDPOINT - Your Azure OpenAI resource URL\n"
                "  AZURE_OPENAI_API_KEY - Your API key\n"
                "  AZURE_OPENAI_CHAT_DEPLOYMENT_NAME (optional) - Model deployment name"
            )

        api_version = os.getenv("AZURE_OPENAI_API_VERSION", self.DEFAULT_API_VERSION)

        client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )

        self.logger.info(f"Azure OpenAI client initialized (API version: {api_version})")
        return client

    async def _call_api_with_retry(self, messages: List[Dict],
                                    temperature: Optional[float] = None) -> Tuple[str, Dict]:
        """
        Make API call with exponential backoff retry logic.

        Returns: Tuple of (response_content, usage_stats)
        """
        deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o")
        max_tokens = int(os.getenv("AZURE_OPENAI_MAX_TOKENS", self.DEFAULT_MAX_TOKENS))
        temp = temperature if temperature is not None else float(
            os.getenv("AZURE_OPENAI_TEMPERATURE", self.DEFAULT_TEMPERATURE)
        )

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                start_time = time.time()

                response = await self.client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                    response_format={"type": "json_object"}
                )

                elapsed = time.time() - start_time
                self.generation_stats['api_calls'] += 1
                self.generation_stats['total_time'] += elapsed

                if not response.choices:
                    raise ValueError("Empty response from API")

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty content in response")

                # Track token usage
                if response.usage:
                    self.token_usage['prompt'] += response.usage.prompt_tokens
                    self.token_usage['completion'] += response.usage.completion_tokens
                    self.token_usage['total'] += response.usage.total_tokens

                usage = {
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                    'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                    'elapsed_seconds': elapsed
                }

                self.logger.debug(f"API call succeeded in {elapsed:.2f}s, {usage['total_tokens'] if response.usage else 0} tokens")
                return content, usage

            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_DELAY_BASE ** (attempt + 1)
                    self.logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"API call failed after {self.MAX_RETRIES} attempts: {e}")

        raise last_error

    def _get_system_prompt(self, card_types: List[CardType], difficulty: Difficulty) -> str:
        """
        Generate system prompt incorporating knowledge formulation best practices.

        Based on SuperMemo's 20 Rules of Formulating Knowledge.
        """
        type_instructions = self._get_card_type_instructions(card_types)
        difficulty_guidance = self._get_difficulty_guidance(difficulty)

        return f"""You are an expert at creating high-quality Anki flashcards based on learning science.

CORE PRINCIPLES (SuperMemo's 20 Rules):

1. MINIMUM INFORMATION PRINCIPLE
   - Each card tests exactly ONE piece of knowledge
   - If you can split a card, split it
   - Bad: "What are the 3 types of X?" → Good: Separate card for each type

2. AVOID SETS AND ENUMERATIONS
   - Never ask "List all..." or "What are the N types of..."
   - Break lists into: individual facts, overlapping cloze deletions, or hierarchical cards
   - Example: Instead of "Name 3 AWS compute services"
     → "EC2 provides {{{{c1::virtual servers}}}} in AWS"
     → "Lambda provides {{{{c1::serverless functions}}}} in AWS"

3. USE CLOZE DELETIONS FOR FACTS
   - Format: "The {{{{c1::answer}}}} is the key part"
   - Cloze is more effective than basic Q&A for factual recall
   - Can have multiple clozes: {{{{c1::first}}}} and {{{{c2::second}}}}

4. INCLUDE CONTEXT IN QUESTIONS
   - Questions must be unambiguous without seeing the answer
   - Bad: "What is it used for?" → Good: "What is Docker used for in software development?"

5. OPTIMIZE WORDING
   - Use simple, direct language
   - Eliminate unnecessary words
   - Front-load important information

6. ADD MNEMONIC HOOKS (when helpful)
   - Memory aids, acronyms, visual associations
   - Only when the fact is genuinely difficult to remember

{type_instructions}

{difficulty_guidance}

OUTPUT FORMAT (JSON):
{{
  "cards": [
    {{
      "type": "basic|cloze|reverse|concept",
      "question": "Clear, context-rich question",
      "answer": "Concise, specific answer",
      "cloze_text": "For cloze: The {{{{c1::answer}}}} goes here",
      "concept": "Core concept being tested",
      "tags": ["topic", "subtopic"],
      "mnemonic": "Optional memory aid",
      "example": "Concrete example if abstract concept",
      "prerequisites": ["concepts to learn first"],
      "interference_notes": "Similar concepts that might cause confusion"
    }}
  ],
  "quality_notes": "Brief assessment of the source material quality"
}}

CRITICAL RULES:
- Generate 8-15 cards depending on content density
- Each card MUST be atomic (one fact)
- Prefer cloze for definitions and facts
- Use reverse cards for terminology (term↔definition)
- Include prerequisites for complex topics
- Add tags for organization (topic::subtopic format)
- Flag potential interference with similar concepts"""

    def _get_card_type_instructions(self, card_types: List[CardType]) -> str:
        """Get instructions for requested card types"""
        instructions = ["CARD TYPES TO GENERATE:"]

        type_details = {
            CardType.BASIC: """
   BASIC: Simple question → answer
   - Use for "why" and "how" questions
   - Question must be self-contained
   - Answer should be 1-2 sentences max""",

            CardType.CLOZE: """
   CLOZE: Fill-in-the-blank using {{c1::answer}} syntax
   - Best for definitions, facts, and terminology
   - The blank should be the KEY piece of information
   - Context around the blank helps recall
   - Example: "In Python, {{c1::list comprehensions}} provide a concise way to create lists" """,

            CardType.REVERSE: """
   REVERSE: Generate BOTH directions (will create 2 cards)
   - Question→Answer AND Answer→Question
   - Perfect for term↔definition pairs
   - Mark with "type": "reverse" """,

            CardType.CONCEPT_EXAMPLE: """
   CONCEPT: Abstract concept with concrete example
   - Include both the concept and a memorable example
   - Example helps anchor abstract knowledge
   - "example" field is required for this type""",

            CardType.ENUMERATION: """
   ENUMERATION HANDLING: Break down lists into atomic cards
   - Never ask to list multiple items
   - Create overlapping cloze cards instead
   - Example for "3 pillars of AWS":
     → "{{c1::Compute}}, Storage, and Networking are AWS's three pillars"
     → "Compute, {{c1::Storage}}, and Networking are AWS's three pillars"
     → "Compute, Storage, and {{c1::Networking}} are AWS's three pillars" """
        }

        for ct in card_types:
            if ct in type_details:
                instructions.append(type_details[ct])

        return '\n'.join(instructions)

    def _get_difficulty_guidance(self, difficulty: Difficulty) -> str:
        """Get guidance based on difficulty level"""
        guidance = {
            Difficulty.BEGINNER: """
DIFFICULTY: BEGINNER
- Focus on foundational concepts and definitions
- More cloze cards for terminology
- Simpler language, more context
- Include more prerequisites
- Add helpful mnemonics""",

            Difficulty.INTERMEDIATE: """
DIFFICULTY: INTERMEDIATE
- Balance concepts and applications
- Include "how" and "why" questions
- Connect concepts to each other
- Moderate prerequisite assumptions""",

            Difficulty.ADVANCED: """
DIFFICULTY: ADVANCED
- Focus on nuances, edge cases, and deep understanding
- Include comparisons between similar concepts
- Address common misconceptions
- Assume foundational knowledge
- Include practical application scenarios"""
        }
        return guidance.get(difficulty, guidance[Difficulty.INTERMEDIATE])

    def _parse_response(self, response: str, topic: str,
                        difficulty: Difficulty, source: Optional[str]) -> List[Flashcard]:
        """Parse API response and create Flashcard objects with quality metrics"""
        try:
            # Handle potential JSON wrapped in markdown code blocks
            cleaned = response.strip()
            if cleaned.startswith('```'):
                cleaned = re.sub(r'^```(?:json)?\n?', '', cleaned)
                cleaned = re.sub(r'\n?```$', '', cleaned)

            data = json.loads(cleaned)
            cards = []

            for card_data in data.get('cards', []):
                card_type = CardType(card_data.get('type', 'basic'))
                tags = card_data.get('tags', [])
                if topic and topic.lower() not in [t.lower() for t in tags]:
                    tags.insert(0, topic)

                # Calculate quality metrics
                quality = self._assess_card_quality(card_data)

                flashcard = Flashcard(
                    question=card_data.get('question', ''),
                    answer=card_data.get('answer', ''),
                    card_type=card_type,
                    topic=topic,
                    tags=tags,
                    difficulty=difficulty,
                    concept=card_data.get('concept', ''),
                    mnemonic_hint=card_data.get('mnemonic'),
                    example=card_data.get('example'),
                    source=source,
                    prerequisites=card_data.get('prerequisites', []),
                    quality=quality,
                    cloze_text=card_data.get('cloze_text')
                )
                cards.append(flashcard)

                # Generate reverse card if requested
                if card_type == CardType.REVERSE:
                    reverse_card = Flashcard(
                        question=card_data.get('answer', ''),
                        answer=card_data.get('question', ''),
                        card_type=CardType.BASIC,
                        topic=topic,
                        tags=tags + ['reverse'],
                        difficulty=difficulty,
                        concept=card_data.get('concept', '') + ' (reverse)',
                        source=source,
                        quality=quality
                    )
                    cards.append(reverse_card)

            self.generation_stats['cards_generated'] += len(cards)
            return cards

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}\nResponse: {response[:500]}")
            raise ValueError(f"Invalid JSON response: {e}")

    def _assess_card_quality(self, card_data: Dict) -> QualityMetrics:
        """
        Assess flashcard quality based on learning science principles.

        Evaluates:
        - Atomicity: Is it testing one thing?
        - Clarity: Is the question unambiguous?
        - Context: Does it include necessary context?
        - Interference risk: Could it be confused with similar cards?
        """
        question = card_data.get('question', '')
        answer = card_data.get('answer', '')
        suggestions = []

        # Atomicity check
        atomicity = 1.0
        list_indicators = ['list', 'name all', 'what are the', 'enumerate', 'multiple']
        if any(ind in question.lower() for ind in list_indicators):
            atomicity = 0.3
            suggestions.append("Consider breaking into multiple atomic cards")
        if len(answer.split(',')) > 2:
            atomicity = min(atomicity, 0.5)
            suggestions.append("Answer contains multiple items - consider splitting")

        # Clarity check
        clarity = 1.0
        vague_words = ['it', 'this', 'that', 'they', 'these']
        if question.lower().startswith(tuple(vague_words)):
            clarity = 0.5
            suggestions.append("Question starts with vague pronoun - add context")
        if len(question) < 20:
            clarity = min(clarity, 0.6)
            suggestions.append("Question may be too brief - ensure it's self-contained")

        # Context check
        context = 1.0
        if '?' in question and len(question.split()) < 5:
            context = 0.6
            suggestions.append("Question may lack sufficient context")
        if card_data.get('concept'):
            context = min(context + 0.1, 1.0)

        # Interference risk (based on notes provided by LLM)
        interference = 0.2  # Base risk
        if card_data.get('interference_notes'):
            interference = 0.5
            suggestions.append(f"Watch for confusion with: {card_data.get('interference_notes')}")

        return QualityMetrics(
            atomicity_score=atomicity,
            clarity_score=clarity,
            context_score=context,
            interference_risk=interference,
            overall_score=0,  # Calculated in __post_init__
            suggestions=suggestions
        )

    async def generate_flashcards(
        self,
        content: str,
        topic: str,
        difficulty: Difficulty = Difficulty.INTERMEDIATE,
        card_types: Optional[List[CardType]] = None,
        source: Optional[str] = None,
        custom_instructions: Optional[str] = None
    ) -> List[Flashcard]:
        """
        Generate high-quality flashcards from content.

        Args:
            content: The text to create flashcards from
            topic: Topic name for tagging
            difficulty: Learning level (affects card complexity)
            card_types: Types of cards to generate (default: all types)
            source: Source reference (URL, book, etc.)
            custom_instructions: Additional generation instructions

        Returns:
            List of Flashcard objects with quality metrics
        """
        # Input validation
        if not content or len(content.strip()) < 50:
            raise ValueError("Content must be at least 50 characters for meaningful flashcard generation")

        if not topic or len(topic.strip()) < 2:
            raise ValueError("Topic must be provided")

        # Default to all card types
        if card_types is None:
            card_types = [CardType.BASIC, CardType.CLOZE, CardType.REVERSE, CardType.CONCEPT_EXAMPLE]

        self.logger.info(f"Generating flashcards for '{topic}' ({difficulty.value})")
        self.logger.info(f"Content length: {len(content)} chars, Card types: {[ct.value for ct in card_types]}")

        system_prompt = self._get_system_prompt(card_types, difficulty)

        user_prompt = f"""TOPIC: {topic}
SOURCE: {source or 'User provided content'}

CONTENT TO PROCESS:
{content}

{f'ADDITIONAL INSTRUCTIONS: {custom_instructions}' if custom_instructions else ''}

Generate high-quality flashcards following all the principles above. Focus on the most important concepts."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response, usage = await self._call_api_with_retry(messages)

        self.logger.debug(f"API response: {response[:500]}...")

        flashcards = self._parse_response(response, topic, difficulty, source)

        # Log quality summary
        if flashcards:
            avg_quality = sum(c.quality.overall_score for c in flashcards if c.quality) / len(flashcards)
            self.logger.info(f"Generated {len(flashcards)} cards, average quality: {avg_quality:.2f}")

        return flashcards

    async def generate_from_wikipedia(
        self,
        topic: str,
        difficulty: Difficulty = Difficulty.INTERMEDIATE,
        sentences: int = 10
    ) -> List[Flashcard]:
        """
        Generate flashcards from Wikipedia article.

        Args:
            topic: Wikipedia article title to search
            difficulty: Learning level
            sentences: Number of sentences to extract

        Returns:
            List of flashcards with Wikipedia source attribution
        """
        if not WIKIPEDIA_AVAILABLE:
            raise ImportError("Wikipedia library not installed. Run: pip install wikipedia")

        self.logger.info(f"Fetching Wikipedia article: {topic}")

        try:
            page = wikipedia.page(topic)
            content = wikipedia.summary(topic, sentences=sentences)
            source = page.url

            self.logger.info(f"Retrieved {len(content)} chars from Wikipedia")

            return await self.generate_flashcards(
                content=content,
                topic=topic,
                difficulty=difficulty,
                source=source
            )
        except wikipedia.exceptions.DisambiguationError as e:
            self.logger.warning(f"Ambiguous topic, using first option: {e.options[0]}")
            return await self.generate_from_wikipedia(e.options[0], difficulty, sentences)
        except wikipedia.exceptions.PageError:
            raise ValueError(f"Wikipedia article not found: {topic}")

    def export_to_anki_csv(self, flashcards: List[Flashcard],
                           filename: Optional[str] = None) -> str:
        """
        Export flashcards in Anki-compatible CSV format.

        The output can be imported directly into Anki:
        1. File → Import
        2. Select the CSV file
        3. Set field separator to comma
        4. Map fields appropriately

        For cloze cards, import into a Cloze note type.
        """
        if not flashcards:
            raise ValueError("No flashcards to export")

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_slug = re.sub(r'[^\w\s-]', '', flashcards[0].topic).strip().replace(' ', '_')[:20]
            filename = f"anki_{topic_slug}_{timestamp}.csv"

        # Separate cloze and non-cloze cards (different Anki note types)
        cloze_cards = [c for c in flashcards if c.card_type == CardType.CLOZE and c.cloze_text]
        basic_cards = [c for c in flashcards if c not in cloze_cards]

        files_created = []

        # Export basic cards
        if basic_cards:
            basic_filename = filename.replace('.csv', '_basic.csv')
            with open(basic_filename, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['Front', 'Back', 'Tags', 'Concept', 'Mnemonic', 'Example', 'Source']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for card in basic_cards:
                    data = card.to_anki_format()
                    writer.writerow({k: data.get(k, '') for k in fieldnames})
            files_created.append(basic_filename)
            self.logger.info(f"Exported {len(basic_cards)} basic cards to {basic_filename}")

        # Export cloze cards
        if cloze_cards:
            cloze_filename = filename.replace('.csv', '_cloze.csv')
            with open(cloze_filename, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['Text', 'Extra', 'Tags']  # Anki cloze format
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for card in cloze_cards:
                    writer.writerow({
                        'Text': card.cloze_text,
                        'Extra': card.example or card.mnemonic_hint or '',
                        'Tags': ' '.join(card.tags)
                    })
            files_created.append(cloze_filename)
            self.logger.info(f"Exported {len(cloze_cards)} cloze cards to {cloze_filename}")

        return ', '.join(files_created)

    def export_to_csv(self, flashcards: List[Flashcard],
                      filename: Optional[str] = None) -> str:
        """Export flashcards to general CSV format (works with any app)"""
        if not flashcards:
            raise ValueError("No flashcards to export")

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flashcards_{timestamp}.csv"

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = list(flashcards[0].to_dict().keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for card in flashcards:
                writer.writerow(card.to_dict())

        self.logger.info(f"Exported {len(flashcards)} flashcards to {filename}")
        return filename

    def print_flashcards(self, flashcards: List[Flashcard], show_quality: bool = True):
        """Display flashcards with quality metrics"""
        if not flashcards:
            print("No flashcards to display")
            return

        print(f"\n{'='*60}")
        print(f" FLASHCARDS: {flashcards[0].topic}")
        print(f" Generated: {len(flashcards)} cards")
        print(f"{'='*60}\n")

        for i, card in enumerate(flashcards, 1):
            type_icon = {
                CardType.BASIC: "📝",
                CardType.CLOZE: "📋",
                CardType.REVERSE: "🔄",
                CardType.CONCEPT_EXAMPLE: "💡",
                CardType.ENUMERATION: "📊"
            }.get(card.card_type, "📝")

            print(f"{type_icon} Card {i} [{card.card_type.value.upper()}]")
            print(f"   Q: {card.question}")

            if card.cloze_text:
                print(f"   Cloze: {card.cloze_text}")
            else:
                print(f"   A: {card.answer}")

            print(f"   Concept: {card.concept}")

            if card.mnemonic_hint:
                print(f"   💭 Mnemonic: {card.mnemonic_hint}")

            if card.example:
                print(f"   📌 Example: {card.example}")

            if card.prerequisites:
                print(f"   ⚡ Prerequisites: {', '.join(card.prerequisites)}")

            if show_quality and card.quality:
                q = card.quality
                score_bar = "█" * int(q.overall_score * 10) + "░" * (10 - int(q.overall_score * 10))
                print(f"   Quality: [{score_bar}] {q.overall_score:.1%}")
                if q.suggestions:
                    for suggestion in q.suggestions:
                        print(f"      ⚠️  {suggestion}")

            print()

    def print_statistics(self):
        """Print generation statistics"""
        print(f"\n{'='*40}")
        print(" GENERATION STATISTICS")
        print(f"{'='*40}")
        print(f" Cards generated: {self.generation_stats['cards_generated']}")
        print(f" API calls made: {self.generation_stats['api_calls']}")
        print(f" Total time: {self.generation_stats['total_time']:.2f}s")
        print(f" Token usage:")
        print(f"   Prompt: {self.token_usage['prompt']}")
        print(f"   Completion: {self.token_usage['completion']}")
        print(f"   Total: {self.token_usage['total']}")
        print()


async def main():
    """Interactive CLI for flashcard generation"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║        HIGH-QUALITY ANKI FLASHCARD GENERATOR                 ║
║   Based on SuperMemo's 20 Rules of Formulating Knowledge     ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    # Check environment
    if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
        print("❌ Missing Azure OpenAI configuration!")
        print("\nRequired environment variables:")
        print("  AZURE_OPENAI_ENDPOINT - Your Azure OpenAI resource URL")
        print("  AZURE_OPENAI_API_KEY - Your API key")
        print("\nOptional:")
        print("  AZURE_OPENAI_CHAT_DEPLOYMENT_NAME - Model name (default: gpt-4o)")
        print("  AZURE_OPENAI_MAX_TOKENS - Max tokens (default: 4000)")
        print("  AZURE_OPENAI_TEMPERATURE - Temperature (default: 0.2)")
        return

    # Get user input
    print("📚 Enter topic (e.g., 'Python asyncio', 'Kubernetes'):")
    topic = input("   > ").strip()
    if not topic:
        topic = "General Knowledge"

    print("\n📊 Select difficulty:")
    print("   1. Beginner - Focus on foundations and terminology")
    print("   2. Intermediate - Balance concepts and applications")
    print("   3. Advanced - Nuances, edge cases, deep understanding")
    diff_choice = input("   > ").strip()
    difficulty = {
        '1': Difficulty.BEGINNER,
        '2': Difficulty.INTERMEDIATE,
        '3': Difficulty.ADVANCED,
        'beginner': Difficulty.BEGINNER,
        'intermediate': Difficulty.INTERMEDIATE,
        'advanced': Difficulty.ADVANCED
    }.get(diff_choice.lower(), Difficulty.INTERMEDIATE)

    print("\n📝 Choose input source:")
    print("   1. Paste text content")
    if WIKIPEDIA_AVAILABLE:
        print("   2. Fetch from Wikipedia")
    source_choice = input("   > ").strip()

    try:
        generator = FlashcardGenerator()
        flashcards = []

        if source_choice == '2' and WIKIPEDIA_AVAILABLE:
            print(f"\n🌐 Fetching Wikipedia article for '{topic}'...")
            flashcards = await generator.generate_from_wikipedia(topic, difficulty)
        else:
            print("\n📄 Paste your content (press Enter twice when done):")
            lines = []
            empty_count = 0
            while True:
                line = input()
                if line == "":
                    empty_count += 1
                    if empty_count >= 2:
                        break
                else:
                    empty_count = 0
                lines.append(line)

            content = "\n".join(lines[:-1]) if lines else ""

            if len(content.strip()) < 50:
                print("❌ Content too short. Please provide at least 50 characters.")
                return

            print("\n⚙️  Generating high-quality flashcards...")
            flashcards = await generator.generate_flashcards(
                content=content,
                topic=topic,
                difficulty=difficulty
            )

        if flashcards:
            generator.print_flashcards(flashcards)

            # Export options
            print("\n💾 Export options:")
            print("   1. Anki format (separate files for basic/cloze)")
            print("   2. General CSV (works with any app)")
            print("   3. Both")
            export_choice = input("   > ").strip() or '3'

            if export_choice in ['1', '3']:
                anki_files = generator.export_to_anki_csv(flashcards)
                print(f"   ✅ Anki files: {anki_files}")

            if export_choice in ['2', '3']:
                csv_file = generator.export_to_csv(flashcards)
                print(f"   ✅ CSV file: {csv_file}")

            generator.print_statistics()

            print("\n📖 IMPORT INTO ANKI:")
            print("   1. Open Anki → File → Import")
            print("   2. Select the _basic.csv file for Basic note type")
            print("   3. Select the _cloze.csv file for Cloze note type")
            print("   4. Map fields: Front→Front, Back→Back, Tags→Tags")
        else:
            print("❌ No flashcards generated. Try different content.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.exception("Generation failed")


if __name__ == "__main__":
    asyncio.run(main())
