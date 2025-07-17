"""
Simplified Q&A Flashcard System with Azure OpenAI SDK
Features CSV export, file input, and automatic content generation
"""

import json
import csv
import asyncio
import logging
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
from pathlib import Path
import wikipedia

# Azure OpenAI SDK imports
from openai import AsyncAzureOpenAI
from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential


class DifficultyLevel(Enum):
    """Flashcard difficulty levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ContentSource(Enum):
    """Content source options"""
    AUTO_GENERATE = "auto_generate"
    TEXT_FILE = "text_file"
    DIRECT_INPUT = "direct_input"
    PREDEFINED = "predefined"
    WIKIPEDIA = "wikipedia"


@dataclass
class QAFlashcard:
    """Simple Q&A flashcard data structure"""
    question: str
    answer: str
    topic: str
    difficulty: str
    quality_score: float
    concept: str
    id: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export"""
        return {
            'id': self.id,
            'question': self.question,
            'answer': self.answer,
            'topic': self.topic,
            'difficulty': self.difficulty,
            'quality_score': self.quality_score,
            'concept': self.concept,
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


class AzureConfig:
    """Azure OpenAI configuration following best practices"""
    
    def __init__(self):
        # Use environment variables for secure configuration
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "o4-mini")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure OpenAI configuration missing. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables.")


class ContentManager:
    """Manages content sources for flashcard generation"""
    
    def __init__(self, content_dir: str = "content"):
        self.content_dir = Path(content_dir)
        self.content_dir.mkdir(exist_ok=True)
        self._create_sample_files()
    
    def _create_sample_files(self):
        """Create sample content files if they don't exist"""
        sample_files = {
            "azure_openai.txt": """
Azure OpenAI Service provides REST API access to OpenAI's language models including GPT-4, GPT-3.5-turbo, and embedding models.

Key Features:
- Multiple model deployments with custom names
- Fine-tuning capabilities for custom models  
- Content filtering and safety features
- Token-based pricing and quota management
- Integration with Azure security and compliance

Important Parameters:
- Temperature: Controls randomness (0.0-2.0)
- Max tokens: Maximum response length
- Top-p: Nucleus sampling parameter
- Frequency penalty: Reduces repetition
- Presence penalty: Encourages topic diversity

Authentication uses Azure AD and API keys.
Deployments can be managed through Azure portal or REST APIs.
            """,
            
            "azure_functions.txt": """
Azure Functions is a serverless compute service that lets you run event-triggered code without having to explicitly provision or manage infrastructure.

Key Features:
- Event-driven and serverless architecture
- Multiple programming language support (C#, JavaScript, Python, PowerShell, Java)
- Automatic scaling based on demand
- Pay-per-execution pricing model
- Integration with Azure services and third-party services

Trigger Types:
- HTTP triggers for REST APIs
- Timer triggers for scheduled tasks
- Blob storage triggers for file processing
- Queue triggers for message processing
- Event Grid triggers for event handling

Hosting Plans:
- Consumption Plan (serverless)
- Premium Plan (enhanced performance)
- App Service Plan (dedicated resources)
            """,
            
            "custom_content.txt": """
Paste your custom content here for flashcard generation.

You can include:
- Technical documentation
- Study materials
- Learning notes
- Course content
- Any text-based information

The AI will automatically generate high-quality flashcards from this content.
            """
        }
        
        for filename, content in sample_files.items():
            file_path = self.content_dir / filename
            if not file_path.exists():
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content.strip())
    
    def list_available_files(self) -> List[str]:
        """List available content files"""
        return [f.name for f in self.content_dir.glob("*.txt")]
    
    def read_content_file(self, filename: str) -> str:
        """Read content from a text file"""
        file_path = self.content_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Content file not found: {filename}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def create_content_file(self, filename: str, content: str):
        """Create a new content file"""
        file_path = self.content_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def get_content_dir_path(self) -> str:
        """Get the content directory path"""
        return str(self.content_dir.absolute())


class SimplifiedFlashcardSystem:
    """
    Simplified flashcard system with iterative refinement using Azure OpenAI SDK
    """
    
    def __init__(self):
        self.config = AzureConfig()
        self.client = None
        self.logger = self._setup_logging()
        self.content_manager = ContentManager()
        self._initialize_client()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging following Azure best practices"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _initialize_client(self):
        """Initialize Azure OpenAI client following Azure best practices"""
        try:
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                api_version=self.config.api_version,
                timeout=60.0,
                max_retries=3
            )
            self.logger.info("Successfully initialized Azure OpenAI client")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            raise

    async def _call_azure_openai(self, messages: List[Dict], max_tokens: int = 3000) -> str:
        """Make async call to Azure OpenAI with enhanced error handling"""
        try:
            self.logger.info(f"Making Azure OpenAI API call to endpoint: {self.config.endpoint}")
            self.logger.info(f"Using deployment: {self.config.deployment_name}")
            
            response = await self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9,
                timeout=60
            )
            
            content = response.choices[0].message.content
            self.logger.info(f"Received response of length: {len(content) if content else 0}")
            
            if not content:
                raise ValueError("Empty response from Azure OpenAI")
                
            return content
            
        except Exception as e:
            self.logger.error(f"Azure OpenAI API call failed: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}")
            if hasattr(e, 'response'):
                self.logger.error(f"HTTP status: {e.response.status_code if hasattr(e.response, 'status_code') else 'Unknown'}")
            raise

    async def _auto_generate_content(self, topic: str, difficulty: str) -> str:
        """Auto-generate educational content with enhanced logging"""
        
        self.logger.info(f"Auto-generating content for topic: '{topic}', difficulty: '{difficulty}'")
        
        system_message = """You are an expert technical educator specializing in Azure and cloud technologies. 
Generate comprehensive, accurate educational content for the given topic that would be suitable for creating flashcards.

CONTENT REQUIREMENTS:
1. Include key concepts, definitions, and terminology
2. Cover practical implementation details
3. Mention important features, benefits, and limitations
4. Include configuration options and parameters where relevant
5. Add real-world use cases and examples
6. Structure content with clear sections and bullet points
7. Focus on information that would be valuable for learning and retention

Generate content that is:
- Technically accurate and up-to-date
- Well-structured and easy to understand
- Comprehensive but focused on key learning objectives
- Suitable for the specified difficulty level"""

        user_message = f"""Topic: {topic}
Target Difficulty Level: {difficulty}

Generate comprehensive educational content about this topic that would be perfect for creating high-quality flashcards. 
Include key concepts, features, implementation details, and practical information that learners should know."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            content = await self._call_azure_openai(messages, max_tokens=4000)
            self.logger.info(f"Successfully generated content of {len(content)} characters")
            return content
        except Exception as e:
            self.logger.error(f"Failed to auto-generate content: {str(e)}")
            raise

    async def _generate_flashcards(self, topic: str, content: str, difficulty: str, feedback: str) -> str:
        """Generate flashcards with enhanced error handling"""
        
        self.logger.info(f"Generating flashcards for topic: '{topic}'")
        self.logger.info(f"Content preview: {content[:200]}...")
        
        system_message = """You are an expert at creating ultra-concise, high-quality technical flashcards.

FLASHCARD GENERATION RULES - FOLLOW STRICTLY:
1. EXTREME ATOMICITY – Each card tests exactly ONE atomic concept. Split multi-part answers into separate cards.
2. ULTRA CONCISE – Questions must be 4-10 words. Answers must be 2-7 words. Ruthlessly eliminate all filler words.
3. STANDALONE CONTEXT – Begin questions with domain context (e.g., "In MQTT, what...").
4. DIRECT QUESTIONS – Use "what," "how," "when," "why" formats that require specific answers.
5. NO HINTS – Never include answer clues in the question.
6. TECHNICAL PRECISION – Use exact terminology with zero fluff or ambiguity.
7. PRACTICAL FOCUS – Emphasize function, purpose, mechanism, or use case.
8. BIDIRECTIONAL – Answer should make sense as a question if reversed.
9. UNIQUE CONCEPTS – Each card must cover a distinct concept with no overlap.
10. DESIGN RATIONALE – When possible, highlight why something works as it does.

WORD ELIMINATION PATTERNS:
- Remove "the purpose of" → use "what does X do"
- Remove "is used to" → use direct verbs
- Cut articles (a, an, the) when meaning remains clear
- Replace "in order to" with "to"
- Replace "X is Y" with just "Y" in answers
- Replace "used for" with direct function words

FORMATTING PATTERNS:
- Question: "In [domain], what does [X] do?" (preferred format)
- Question: "What [verb] [X] in [domain]?"
- Question: "How does [X] work in [domain]?"
- Answer: Use 2-7 words with key technical terms only

REWARD CARDS THAT:
- Highlight architectural decisions and system tradeoffs
- Cover operational scenarios including failure modes
- Demonstrate cross-system reasoning or integration points
- Reveal the "why" behind technical decisions

Create 5-8 flashcards in this exact JSON format:
{
  "flashcards": [
    {
      "question": "In Azure OpenAI, what does temperature control?",
      "answer": "Response randomness via token probabilities.",
      "concept": "Azure OpenAI Parameters",
      "difficulty": "beginner",
      "quality_factors": {
        "conceptual_atomicity": 1.0,
        "conciseness": 0.9,
        "standalone_context": 1.0,
        "requires_specific_answer": 1.0,
        "no_hints": 1.0,
        "technical_precision": 1.0, 
        "practical_value": 0.7,
        "bidirectional_design": 0.8,
        "uniqueness": 1.0,
        "design_rationale": 0.5
      }
    }
  ]
}"""

        user_message = f"""Topic: {topic}
Content: {content}
Target Difficulty: {difficulty}
Previous Feedback: {feedback}

Generate high-quality flashcards following the rules above."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = await self._call_azure_openai(messages)
            self.logger.info(f"Generated flashcards response preview: {response[:300]}...")
            return response
        except Exception as e:
            self.logger.error(f"Failed to generate flashcards: {str(e)}")
            raise

    async def _assess_flashcards(self, flashcards: str, topic: str) -> str:
        """Assess flashcard quality using Azure OpenAI"""
        
        system_message = """You are a technical flashcard quality expert. Rate each card based on our 10-point criteria system.

-QUALITY FACTORS (1.0=perfect, 0.0=failing):
-1. CONCEPTUAL ATOMICITY: Does the card test exactly one distinct technical idea?
-2. CONCISENESS: Are both question and answer ≤15 words each, with no filler?
-3. STANDALONE CONTEXT: Does the question include sufficient system/domain context?
-4. SPECIFIC ANSWER: Does the question require a specific, non-binary answer?
-5. NO HINTS: Does the question avoid revealing the answer or providing clues?
-6. TECHNICAL PRECISION: Does the card use accurate domain-specific terms?
-7. PRACTICAL VALUE: Does the card focus on how/why features are used, failure modes, or operational aspects?
-8. BIDIRECTIONAL DESIGN: Could the answer work as a standalone question?
-9. UNIQUENESS: Is the card non-overlapping with other cards in the set?
-10. DESIGN RATIONALE: Does the card reveal design intent or architectural tradeoffs?
+QUALITY FACTORS (1.0=perfect, 0.0=failing):
+1. CONCEPTUAL ATOMICITY: Does the card test exactly one atomic concept with no additional concepts?
+2. CONCISENESS: Is question 4-10 words and answer 2-7 words with zero filler words?
+3. STANDALONE CONTEXT: Does question begin with domain context (e.g., "In MQTT, what...")?
+4. SPECIFIC ANSWER: Does the question require a specific, non-binary answer?
+5. NO HINTS: Does the question avoid any words that appear in the answer?
+6. TECHNICAL PRECISION: Does the card use precise domain terminology without any vague terms?
+7. PRACTICAL VALUE: Does the card focus on function, mechanism, or operational use case?
+8. BIDIRECTIONAL DESIGN: Would the answer work naturally as a standalone question if reversed?
+9. UNIQUENESS: Is the concept entirely distinct from all other cards in the set?
+10. DESIGN RATIONALE: Does the card reveal why something works as it does or architectural intent?

REWARD BONUS FACTORS:
- ARCHITECTURE FOCUS: Card covers architectural decisions or system tradeoffs (+0.5)
- OPERATIONAL INSIGHT: Card covers failure modes or operational scenarios (+0.5)
- CROSS-SYSTEM REASONING: Card demonstrates integration points or cross-service knowledge (+0.5)

PENALTY FACTORS:
- CLARITY ISSUES: Card is confusing or ambiguous (-0.5)
- ATOMICITY ISSUES: Card covers multiple concepts or is unfocused (-0.7)
- UNIQUENESS ISSUES: Card duplicates another card's concept (-0.7)
+- VERBOSITY: Question > 10 words or answer > 7 words (-0.5 per extra word)
+- MISSING CONTEXT: Question doesn't begin with domain context (-0.8)
+- WORDINESS: Uses "is", "are", "the", or other filler words unnecessarily (-0.2 per instance)

+PERFECT CARD EXAMPLES:
+- Q: "In MQTT, what function does broker serve?"
+  A: "Routes messages between publishers and subscribers."
+
+- Q: "What does MQTT QoS 0 guarantee?"
+  A: "At most once delivery."
+
+- Q: "How are MQTT messages categorized?"
+  A: "By hierarchical topic strings."
+
+- Q: "What port does MQTT use unencrypted?"
+  A: "1883."
+
+- Q: "What does MQTT Keep Alive interval do?"
+  A: "Prevents connection timeout by sending periodic messages."

For each flashcard, provide:
1. Quality factors scored individually (0.0-1.0)
2. Overall weighted score (0.0-10.0)
3. Any applicable reward/penalty factors
4. Specific improvement suggestions ranked by priority
5. Words to eliminate or rephrase for conciseness

Return this JSON format:
{
  "assessments": [
    {
      "card_index": 0,
-      "quality_factors": {
-        "conceptual_atomicity": 1.0,
-        "conciseness": 0.7,
-        "standalone_context": 0.9,
-        "requires_specific_answer": 1.0,
-        "no_hints": 1.0,
-        "technical_precision": 1.0,
-        "practical_value": 0.8,
-        "bidirectional_design": 0.6,
-        "uniqueness": 1.0,
-        "design_rationale": 0.5
-      },
+      "quality_factors": {
+        "conceptual_atomicity": 1.0,
+        "conciseness": 0.6,
+        "standalone_context": 1.0,
+        "requires_specific_answer": 1.0,
+        "no_hints": 1.0,
+        "technical_precision": 1.0,
+        "practical_value": 0.8,
+        "bidirectional_design": 0.6,
+        "uniqueness": 1.0,
+        "design_rationale": 0.5
+      },
-      "weighted_score": 8.5,
-      "reward_factors": ["operational_insight"],
-      "penalty_factors": [],
-      "priority_improvements": [
-        "reduce question length from 18 to 15 words",
-        "make answer more bidirectional by including key concept"
-      ],
+      "weighted_score": 8.2,
+      "reward_factors": ["operational_insight"],
+      "penalty_factors": ["verbosity"],
+      "priority_improvements": [
+        "reduce question from 12 to 7 words",
+        "reduce answer from 9 to 5 words",
+        "make answer more bidirectional"
+      ],
-      "conciseness_suggestions": {
-        "question": ["remove 'when using'", "replace 'in order to' with 'to'"],
-        "answer": []
-      }
+      "conciseness_suggestions": {
+        "question": ["remove 'what is the'", "replace with 'In X, what does Y do?'"],
+        "answer": ["remove 'the'", "remove 'and'", "use shorter verb"]
+      }
    }
  ],
  "overall_score": 7.8,
  "quality_distribution": {
    "excellent": 3,
    "good": 4,
    "needs_work": 1,
    "poor": 0
  },
  "critical_issues": 2,
-  "improvement_priorities": [
-    "increase conciseness in questions (5 cards exceed 15 words)",
-    "improve bidirectional design (4 cards have low scores)",
-    "enhance design rationale coverage (most cards score below 0.6)"
-  ]
+  "improvement_priorities": [
+    "reduce question length (target: 7 words average)",
+    "reduce answer length (target: 5 words average)",
+    "ensure domain context prefix in all questions",
+    "eliminate all articles and filler words"
+  ]
}"""

        user_message = f"""Flashcards: {flashcards}
Topic: {topic}

Assess the quality of these flashcards according to the criteria above."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        return await self._call_azure_openai(messages)

    async def _refine_flashcards(self, flashcards: str, assessment: str, topic: str) -> str:
        """Refine flashcards based on assessment using Azure OpenAI"""
        
        system_message = """You are a technical flashcard improvement specialist obsessed with creating ultra-concise flashcards. 
Follow these guidelines exactly to refine the cards based on quality assessment.

ULTRA-CONCISENESS IMPERATIVES:
1. SPLIT MULTI-CONCEPT CARDS: Break any card with multiple facts into separate atomic cards.
2. EXTREME BREVITY: Questions must be 4-10 words; answers must be 2-7 words.
3. DOMAIN PREFIX PATTERN: Start every question with "In [domain], what..." format.
4. DIRECT WORDING: Use direct question patterns (what, how, when, why) only.
5. RUTHLESS WORD ELIMINATION: Remove ALL articles (a, an, the) and filler words.
6. TECHNICAL TERMINOLOGY: Keep precise technical terms, remove everything else.
7. FUNCTIONAL FOCUS: Emphasize mechanism, function, or purpose with action verbs.
8. REVERSIBLE FORMAT: Structure answers so they could function as questions.
9. CONCEPT ISOLATION: Ensure each card covers exactly one unique technical point.
10. ARCHITECTURAL INSIGHT: When possible, reveal why a design choice exists.

WORD ELIMINATION CHECKLIST:
- Remove all forms of "to be" (is, are, was, were)
- Remove all articles (a, an, the) 
- Remove helping verbs (can, may, might, will)
- Remove "in order to", replace with "to" or nothing
- Remove "purpose of" and "role of", replace with "what does X do"
- Replace "is responsible for" with direct verbs
- Replace "is used to" with function words
- Cut ALL words that don't add technical meaning

REFINEMENT INSTRUCTIONS:
- Apply the specific improvement suggestions from the assessment
- Prioritize fixing cards with the lowest scores
- Implement all conciseness suggestions to reduce word count
- Do not sacrifice technical accuracy for brevity
- Preserve the core concept of each card
- Add missing context prefixes where needed

MQTT EXAMPLES TO FOLLOW EXACTLY:
- ORIGINAL: "What is the role of a broker in MQTT?"
  IMPROVED: "In MQTT, what function does broker serve?"
  
- ORIGINAL: "What are the three QoS levels in MQTT and their guarantees?"
  IMPROVED (SPLIT INTO THREE):
    "What does MQTT QoS 0 guarantee?"
    "What does MQTT QoS 1 guarantee?"
    "What does MQTT QoS 2 guarantee?"
    
- ORIGINAL: "What is the purpose of retained messages in MQTT?"
  IMPROVED: "What function do retained messages serve in MQTT?"
  
- ORIGINAL: "What is the function of the Last Will and Testament (LWT) in MQTT?"
  IMPROVED: "What does MQTT's LWT notify?"
  
- ORIGINAL: "What is the default port for unencrypted MQTT communication?"
  IMPROVED: "What port does MQTT use for unencrypted traffic?"
  
- ORIGINAL: "What is the purpose of the Keep Alive interval in MQTT?"
  IMPROVED: "What does MQTT Keep Alive interval do?"
  
- ORIGINAL: "What is the hierarchical structure used to categorize messages in MQTT?"
  IMPROVED: "How are MQTT messages categorized?"

Return improved flashcards in this JSON format:
{
  "flashcards": [
    {
      "question": "In MQTT, what function does broker serve?",
      "answer": "Routes messages between publishers and subscribers.", 
      "concept": "concept name",
      "difficulty": "beginner|intermediate|advanced",
      "quality_factors": {
        "conceptual_atomicity": 1.0,
        "conciseness": 1.0,
        "standalone_context": 1.0,
        "requires_specific_answer": 1.0,
        "no_hints": 1.0,
        "technical_precision": 1.0,
        "practical_value": 0.9,
        "bidirectional_design": 0.8,
        "uniqueness": 1.0,
        "design_rationale": 0.7
      },
      "improvements": [
        "reduced question from 15 to 7 words",
        "reduced answer from 12 to 6 words",
        "added Azure context prefix",
        "made answer bidirectional",
        "enhanced technical precision"
      ]
    }
  ],
  "improvement_summary": {
    "cards_improved": 5,
    "average_quality_gain": 1.2,
    "average_word_reduction": 5.3,
    "quality_improvements": {
      "conceptual_atomicity": +0.2,
      "conciseness": +0.5,
      "standalone_context": +0.3,
      "requires_specific_answer": +0.1,
      "no_hints": +0.1,
      "technical_precision": +0.2,
      "practical_value": +0.3,
      "bidirectional_design": +0.4,
      "uniqueness": +0.1,
      "design_rationale": +0.2
    },
    "remaining_issues": 1
  }
}"""

        user_message = f"""Original Flashcards: {flashcards}
Quality Assessment: {assessment}
Topic: {topic}

Improve the flashcards based on the assessment feedback."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        return await self._call_azure_openai(messages)

    async def create_flashcards(
        self,
        topic: str,
        content: str = "",
        content_source: ContentSource = ContentSource.DIRECT_INPUT,
        content_file: str = "",
        difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE,
        max_iterations: int = 3,
        quality_threshold: float = 8.0
    ) -> List[QAFlashcard]:
        """
        Create high-quality flashcards with multiple content source options
        """
        
        self.logger.info(f"Creating flashcards for topic: {topic}")
        self.logger.info(f"Content source: {content_source.value}, Target difficulty: {difficulty.value}")
        
        # Handle different content sources
        final_content = ""
        
        if content_source == ContentSource.AUTO_GENERATE:
            self.logger.info("Auto-generating content...")
            final_content = await self._auto_generate_content(topic, difficulty.value)
            
        elif content_source == ContentSource.TEXT_FILE:
            if not content_file:
                raise ValueError("Content file must be specified when using TEXT_FILE source")
            self.logger.info(f"Reading content from file: {content_file}")
            final_content = self.content_manager.read_content_file(content_file)
            
        elif content_source == ContentSource.DIRECT_INPUT:
            if not content:
                raise ValueError("Content must be provided when using DIRECT_INPUT source")
            final_content = content
            
        else:  # PREDEFINED
            self.logger.info("Using predefined content...")
            final_content = self._get_predefined_content(topic)
        
        # New Wikipedia source
        if content_source == ContentSource.WIKIPEDIA:
            self.logger.info(f"Fetching content from Wikipedia for topic: {topic}")
            try:
                final_content = wikipedia.summary(topic, sentences=10)
            except Exception as e:
                self.logger.error(f"Failed to fetch Wikipedia content: {str(e)}")
                raise ValueError(f"Could not fetch Wikipedia content for '{topic}'")
        
        if not final_content.strip():
            raise ValueError("No content available for flashcard generation")
        
        self.logger.info(f"Content length: {len(final_content)} characters")
        
        # Initialize tracking metrics for iterative improvement
        feedback = "Initial creation - focus on high-quality, concise flashcards."
        flashcard_data = None
        overall_score = 0
        iteration_metrics = []
        quality_factors_history = {}
        
        try:
            for iteration in range(max_iterations):
                self.logger.info(f"Iteration {iteration + 1}/{max_iterations}")
                
                # Stage 1: Generate flashcards
                self.logger.info("Generating flashcards...")
                generation_response = await self._generate_flashcards(topic, final_content, difficulty.value, feedback)
                
                try:
                    flashcard_data = json.loads(generation_response)
                    cards_data = flashcard_data.get('flashcards', [])
                    self.logger.info(f"Generated {len(cards_data)} cards")
                    
                    if len(cards_data) == 0:
                        continue
                        
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse generated flashcards: {str(e)}")
                    continue

                # Stage 2: Assess quality
                self.logger.info("Assessing flashcard quality...")
                assessment_response = await self._assess_flashcards(json.dumps(flashcard_data), topic)
                
                try:
                    assessment = json.loads(assessment_response)
                    # Extract assessment metrics
                    overall_score = assessment.get('overall_score', 0)
                    critical_issues = assessment.get('critical_issues', 0)
                    quality_distribution = assessment.get('quality_distribution', {})
                    
                    # Detailed quality tracking
                    self.logger.info(f"Quality score: {overall_score:.1f}/10")
                    self.logger.info(f"Critical issues: {critical_issues}")
                    self.logger.info(f"Quality distribution: {quality_distribution}")
                    
                    # Track quality factors for individual cards
                    card_quality_factors = {}
                    assessments = assessment.get('assessments', [])
                    for card_assessment in assessments:
                        card_idx = card_assessment.get('card_index', 0)
                        factors = card_assessment.get('quality_factors', {})
                        weighted_score = card_assessment.get('weighted_score', 0)
                        card_quality_factors[card_idx] = {
                            'factors': factors,
                            'score': weighted_score
                        }
                    
                    # Store metrics for this iteration
                    iteration_metrics.append({
                        'iteration': iteration + 1,
                        'overall_score': overall_score,
                        'critical_issues': critical_issues,
                        'quality_distribution': quality_distribution,
                        'improvement_priorities': assessment.get('improvement_priorities', [])
                    })
                    
                    # Quality factors tracking
                    quality_factors_history[iteration] = card_quality_factors
                    
                    # Check if quality threshold is met across all criteria
                    quality_criteria_met = True
                    
                    # Calculate minimum acceptable scores for key factors
                    min_atomicity = 0.9    # Each card must cover exactly one concept
                    min_conciseness = 0.8   # Questions/answers must be concise
                    min_standalone = 0.9    # Questions must have context
                    min_no_hints = 0.9     # No hints in questions
                    
                    for card_assessment in assessments:
                        factors = card_assessment.get('quality_factors', {})
                        # Check critical quality factors
                        if (factors.get('conceptual_atomicity', 0) < min_atomicity or
                            factors.get('conciseness', 0) < min_conciseness or
                            factors.get('standalone_context', 0) < min_standalone or
                            factors.get('no_hints', 0) < min_no_hints):
                            quality_criteria_met = False
                            break
                    
                    if overall_score >= quality_threshold and critical_issues == 0 and quality_criteria_met:
                        self.logger.info("Quality threshold reached across all critical factors!")
                        break
                        
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse assessment: {str(e)}")
                    continue

                # Stage 3: Refine flashcards
                self.logger.info("Refining flashcards...")
                refinement_response = await self._refine_flashcards(
                    json.dumps(flashcard_data), json.dumps(assessment), topic
                )
                
                try:
                    refined_data = json.loads(refinement_response)
                    improvement_summary = refined_data.get('improvement_summary', {})
                    
                    # Enhanced refinement tracking
                    cards_improved = improvement_summary.get('cards_improved', 0)
                    quality_gain = improvement_summary.get('average_quality_gain', 0)
                    word_reduction = improvement_summary.get('average_word_reduction', 0)
                    quality_improvements = improvement_summary.get('quality_improvements', {});
                    
                    self.logger.info(f"Cards improved: {cards_improved}")
                    self.logger.info(f"Quality gain: +{quality_gain:.1f}")
                    self.logger.info(f"Word reduction: {word_reduction:.1f} words/card")
                    self.logger.info(f"Factor improvements: {quality_improvements}");
                    
                    # Update for next iteration
                    flashcard_data = refined_data;
                    # Use prioritized improvement areas for targeted feedback
                    priorities = assessment.get('improvement_priorities', []);
                    if priorities:
                        feedback = f"Focus on: {'; '.join(priorities[:2])}"
                    else:
                        feedback = "Continue improving quality with focus on technical precision and bidirectional design."
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse refinement: {str(e)}")
                    break

            # Convert to QAFlashcard objects
            final_cards = []
            if flashcard_data:
                cards_data = flashcard_data.get('flashcards', [])
                
                for i, card_data in enumerate(cards_data):
                    # Extract quality factors for final cards
                    quality_factors = card_data.get('quality_factors', {})
                    # Calculate quality score based on weighted factors
                    card_quality = 0
                    if quality_factors:
                        weights = {
                            'conceptual_atomicity': 0.15,
                            'conciseness': 0.15,
                            'standalone_context': 0.1,
                            'requires_specific_answer': 0.1,
                            'no_hints': 0.1,
                            'technical_precision': 0.1,
                            'practical_value': 0.1,
                            'bidirectional_design': 0.1,
                            'uniqueness': 0.05,
                            'design_rationale': 0.05
                        }
                        
                        card_quality = sum(quality_factors.get(factor, 0) * weight 
                                        for factor, weight in weights.items()) * 10
                    else:
                        card_quality = overall_score
                    
                    flashcard = QAFlashcard(
                        question=card_data.get('question', ''),
                        answer=card_data.get('answer', ''),
                        topic=topic,
                        difficulty=card_data.get('difficulty', difficulty.value),
                        quality_score=card_quality,
                        concept=card_data.get('concept', ''),
                        id=f"{topic.lower().replace(' ', '_')}_{i+1:03d}"
                    )
                    final_cards.append(flashcard)
            
            # Log final quality metrics
            if iteration_metrics:
                final_metrics = iteration_metrics[-1]
                self.logger.info(f"Final quality metrics:")
                self.logger.info(f"Overall score: {final_metrics['overall_score']:.1f}/10")
                self.logger.info(f"Quality distribution: {final_metrics['quality_distribution']}")
            
            self.logger.info(f"Successfully created {len(final_cards)} flashcards")
            return final_cards
            
        except Exception as e:
            self.logger.error(f"Error during flashcard creation: {str(e)}")
            raise

    def _get_predefined_content(self, topic: str) -> str:
        """Get predefined content for common topics"""
        predefined_topics = {
            "azure openai": "azure_openai.txt",
            "azure functions": "azure_functions.txt"
        }
        
        topic_lower = topic.lower()
        for key, filename in predefined_topics.items():
            if key in topic_lower:
                try:
                    return self.content_manager.read_content_file(filename)
                except FileNotFoundError:
                    pass
        
        return f"Please provide content about {topic} to generate flashcards."

    def export_to_csv(self, flashcards: List[QAFlashcard], filename: str = None) -> str:
        """Export flashcards to CSV format"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"azure_ai_flashcards_{timestamp}.csv"
        
        filepath = os.path.join(os.getcwd(), filename)
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                if flashcards:
                    fieldnames = list(flashcards[0].to_dict().keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for flashcard in flashcards:
                        writer.writerow(flashcard.to_dict())
                
            self.logger.info(f"Exported {len(flashcards)} flashcards to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to export CSV: {str(e)}")
            raise

    def print_flashcards(self, flashcards: List[QAFlashcard]):
        """Print flashcards in a readable format"""
        
        print(f"\n🎯 AZURE AI FLASHCARDS")
        print(f"{'='*60}")
        print(f"Total Cards: {len(flashcards)}")
        
        if flashcards:
            avg_quality = sum(card.quality_score for card in flashcards) / len(flashcards)
            print(f"Average Quality: {avg_quality:.1f}/10")
            
            difficulty_counts = {}
            for card in flashcards:
                difficulty_counts[card.difficulty] = difficulty_counts.get(card.difficulty, 0) + 1
            
            print(f"Difficulty Distribution: {difficulty_counts}")
        
        print(f"\n📚 FLASHCARD DETAILS:")
        for i, card in enumerate(flashcards, 1):
            print(f"\nCard {i} ({card.difficulty}):")
            print(f"  Q: {card.question}")
            print(f"  A: {card.answer}")
            print(f"  Concept: {card.concept}")
            print(f"  Quality: {card.quality_score:.1f}/10")


def get_user_input() -> Dict:
    """Interactive user input for flashcard generation"""
    
    print("\n🎯 AZURE AI FLASHCARD GENERATOR")
    print("=" * 50)
    
    # Topic input
    topic = input("Enter the topic for flashcards: ").strip()
    if not topic:
        topic = "Azure OpenAI Service"
    
    # Content source selection
    print("\nContent Source Options:")
    print("1. Auto-generate content (AI creates content for you)")
    print("2. Read from text file")
    print("3. Paste content directly")
    print("4. Use predefined content")
    print("5. Fetch content from Wikipedia")
    
    source_choice = input("Choose content source (1-4): ").strip()
    
    content_source_map = {
        "1": ContentSource.AUTO_GENERATE,
        "2": ContentSource.TEXT_FILE,
        "3": ContentSource.DIRECT_INPUT,
        "4": ContentSource.PREDEFINED,
        "5": ContentSource.WIKIPEDIA
    }
    
    content_source = content_source_map.get(source_choice, ContentSource.AUTO_GENERATE)
    content = ""
    content_file = ""

    if content_source == ContentSource.WIKIPEDIA:
        print(f"\n✨ Will fetch content from Wikipedia for '{topic}'")
    
    if content_source == ContentSource.AUTO_GENERATE:
        print(f"\n✨ AI will auto-generate content about '{topic}'")
        
    elif content_source == ContentSource.TEXT_FILE:
        # Show available files
        content_manager = ContentManager()
        available_files = content_manager.list_available_files()
        
        print("\nAvailable content files:")
        print(f"Content directory: {content_manager.get_content_dir_path()}")
        for i, filename in enumerate(available_files, 1):
            print(f"{i}. {filename}")
        
        print(f"\n💡 Tip: You can add your own .txt files to the content directory")
        print(f"   Or edit existing files like 'custom_content.txt'")
        
        file_choice = input(f"\nChoose file (1-{len(available_files)}) or enter filename: ").strip()
        
        try:
            file_idx = int(file_choice) - 1
            if 0 <= file_idx < len(available_files):
                content_file = available_files[file_idx]
            else:
                content_file = file_choice
        except ValueError:
            content_file = file_choice
            
        if not content_file.endswith('.txt'):
            content_file += '.txt'
            
    elif content_source == ContentSource.DIRECT_INPUT:
        print("\nPaste your content (press Enter twice when done):")
        lines = []
        empty_line_count = 0
        while True:
            line = input()
            if line == "":
                empty_line_count += 1
                if empty_line_count >= 2:
                    break
            else:
                empty_line_count = 0
            lines.append(line)
        content = "\n".join(lines[:-1])  # Remove last empty line
    
    # Difficulty selection
    print("\nDifficulty Levels:")
    print("1. Beginner")
    print("2. Intermediate")
    print("3. Advanced")
    
    diff_choice = input("Choose difficulty (1-3, default: 2): ").strip()
    difficulty_map = {
        "1": DifficultyLevel.BEGINNER,
        "2": DifficultyLevel.INTERMEDIATE,
        "3": DifficultyLevel.ADVANCED
    }
    difficulty = difficulty_map.get(diff_choice, DifficultyLevel.INTERMEDIATE)
    
    # Quality settings
    quality_threshold = input("Quality threshold (1-10, default: 8.0): ").strip()

    return {
        "topic": topic,
        "content_source": content_source,
        "content": content,
        "content_file": content_file,
        "difficulty": difficulty,
        "quality_threshold": float(quality_threshold) if quality_threshold else 8.0
    }


def check_azure_configuration() -> bool:
    """Check Azure OpenAI configuration and connectivity"""
    print("🔍 AZURE OPENAI CONFIGURATION CHECK")
    print("=" * 50)
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "o4-mini")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    status = True
    if endpoint:
        print(f"Endpoint: ✅ {endpoint}")
    else:
        print("Endpoint: ❌ Missing AZURE_OPENAI_ENDPOINT")
        status = False
    if api_key:
        print(f"API Key: ✅ {'*' * 4 + api_key[-4:]}")
    else:
        print("API Key: ❌ Missing AZURE_OPENAI_API_KEY")
        status = False
    print(f"Deployment Name: ✅ {deployment}")
    print(f"API Version: ✅ {api_version}")

    if not status:
        print("\n❌ Configuration incomplete. Please set the missing environment variables and retry.")
    else:
        print("\n✅ Configuration looks good!")
    return status


async def create_auto_generated_flashcards():
    """Demo: Auto-generate content and create flashcards"""
    system = SimplifiedFlashcardSystem()
    flashcards = await system.create_flashcards(
        topic="Azure Auto Topic",
        content_source=ContentSource.AUTO_GENERATE,
        difficulty=DifficultyLevel.INTERMEDIATE
    )
    system.print_flashcards(flashcards)
    return flashcards

async def create_file_based_flashcards():
    """Demo: Create flashcards from text file"""
    system = SimplifiedFlashcardSystem()
    flashcards = await system.create_flashcards(
        topic="Azure File Topic",
        content_source=ContentSource.TEXT_FILE,
        content_file="azure_openai.txt",
        difficulty=DifficultyLevel.INTERMEDIATE
    )
    system.print_flashcards(flashcards)
    return flashcards

async def interactive_flashcard_generator():
    """Interactive flashcard generation"""
    config = get_user_input()
    system = SimplifiedFlashcardSystem()
    flashcards = await system.create_flashcards(
        topic=config["topic"],
        content_source=config["content_source"],
        content_file=config.get("content_file", ""),
        content=config.get("content", ""),
        difficulty=config["difficulty"],
        max_iterations=config.get("max_iterations", 3),
        quality_threshold=config.get("quality_threshold", 8.0)
    )
    system.print_flashcards(flashcards)
    system.export_to_csv(flashcards)
    return flashcards

if __name__ == "__main__":
    import sys
    # Check Azure configuration
    if not check_azure_configuration():
        sys.exit(1)
    # Determine mode based on command-line args
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "--auto":
            print("🚀 Auto-generating flashcards...")
            asyncio.run(create_auto_generated_flashcards())
        elif mode == "--file":
            print("🚀 Creating flashcards from file...")
            asyncio.run(create_file_based_flashcards())
        elif mode == "--check":
            check_azure_configuration()
        elif mode in ("--help", "-h"):  # help
            print("Usage: python flashcard-generator.py [--auto|--file|--check]")
        else:
            print(f"Unknown option: {mode}. Use --help for usage.")
    else:
        # Interactive mode: show prompts
        asyncio.run(interactive_flashcard_generator())