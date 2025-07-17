"""
Simple Azure OpenAI Flashcard Generator
Core functionality: Generate Q&A flashcards and export to CSV
"""

import os
import csv
import json
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict
from openai import AsyncAzureOpenAI


@dataclass
class Flashcard:
    """Simple flashcard data structure"""
    question: str
    answer: str
    topic: str
    difficulty: str
    concept: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export"""
        return {
            'question': self.question,
            'answer': self.answer,
            'topic': self.topic,
            'difficulty': self.difficulty,
            'concept': self.concept,
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


class SimpleFlashcardGenerator:
    """Simple flashcard generator using Azure OpenAI"""
    
    def __init__(self):
        # Setup logging with debug level
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize Azure OpenAI client
        self.client = self._initialize_azure_client()
    
    def _initialize_azure_client(self):
        """Initialize Azure OpenAI client with error handling"""
        try:
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            
            if not endpoint or not api_key:
                raise ValueError(
                    "Missing Azure OpenAI configuration. Please set:\n"
                    "- AZURE_OPENAI_ENDPOINT\n"
                    "- AZURE_OPENAI_API_KEY"
                )
            
            client = AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version="2024-12-01-preview"
            )
            
            self.logger.info("Azure OpenAI client initialized successfully")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise
    
    async def _call_azure_openai(self, messages: List[Dict]) -> str:
        """Make API call to Azure OpenAI"""
        try:
            deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o")
            
            self.logger.info(f"Making API call to deployment: {deployment_name}")
            
            response = await self.client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                max_tokens=2000,
                temperature=0.1
            )
            
            self.logger.info(f"Response received. Choices count: {len(response.choices)}")
            
            if not response.choices:
                raise ValueError("No choices in response from Azure OpenAI")
            
            content = response.choices[0].message.content
            self.logger.info(f"Content received: {len(content) if content else 0} characters")
            self.logger.debug(f"Content preview: {content[:200] if content else 'None'}...")
            
            if not content:
                raise ValueError("Empty content in response from Azure OpenAI")
            
            return content
            
        except Exception as e:
            self.logger.error(f"Azure OpenAI API call failed: {e}")
            raise
    
    async def generate_flashcards(self, topic: str, content: str, difficulty: str = "intermediate") -> List[Flashcard]:
        """Generate flashcards from content"""
        self.logger.info(f"Generating flashcards for topic: {topic}")
        
        # Create prompt for flashcard generation
        system_prompt = """You are an expert at creating educational flashcards. 

Create 5-8 high-quality Q&A flashcards from the provided content.

RULES:
1. Keep questions concise (under 15 words)
2. Keep answers specific and clear (under 20 words)
3. Focus on key concepts and practical knowledge
4. Make questions standalone (include context)
5. Ensure each flashcard covers a unique concept

Return JSON in this exact format:
{
  "flashcards": [
    {
      "question": "What does Azure OpenAI provide?",
      "answer": "REST API access to OpenAI language models",
      "concept": "Azure OpenAI Service"
    }
  ]
}"""
        
        user_prompt = f"""Topic: {topic}
Difficulty: {difficulty}
Content: {content}

Generate flashcards from this content."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self._call_azure_openai(messages)
            
            # Log the raw response for debugging
            self.logger.info(f"Raw response length: {len(response)}")
            self.logger.debug(f"Raw response: {response}")
            
            # Parse JSON response
            if not response.strip():
                raise ValueError("Empty response from Azure OpenAI")
            
            flashcard_data = json.loads(response)
            flashcards = []
            
            if "flashcards" not in flashcard_data:
                self.logger.error(f"No 'flashcards' key in response: {flashcard_data}")
                raise ValueError("Invalid response format: missing 'flashcards' key")
            
            for card_data in flashcard_data.get("flashcards", []):
                flashcard = Flashcard(
                    question=card_data.get("question", ""),
                    answer=card_data.get("answer", ""),
                    topic=topic,
                    difficulty=difficulty,
                    concept=card_data.get("concept", "")
                )
                flashcards.append(flashcard)
            
            self.logger.info(f"Generated {len(flashcards)} flashcards")
            return flashcards
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.error(f"Response that failed to parse: '{response[:500]}...'")
            raise ValueError(f"Invalid JSON response from Azure OpenAI: {e}")
        except Exception as e:
            self.logger.error(f"Error generating flashcards: {e}")
            raise
    
    def export_to_csv(self, flashcards: List[Flashcard], filename: str = None) -> str:
        """Export flashcards to CSV file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flashcards_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                if flashcards:
                    fieldnames = list(flashcards[0].to_dict().keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for flashcard in flashcards:
                        writer.writerow(flashcard.to_dict())
            
            self.logger.info(f"Exported {len(flashcards)} flashcards to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to export CSV: {e}")
            raise
    
    def print_flashcards(self, flashcards: List[Flashcard]):
        """Print flashcards to console"""
        print(f"\n=== FLASHCARDS: {flashcards[0].topic if flashcards else 'None'} ===")
        print(f"Total: {len(flashcards)} cards\n")
        
        for i, card in enumerate(flashcards, 1):
            print(f"Card {i}:")
            print(f"  Q: {card.question}")
            print(f"  A: {card.answer}")
            print(f"  Concept: {card.concept}")
            print()


async def main():
    """Main function for interactive use"""
    # Check environment variables
    if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
        print("❌ Missing Azure OpenAI configuration!")
        print("Please set these environment variables:")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_API_KEY")
        print("- AZURE_OPENAI_CHAT_DEPLOYMENT_NAME (optional, defaults to 'o4-mini')")
        return
    
    print("🎯 Simple Azure OpenAI Flashcard Generator")
    print("=" * 50)
    
    # Get user input
    topic = input("Enter topic: ").strip() or "Azure OpenAI"
    difficulty = input("Enter difficulty (beginner/intermediate/advanced): ").strip() or "intermediate"
    
    print("\nEnter content (paste your text, then press Enter twice):")
    content_lines = []
    empty_count = 0
    while True:
        line = input()
        if line == "":
            empty_count += 1
            if empty_count >= 2:
                break
        else:
            empty_count = 0
        content_lines.append(line)
    
    content = "\n".join(content_lines[:-1])  # Remove last empty line
    
    if not content.strip():
        print("❌ No content provided!")
        return
    
    try:
        # Generate flashcards
        generator = SimpleFlashcardGenerator()
        flashcards = await generator.generate_flashcards(topic, content, difficulty)
        
        # Display results
        generator.print_flashcards(flashcards)
        
        # Export to CSV
        filename = generator.export_to_csv(flashcards)
        print(f"✅ Flashcards exported to: {filename}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
