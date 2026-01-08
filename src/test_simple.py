"""
Test script for the High-Quality Anki Flashcard Generator

Tests:
- Environment configuration
- Basic flashcard generation
- Multiple card types
- Quality metrics calculation
- Export functionality
"""

import os
import asyncio


def check_environment():
    """Verify Azure OpenAI environment configuration"""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o")

    print("=" * 50)
    print(" ENVIRONMENT CHECK")
    print("=" * 50)

    print(f"\n  AZURE_OPENAI_ENDPOINT: {'SET' if endpoint else 'MISSING'}")
    if endpoint:
        # Show partial URL for verification
        print(f"    → {endpoint[:30]}..." if len(endpoint) > 30 else f"    → {endpoint}")

    print(f"\n  AZURE_OPENAI_API_KEY: {'SET' if api_key else 'MISSING'}")
    if api_key:
        print(f"    → ...{api_key[-4:]}")

    print(f"\n  AZURE_OPENAI_CHAT_DEPLOYMENT_NAME: {deployment}")

    print(f"\n  Optional settings:")
    print(f"    MAX_TOKENS: {os.getenv('AZURE_OPENAI_MAX_TOKENS', '4000 (default)')}")
    print(f"    TEMPERATURE: {os.getenv('AZURE_OPENAI_TEMPERATURE', '0.2 (default)')}")
    print(f"    API_VERSION: {os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview (default)')}")

    return bool(endpoint and api_key)


async def test_flashcard_generation():
    """Test basic flashcard generation with quality metrics"""

    # Sample content with terminology and lists
    # This tests the generator's ability to create atomic cards
    test_content = """
REST API Design Principles:

REST (Representational State Transfer) is an architectural style for
designing networked applications.

HTTP Methods:
- GET: Retrieve resources (idempotent, safe)
- POST: Create new resources
- PUT: Update/replace entire resource (idempotent)
- PATCH: Partial resource update
- DELETE: Remove resource (idempotent)

Status Codes:
- 2xx Success: 200 OK, 201 Created, 204 No Content
- 4xx Client Error: 400 Bad Request, 401 Unauthorized, 404 Not Found
- 5xx Server Error: 500 Internal Server Error, 503 Service Unavailable

Best Practices:
- Use nouns for resource names, not verbs
- Version your API (e.g., /v1/users)
- Return appropriate status codes
- Use pagination for large collections
- Implement proper error responses with details
"""

    print("\n" + "=" * 50)
    print(" FLASHCARD GENERATION TEST")
    print("=" * 50)

    try:
        from simple_flashcard_generator import (
            FlashcardGenerator,
            CardType,
            Difficulty
        )

        generator = FlashcardGenerator()

        # Test with multiple card types
        print("\n  Generating flashcards...")
        flashcards = await generator.generate_flashcards(
            content=test_content,
            topic="REST APIs",
            difficulty=Difficulty.INTERMEDIATE,
            card_types=[CardType.BASIC, CardType.CLOZE, CardType.REVERSE]
        )

        print(f"  Generated {len(flashcards)} cards")

        # Verify card types were generated
        types_found = set(c.card_type for c in flashcards)
        print(f"  Card types: {[t.value for t in types_found]}")

        # Check quality metrics
        qualities = [c.quality.overall_score for c in flashcards if c.quality]
        if qualities:
            avg_quality = sum(qualities) / len(qualities)
            min_quality = min(qualities)
            max_quality = max(qualities)
            print(f"  Quality scores: avg={avg_quality:.2f}, min={min_quality:.2f}, max={max_quality:.2f}")

        # Test export
        csv_file = generator.export_to_csv(flashcards, "test_flashcards.csv")
        print(f"  Exported CSV: {csv_file}")

        anki_files = generator.export_to_anki_csv(flashcards, "test_anki.csv")
        print(f"  Exported Anki: {anki_files}")

        # Display sample cards
        print("\n  Sample cards:")
        for i, card in enumerate(flashcards[:3], 1):
            print(f"\n  Card {i} [{card.card_type.value}]:")
            print(f"    Q: {card.question[:60]}...")
            if card.cloze_text:
                print(f"    Cloze: {card.cloze_text[:60]}...")
            else:
                print(f"    A: {card.answer[:60]}...")
            if card.quality:
                print(f"    Quality: {card.quality.overall_score:.1%}")

        generator.print_statistics()

        print("\n  TEST PASSED")
        return True

    except ImportError as e:
        print(f"\n  IMPORT ERROR: {e}")
        print("  Make sure you're running from the src directory")
        return False

    except Exception as e:
        print(f"\n  TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_wikipedia_integration():
    """Test Wikipedia content fetching"""

    print("\n" + "=" * 50)
    print(" WIKIPEDIA INTEGRATION TEST")
    print("=" * 50)

    try:
        from simple_flashcard_generator import FlashcardGenerator, Difficulty

        generator = FlashcardGenerator()

        print("\n  Fetching from Wikipedia: 'Spaced repetition'...")
        flashcards = await generator.generate_from_wikipedia(
            topic="Spaced repetition",
            difficulty=Difficulty.BEGINNER,
            sentences=8
        )

        print(f"  Generated {len(flashcards)} cards")
        if flashcards and flashcards[0].source:
            print(f"  Source: {flashcards[0].source}")

        print("\n  TEST PASSED")
        return True

    except ImportError as e:
        print(f"\n  Wikipedia library not available: {e}")
        print("  Install with: pip install wikipedia")
        return False

    except Exception as e:
        print(f"\n  TEST FAILED: {e}")
        return False


async def run_tests():
    """Run all tests"""

    print("""
╔═══════════════════════════════════════════════════════════════╗
║   HIGH-QUALITY ANKI FLASHCARD GENERATOR - TEST SUITE         ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    # Check environment first
    if not check_environment():
        print("\n" + "=" * 50)
        print(" CONFIGURATION REQUIRED")
        print("=" * 50)
        print("\nSet these environment variables before testing:\n")
        print("  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'")
        print("  export AZURE_OPENAI_API_KEY='your-api-key'")
        print("  export AZURE_OPENAI_CHAT_DEPLOYMENT_NAME='gpt-4o'  # optional")
        return

    results = {}

    # Run tests
    results['generation'] = await test_flashcard_generation()
    results['wikipedia'] = await test_wikipedia_integration()

    # Summary
    print("\n" + "=" * 50)
    print(" TEST SUMMARY")
    print("=" * 50)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")


if __name__ == "__main__":
    asyncio.run(run_tests())
