"""
Demo script for the simple flashcard generator
Shows different ways to use the generator
"""

import asyncio
from simple_flashcard_generator import SimpleFlashcardGenerator


async def demo_with_azure_content():
    """Demo: Generate flashcards about Azure OpenAI"""
    
    content = """
Azure OpenAI Service provides REST API access to OpenAI's powerful language models including GPT-4, GPT-3.5-turbo, and Embeddings models.

Key Features:
- Multiple model deployments with custom names
- Enterprise-grade security and compliance
- Content filtering and safety features
- Token-based pricing and quota management
- Integration with Azure security and identity

Authentication:
- Uses Azure Active Directory for secure access
- API keys for programmatic access
- Managed identity support for Azure resources

Important Parameters:
- Temperature: Controls randomness (0.0-2.0)
- Max tokens: Maximum response length
- Top-p: Nucleus sampling parameter
- Frequency penalty: Reduces repetition
- Presence penalty: Encourages topic diversity

Deployments can be managed through Azure portal or REST APIs.
Content filtering policies help ensure responsible AI use.
"""
    
    print("🚀 Demo: Generating flashcards about Azure OpenAI")
    print("=" * 60)
    
    try:
        generator = SimpleFlashcardGenerator()
        flashcards = await generator.generate_flashcards(
            topic="Azure OpenAI Service",
            content=content,
            difficulty="intermediate"
        )
        
        generator.print_flashcards(flashcards)
        filename = generator.export_to_csv(flashcards, "demo_azure_openai_flashcards.csv")
        print(f"✅ Demo completed! Exported to: {filename}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


async def demo_with_custom_content():
    """Demo: Generate flashcards from custom content"""
    
    content = """
Python asyncio is a library for writing concurrent code using async/await syntax.

Key Concepts:
- Event loop: The core of asyncio that runs asynchronous tasks
- Coroutines: Functions defined with async def
- Tasks: Wrapped coroutines that can be scheduled
- Futures: Objects representing eventual results

Common Functions:
- asyncio.run(): Run an async function
- asyncio.create_task(): Schedule a coroutine
- asyncio.gather(): Run multiple coroutines concurrently
- asyncio.sleep(): Async version of time.sleep()

Benefits:
- Better performance for I/O-bound operations
- Cleaner code than traditional threading
- Built-in support for timeouts and cancellation
"""
    
    print("\n🚀 Demo: Generating flashcards about Python asyncio")
    print("=" * 60)
    
    try:
        generator = SimpleFlashcardGenerator()
        flashcards = await generator.generate_flashcards(
            topic="Python asyncio",
            content=content,
            difficulty="beginner"
        )
        
        generator.print_flashcards(flashcards)
        filename = generator.export_to_csv(flashcards, "demo_python_asyncio_flashcards.csv")
        print(f"✅ Demo completed! Exported to: {filename}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


async def run_all_demos():
    """Run all demo scenarios"""
    print("🎯 Simple Flashcard Generator - Demo Scenarios")
    print("=" * 70)
    
    await demo_with_azure_content()
    await demo_with_custom_content()
    
    print("\n🎉 All demos completed!")


if __name__ == "__main__":
    asyncio.run(run_all_demos())
