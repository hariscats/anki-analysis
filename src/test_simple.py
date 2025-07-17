"""
Quick test script for the simple flashcard generator
"""

import os
import asyncio

# Check if environment variables are set
def check_environment():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o")
    
    print("🔍 Environment Check:")
    print(f"Endpoint: {'✅ Set' if endpoint else '❌ Missing'}")
    if endpoint:
        print(f"  URL: {endpoint}")
    print(f"API Key: {'✅ Set' if api_key else '❌ Missing'}")
    if api_key:
        print(f"  Key: ...{api_key[-4:] if len(api_key) > 4 else api_key}")
    print(f"Deployment: ✅ {deployment}")
    
    return bool(endpoint and api_key)

async def test_simple_generator():
    """Test the simple flashcard generator"""
    
    if not check_environment():
        print("\n❌ Please set Azure OpenAI environment variables first!")
        return
    
    print("\n🧪 Testing Simple Flashcard Generator")
    print("=" * 50)
    
    # Sample content for testing
    test_content = """
Azure Functions is a serverless compute service that lets you run event-triggered code without managing infrastructure.

Key Features:
- Pay only for execution time
- Automatic scaling based on demand
- Multiple programming languages supported
- Integration with Azure services
- Event-driven architecture

Trigger Types:
- HTTP triggers for REST APIs
- Timer triggers for scheduled tasks
- Blob storage triggers for file processing
- Queue triggers for message processing
"""
    
    try:
        # Import here to avoid issues if environment not set
        from simple_flashcard_generator import SimpleFlashcardGenerator
        
        generator = SimpleFlashcardGenerator()
        flashcards = await generator.generate_flashcards(
            topic="Azure Functions",
            content=test_content,
            difficulty="intermediate"
        )
        
        print("✅ Flashcards generated successfully!")
        generator.print_flashcards(flashcards)
        
        filename = generator.export_to_csv(flashcards, "test_flashcards.csv")
        print(f"✅ Exported to: {filename}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_simple_generator())
