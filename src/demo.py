"""
Demo script for the High-Quality Anki Flashcard Generator

Demonstrates:
- Multiple card types (basic, cloze, reverse, concept)
- Different difficulty levels
- Wikipedia integration
- Quality metrics and analysis
- Proper Anki export formats
"""

import asyncio
import os
from simple_flashcard_generator import (
    FlashcardGenerator,
    CardType,
    Difficulty
)


async def demo_knowledge_formulation_principles():
    """
    Demo: Shows how the generator applies SuperMemo's 20 Rules

    Key principles demonstrated:
    1. Atomic cards (one fact per card)
    2. Cloze deletions for factual recall
    3. Breaking down enumerations
    4. Context in questions
    5. Mnemonic hooks
    """

    # This content intentionally includes lists and complex concepts
    # to demonstrate how the generator breaks them into atomic cards
    content = """
Kubernetes Architecture:

The Control Plane consists of several components:
- API Server: The front end for the Kubernetes control plane
- etcd: Consistent and highly-available key-value store for cluster data
- Scheduler: Watches for newly created Pods with no assigned node
- Controller Manager: Runs controller processes

Node components run on every node:
- kubelet: Agent ensuring containers are running in a Pod
- kube-proxy: Network proxy maintaining network rules on nodes
- Container runtime: Software responsible for running containers (Docker, containerd)

Key Concepts:
- Pod: Smallest deployable unit, one or more containers
- Service: Abstract way to expose an application running on Pods
- Deployment: Provides declarative updates for Pods and ReplicaSets
- Namespace: Virtual clusters backed by the same physical cluster
- ConfigMap: API object to store non-confidential data in key-value pairs
- Secret: Similar to ConfigMap but for sensitive information

kubectl commands:
- kubectl get pods: List all pods
- kubectl describe pod <name>: Detailed pod info
- kubectl logs <pod>: View pod logs
- kubectl apply -f <file>: Apply configuration
"""

    print("=" * 70)
    print(" DEMO: Knowledge Formulation Principles")
    print(" Showing how lists/enumerations become atomic cloze cards")
    print("=" * 70)

    generator = FlashcardGenerator()

    flashcards = await generator.generate_flashcards(
        content=content,
        topic="Kubernetes",
        difficulty=Difficulty.INTERMEDIATE,
        card_types=[CardType.BASIC, CardType.CLOZE, CardType.CONCEPT_EXAMPLE],
        custom_instructions="Break down all lists into individual cloze cards. Each component should have its own card."
    )

    generator.print_flashcards(flashcards)

    # Export in Anki format
    files = generator.export_to_anki_csv(flashcards, "demo_kubernetes.csv")
    print(f"Exported Anki files: {files}")

    # Show quality statistics
    if flashcards:
        cloze_count = sum(1 for c in flashcards if c.card_type == CardType.CLOZE)
        basic_count = sum(1 for c in flashcards if c.card_type == CardType.BASIC)
        concept_count = sum(1 for c in flashcards if c.card_type == CardType.CONCEPT_EXAMPLE)

        print(f"\nCard Type Distribution:")
        print(f"  Cloze: {cloze_count} (best for facts)")
        print(f"  Basic: {basic_count} (best for explanations)")
        print(f"  Concept: {concept_count} (with examples)")

    generator.print_statistics()
    return flashcards


async def demo_difficulty_levels():
    """
    Demo: Shows how difficulty affects card generation

    - Beginner: Focus on terminology, more mnemonics
    - Intermediate: Balance concepts and applications
    - Advanced: Nuances, edge cases, comparisons
    """

    content = """
Docker Containers:

Docker is a platform for developing, shipping, and running applications in containers.
Containers are lightweight, portable, and isolated environments that package applications
with their dependencies.

Key benefits:
- Consistency across environments
- Faster deployment and scaling
- Resource efficiency compared to VMs
- Microservices architecture support

Docker architecture:
- Docker Engine: Core runtime
- Docker Images: Read-only templates
- Docker Containers: Running instances of images
- Docker Registry: Storage for images (Docker Hub)
- Dockerfile: Script to build images

Common commands:
- docker build: Create image from Dockerfile
- docker run: Create and start container
- docker ps: List running containers
- docker pull: Download image from registry
"""

    print("\n" + "=" * 70)
    print(" DEMO: Difficulty Levels Comparison")
    print("=" * 70)

    generator = FlashcardGenerator()

    for difficulty in [Difficulty.BEGINNER, Difficulty.ADVANCED]:
        print(f"\n--- {difficulty.value.upper()} LEVEL ---\n")

        flashcards = await generator.generate_flashcards(
            content=content,
            topic="Docker",
            difficulty=difficulty,
            card_types=[CardType.BASIC, CardType.CLOZE]
        )

        # Show first 3 cards to illustrate the difference
        for card in flashcards[:3]:
            print(f"  [{card.card_type.value}] Q: {card.question}")
            if card.cloze_text:
                print(f"         Cloze: {card.cloze_text}")
            else:
                print(f"         A: {card.answer}")
            if card.mnemonic_hint:
                print(f"         Mnemonic: {card.mnemonic_hint}")
            print()

    generator.print_statistics()


async def demo_wikipedia_integration():
    """
    Demo: Generate flashcards directly from Wikipedia

    Shows automatic source attribution and content fetching.
    """

    print("\n" + "=" * 70)
    print(" DEMO: Wikipedia Integration")
    print("=" * 70)

    try:
        generator = FlashcardGenerator()

        # Generate cards from Wikipedia article
        flashcards = await generator.generate_from_wikipedia(
            topic="Python (programming language)",
            difficulty=Difficulty.BEGINNER,
            sentences=15
        )

        generator.print_flashcards(flashcards, show_quality=False)

        # Show source attribution
        if flashcards and flashcards[0].source:
            print(f"\nSource: {flashcards[0].source}")

        files = generator.export_to_anki_csv(flashcards, "demo_python_wiki.csv")
        print(f"Exported: {files}")

        generator.print_statistics()

    except ImportError:
        print("Wikipedia library not installed. Run: pip install wikipedia")
    except Exception as e:
        print(f"Wikipedia demo failed: {e}")


async def demo_reverse_cards():
    """
    Demo: Reverse cards for terminology learning

    Automatically generates both:
    - Term → Definition
    - Definition → Term
    """

    content = """
Machine Learning Terminology:

Supervised Learning: Training with labeled data where the algorithm learns
to map inputs to known outputs.

Unsupervised Learning: Training with unlabeled data where the algorithm
finds patterns and structures on its own.

Overfitting: When a model learns the training data too well, including noise,
performing poorly on new data.

Underfitting: When a model is too simple to capture the underlying patterns
in the data.

Gradient Descent: Optimization algorithm that iteratively adjusts parameters
to minimize the loss function.

Feature Engineering: Process of creating new input features from existing
data to improve model performance.

Cross-Validation: Technique for evaluating model performance by splitting
data into multiple training and validation sets.
"""

    print("\n" + "=" * 70)
    print(" DEMO: Reverse Cards for Terminology")
    print(" Each term generates TWO cards (term→def and def→term)")
    print("=" * 70)

    generator = FlashcardGenerator()

    flashcards = await generator.generate_flashcards(
        content=content,
        topic="Machine Learning",
        difficulty=Difficulty.INTERMEDIATE,
        card_types=[CardType.REVERSE],  # Only reverse cards
        custom_instructions="Create reverse cards for each term-definition pair."
    )

    generator.print_flashcards(flashcards, show_quality=False)

    # Count pairs
    reverse_count = sum(1 for c in flashcards if 'reverse' in c.tags)
    print(f"\nGenerated {len(flashcards)} total cards ({reverse_count} reverse pairs)")

    generator.print_statistics()


async def run_all_demos():
    """Run all demonstration scenarios"""

    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║     HIGH-QUALITY ANKI FLASHCARD GENERATOR - DEMONSTRATION            ║
║                                                                       ║
║  Based on SuperMemo's 20 Rules of Formulating Knowledge               ║
║  - Atomic cards (one fact per card)                                  ║
║  - Cloze deletions for factual recall                               ║
║  - Breaking down lists into individual cards                         ║
║  - Quality metrics and interference detection                        ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)

    # Check environment
    if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
        print("❌ Azure OpenAI credentials not set!")
        print("\nSet these environment variables:")
        print("  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'")
        print("  export AZURE_OPENAI_API_KEY='your-key'")
        print("  export AZURE_OPENAI_CHAT_DEPLOYMENT_NAME='gpt-4o'  # optional")
        return

    try:
        # Run demos
        print("\n[1/4] Knowledge Formulation Principles Demo...")
        await demo_knowledge_formulation_principles()

        print("\n[2/4] Difficulty Levels Demo...")
        await demo_difficulty_levels()

        print("\n[3/4] Reverse Cards Demo...")
        await demo_reverse_cards()

        print("\n[4/4] Wikipedia Integration Demo...")
        await demo_wikipedia_integration()

        print("\n" + "=" * 70)
        print(" ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - demo_kubernetes_basic.csv / demo_kubernetes_cloze.csv")
        print("  - demo_python_wiki_basic.csv / demo_python_wiki_cloze.csv")
        print("\nImport into Anki:")
        print("  1. File → Import → Select CSV")
        print("  2. Use 'Basic' note type for *_basic.csv")
        print("  3. Use 'Cloze' note type for *_cloze.csv")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_all_demos())
