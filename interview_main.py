"""
Interview Assistant - Complete Usage Example

This system:
1. Loads your prepared answers
2. Uses LLM to intelligently judge which answer fits
3. Returns your original answer if good match
4. Generates new answer if no good match

No brittle similarity thresholds!
"""

import os
from config.config import config
from src.document_loader import DocumentLoader
from src.vectorstore import VectorStoreManager
from src.interview_assistant import InterviewAssistant, InterviewPracticeSession
from src.prepare_answers import AnswerFormatter, create_sample_answers_file


def setup_interview_system(answers_dir: str = "./data/prepared_answers"):
    """
    Setup the interview assistant system
    
    Args:
        answers_dir: Directory containing your prepared answers
    """
    print("=" * 60)
    print("Setting up Interview Assistant")
    print("=" * 60)
    
    # Check for sample file
    sample_file = "./data/prepared_answers.txt"
    if not os.path.exists(sample_file):
        print("\nNo prepared answers found. Creating template...")
        create_sample_answers_file(sample_file)
        print(f"\n✓ Created template: {sample_file}")
        print("Please edit this file with your actual interview answers!")
        print("\nFor now, using the sample data for demonstration...\n")
    
    # Step 1: Load your prepared answers
    print("\n[Step 1] Loading your prepared answers...")
    
    # Option A: Load from structured file
    if os.path.exists(sample_file):
        formatter = AnswerFormatter()
        documents = formatter.create_from_text_file(sample_file)
        print(f"Loaded {len(documents)} prepared answers from file")
    
    # Option B: Load from directory of text files
    else:
        loader = DocumentLoader(answers_dir)
        documents = loader.load_all_documents()
        print(f"Loaded {len(documents)} documents from directory")
    
    if not documents:
        print("\n⚠ Warning: No documents found!")
        print("Add your prepared answers to:")
        print(f"  - {sample_file} (structured Q&A format)")
        print(f"  - {answers_dir}/ (individual .txt files)")
        return None, None
    
    # IMPORTANT: Don't chunk! Keep complete answers together
    print("\nNote: Not chunking documents - keeping complete answers intact")
    
    # Step 2: Create vector store (for semantic search)
    print("\n[Step 2] Creating vector store...")
    vs_manager = VectorStoreManager(
        persist_directory="./data/interview_vectorstore",
        collection_name="prepared_answers"
    )
    
    vectorstore_path = "./data/interview_vectorstore"
    if os.path.exists(vectorstore_path):
        print("Loading existing vector store...")
        vectorstore = vs_manager.load_vectorstore()
    else:
        print("Creating new vector store...")
        vectorstore = vs_manager.create_vectorstore(documents)
    
    print(f"Vector store ready: {vs_manager.get_collection_stats()}")
    
    # Step 3: Create interview assistant
    print("\n[Step 3] Creating Interview Assistant...")
    assistant = InterviewAssistant(vectorstore)
    
    print("\n✓ Setup complete!")
    return assistant, documents


def demo_intelligent_matching(assistant: InterviewAssistant):
    """
    Demonstrate how the system handles different types of questions
    """
    print("\n" + "=" * 60)
    print("DEMO: Intelligent Question Matching")
    print("=" * 60)
    
    test_questions = [
        # Question 1: Should find exact match
        {
            'question': "Tell me about a database performance issue you solved",
            'expected': 'PREPARED',
            'note': 'Similar to prepared Q&A - should use original'
        },
        # Question 2: Different phrasing, same intent
        {
            'question': "Have you ever optimized a slow query?",
            'expected': 'PREPARED',
            'note': 'Different wording but same topic'
        },
        # Question 3: Related but not directly covered
        {
            'question': "How do you approach system scalability?",
            'expected': 'GENERATED',
            'note': 'Related to your stories but not directly covered'
        },
        # Question 4: Completely different topic
        {
            'question': "Tell me about your experience with machine learning deployment",
            'expected': 'GENERATED/PREPARED',
            'note': 'Depends on your prepared answers'
        }
    ]
    
    for i, test in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test['note']}")
        print(f"Question: {test['question']}")
        print(f"Expected: {test['expected']}")
        print(f"{'='*60}")
        
        result = assistant.answer_question(
            test['question'],
            top_k=3,
            min_relevance_score=7  # Adjust this based on your preference
        )
        
        print(f"\n>>> ANSWER ({result['source']}) <<<")
        print(f"Confidence: {result['confidence']}/10")
        print(f"\n{result['answer'][:300]}...")
        print(f"\n{assistant.explain_decision(test['question'], result)}")
        
        input("\nPress Enter to continue to next question...")


def interactive_practice_mode(assistant: InterviewAssistant):
    """
    Interactive mode - ask questions and get answers
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE PRACTICE MODE")
    print("=" * 60)
    print("\nAsk interview questions and see how the system responds.")
    print("Type 'quit' to exit, 'summary' for session stats\n")
    
    session = InterviewPracticeSession(assistant)
    
    while True:
        question = input("\nInterview Question: ").strip()
        
        if question.lower() == 'quit':
            break
        
        if question.lower() == 'summary':
            summary = session.get_session_summary()
            print("\n" + "="*60)
            print("SESSION SUMMARY")
            print("="*60)
            print(f"Total questions: {summary['total_questions']}")
            print(f"Prepared answers used: {summary['prepared_answers_used']} ({summary['prepared_percentage']:.1f}%)")
            print(f"Generated answers: {summary['generated_answers']}")
            print(f"Average confidence: {summary['avg_confidence']:.1f}/10")
            continue
        
        if not question:
            continue
        
        try:
            session.practice_question(question, show_candidates=False)
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again with a different question.")


def compare_with_threshold_approach(assistant: InterviewAssistant):
    """
    Compare LLM-judge vs similarity threshold approach
    Shows why LLM-judge is better
    """
    print("\n" + "=" * 60)
    print("COMPARISON: LLM-Judge vs Similarity Threshold")
    print("=" * 60)
    
    test_question = "Have you worked on database optimization?"
    
    print(f"\nTest Question: {test_question}\n")
    
    # LLM-judge approach
    print(">>> LLM-Judge Approach:")
    result_llm = assistant.answer_question(test_question)
    print(f"Decision: {result_llm['source']}")
    print(f"Confidence: {result_llm['confidence']}/10")
    print(f"Reasoning: {result_llm['reasoning']}\n")
    
    # Similarity threshold simulation
    print(">>> Similarity Threshold Approach:")
    docs_with_scores = assistant.vectorstore.similarity_search_with_relevance_scores(
        test_question, k=1
    )
    if docs_with_scores:
        doc, score = docs_with_scores[0]
        print(f"Best similarity score: {score:.3f}")
        print("Problem: Score varies wildly based on wording!")
        print("- 'database optimization' might score 0.85")
        print("- 'optimized slow query' might score 0.65")
        print("- 'improved db performance' might score 0.70")
        print("Same topic, very different scores → Hard to set threshold!")
    
    print("\n>>> Winner: LLM-Judge")
    print("Understands semantic intent, not just keyword matching")


def main():
    """
    Main execution with different modes
    """
    
    # Check API key
    if not config.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found")
        print("Create a .env file with: OPENAI_API_KEY=sk-...")
        return
    
    # Setup system
    assistant, documents = setup_interview_system()
    
    if not assistant:
        print("\n⚠ Setup failed. Please add your prepared answers first.")
        return
    
    # Choose mode
    print("\n" + "=" * 60)
    print("Choose Mode:")
    print("=" * 60)
    print("1. Demo - See how intelligent matching works")
    print("2. Interactive Practice - Ask questions freely")
    print("3. Comparison - LLM-judge vs threshold approach")
    print("4. Quick Test - Test a single question")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        demo_intelligent_matching(assistant)
    elif choice == '2':
        interactive_practice_mode(assistant)
    elif choice == '3':
        compare_with_threshold_approach(assistant)
    elif choice == '4':
        question = input("\nEnter your question: ").strip()
        result = assistant.answer_question(question)
        print(f"\n{'='*60}")
        print(f"ANSWER ({result['source']})")
        print(f"{'='*60}\n")
        print(result['answer'])
        print(f"\n{assistant.explain_decision(question, result)}")
    else:
        print("Invalid choice")
    
    print("\n" + "=" * 60)
    print("Tips for best results:")
    print("=" * 60)
    print("1. Keep complete answers together (don't chunk)")
    print("2. Use STAR format for stories")
    print("3. Include specific metrics and numbers")
    print("4. Adjust min_relevance_score if too many/few prepared answers used")
    print("   - Higher (8-9): Only use perfect matches")
    print("   - Lower (6-7): More willing to use prepared answers")
    print("5. More prepared answers = better coverage")


if __name__ == "__main__":
    main()