"""
RAG-based QA System for Interview Stories
Complete implementation with evaluation

Usage:
1. Add your interview stories to ./data/raw/
2. Choose a provider with ``LLM_PROVIDER`` (`open_source` by default) and set the
   corresponding API token
3. Run: python main.py
"""

import os
from config.config import config
from src.document_loader import DocumentLoader
from src.chunking import ChunkingStrategy
from src.vectorstore import VectorStoreManager
from src.retriever import AdvancedRetriever
from src.qa_chain import QASystem
from src.evaluator import QAEvaluator

def setup_qa_system(data_dir: str = "./data/raw", rebuild: bool = False):
    """
    Main setup function - creates the complete QA system
    
    Args:
        data_dir: Directory containing interview stories
        rebuild: If True, rebuild vector store from scratch
    """
    print("=" * 50)
    print("Setting up RAG QA System")
    print("=" * 50)
    
    # Step 1: Load documents
    print("\n[Step 1] Loading documents...")
    loader = DocumentLoader(data_dir)
    documents = loader.load_all_documents()
    
    if not documents:
        print("No documents found! Add .txt or .pdf files to ./data/raw/")
        # Create sample document for testing
        sample_story = """
        Interview Story: Sarah's Database Challenge
        
        During my internship at TechCorp, I was tasked with optimizing a slow 
        database query that was affecting user experience. The query was taking 
        45 seconds to return results.
        
        Challenge: The table had 10 million rows and multiple joins. I analyzed 
        the execution plan and found missing indexes.
        
        Solution: I added composite indexes on the join columns and refactored 
        the query to use CTEs. This reduced query time to 2 seconds.
        
        Measurement: Used SQL EXPLAIN ANALYZE to measure before/after performance. 
        Also tracked user satisfaction scores which improved from 3.2 to 4.5 stars.
        
        Learning: Always profile before optimizing. The 80/20 rule applies - 
        focus on the biggest bottlenecks first.
        """
        documents = [loader.load_from_string(sample_story, {'source': 'sample'})]
        print("Created sample document for testing")
    
    # Step 2: Chunk documents
    print(f"\n[Step 2] Chunking {len(documents)} documents...")
    chunker = ChunkingStrategy(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    # Try different strategies and compare
    print("Comparing chunking strategies...")
    stats = chunker.compare_strategies(documents)
    for strategy, metrics in stats.items():
        print(f"\n{strategy}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")
    
    # Use recursive character splitting (best for most cases)
    chunks = chunker.recursive_character_split(documents)
    print(f"\nUsing recursive character splitting: {len(chunks)} chunks created")
    
    # Step 3: Create/Load vector store
    print("\n[Step 3] Setting up vector store...")
    vs_manager = VectorStoreManager()
    
    if rebuild or not os.path.exists(config.VECTOR_STORE_PATH):
        print("Creating new vector store...")
        vectorstore = vs_manager.create_vectorstore(chunks)
    else:
        print("Loading existing vector store...")
        try:
            vectorstore = vs_manager.load_vectorstore()
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Creating new vector store...")
            vectorstore = vs_manager.create_vectorstore(chunks)
    
    # Print stats
    stats = vs_manager.get_collection_stats()
    print(f"Vector store stats: {stats}")
    
    # Step 4: Create QA system
    print("\n[Step 4] Creating QA system...")
    qa_system = QASystem(vectorstore)
    
    return qa_system, vs_manager, chunks


def run_example_queries(qa_system: QASystem):
    """Run example queries to test the system"""
    print("\n" + "=" * 50)
    print("Testing QA System")
    print("=" * 50)
    
    example_questions = [
        "What challenges did the engineer face with the database?",
        "How did they measure the improvement?",
        "What was the final performance improvement?",
        "What did they learn from this experience?"
    ]
    
    results = []
    for question in example_questions:
        print(f"\n{'=' * 50}")
        print(f"Q: {question}")
        print(f"{'=' * 50}")
        
        result = qa_system.ask(question)
        print(f"\nA: {result['answer']}")
        print(f"\nSources used: {result['metadata']['num_sources']}")
        
        # Show source previews
        for i, source in enumerate(result['metadata']['sources'], 1):
            print(f"\nSource {i}:")
            print(f"  {source['content_preview']}")
        
        results.append(result)
    
    return results


def run_evaluation(qa_system: QASystem, results: list):
    """Evaluate system performance"""
    print("\n" + "=" * 50)
    print("Evaluating System Performance")
    print("=" * 50)
    
    evaluator = QAEvaluator()
    
    # Create simple evaluation dataset
    eval_dataset = [
        {
            'question': 'What challenges did the engineer face?',
            'ground_truth_answer': 'slow database query taking 45 seconds',
            'relevant_doc_ids': [0]
        },
        {
            'question': 'How did they measure improvement?',
            'ground_truth_answer': 'SQL EXPLAIN ANALYZE and user satisfaction scores',
            'relevant_doc_ids': [0]
        }
    ]
    
    # Run evaluation
    metrics = evaluator.evaluate_system_performance(
        results[:2],  # First 2 results
        eval_dataset
    )
    
    print("\n=== Retrieval Metrics ===")
    if 'avg_retrieval' in metrics:
        for metric, value in metrics['avg_retrieval'].items():
            print(f"{metric}: {value:.3f}")
    
    print("\n=== Answer Quality Metrics ===")
    if metrics['answer_metrics']:
        avg_length = sum(m['answer_length'] for m in metrics['answer_metrics']) / len(metrics['answer_metrics'])
        print(f"Average answer length: {avg_length:.0f} characters")


def run_experiments(qa_system: QASystem, vectorstore):
    """
    Run A/B experiments with different configurations
    This demonstrates how to measure improvements
    """
    print("\n" + "=" * 50)
    print("Running A/B Experiments")
    print("=" * 50)
    
    test_question = "What challenges did the engineer face?"
    
    # Experiment 1: Different retrieval strategies
    print("\nExperiment: Comparing retrieval strategies")
    retriever = AdvancedRetriever(vectorstore)
    
    comparison = retriever.compare_retrieval_methods(test_question, k=4)
    
    for method, results in comparison.items():
        if method != 'overlap':
            print(f"\n{method.upper()}:")
            print(f"  Documents retrieved: {results['num_docs']}")
            print(f"  Total context length: {results['total_length']} chars")
    
    print(f"\nDocument overlap between methods: {comparison['overlap']}")
    
    # Experiment 2: Different prompt templates
    print("\n\nExperiment: A/B testing prompts")
    
    prompt_a = """Context: {context}
Question: {question}
Answer concisely:"""
    
    prompt_b = """You are an expert at analyzing interview experiences.
Context: {context}
Question: {question}
Provide a detailed answer with specific examples:"""
    
    result_a = qa_system.ask_with_custom_prompt(test_question, prompt_a)
    result_b = qa_system.ask_with_custom_prompt(test_question, prompt_b)
    
    print(f"Prompt A answer length: {len(result_a['answer'])} chars")
    print(f"Prompt B answer length: {len(result_b['answer'])} chars")


def main():
    """Main execution flow"""
    
    provider = (config.LLM_PROVIDER or "open_source").lower()

    # Check for required credentials based on provider
    if provider == "openai":
        if not config.OPENAI_API_KEY:
            print("Error: OPENAI_API_KEY not found in environment")
            print("Set LLM_PROVIDER=open_source to use the default Mistral model")
            print("or create a .env file with your OpenAI API key (OPENAI_API_KEY=sk-...)")
            return
    else:
        if not config.HUGGINGFACEHUB_API_TOKEN:
            print("Error: HUGGINGFACEHUB_API_TOKEN not found in environment")
            print("Set LLM_PROVIDER=openai with an OpenAI API key to switch providers")
            print(
                "or create a .env file with your Hugging Face token "
                "(HUGGINGFACEHUB_API_TOKEN=hf_...)"
            )
            return
    
    # Setup system
    qa_system, vs_manager, chunks = setup_qa_system(
        data_dir="./data/raw",
        rebuild=False  # Set to True to rebuild vector store
    )
    
    # Run example queries
    results = run_example_queries(qa_system)
    
    # Run evaluation
    run_evaluation(qa_system, results)
    
    # Run experiments
    run_experiments(qa_system, vs_manager.vectorstore)
    
    print("\n" + "=" * 50)
    print("Setup complete! Now you can:")
    print("1. Add more interview stories to ./data/raw/")
    print("2. Run with rebuild=True to update vector store")
    print("3. Modify prompts in qa_chain.py")
    print("4. Experiment with different chunking strategies")
    print("5. Try different retrieval methods")
    print("=" * 50)


if __name__ == "__main__":
    main()