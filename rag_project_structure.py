# Project Structure
# Create these files in your repo:

"""
qa-system/
├── requirements.txt
├── .env.example
├── config/
│   └── config.py
├── src/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── chunking.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── qa_chain.py
│   └── evaluator.py
├── data/
│   ├── raw/              # Put your interview stories here
│   └── processed/
├── notebooks/
│   └── experiment.ipynb
├── tests/
│   └── test_qa_system.py
└── main.py
"""

# requirements.txt content:
"""
langchain==0.1.0
langchain-community==0.0.10
langchain-openai==0.0.2
chromadb==0.4.22
sentence-transformers==2.2.2
python-dotenv==1.0.0
tiktoken==0.5.2
pandas==2.1.4
numpy==1.26.3
ragas==0.1.0
"""