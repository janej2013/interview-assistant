# Complete Interview Assistant Setup Guide

## 📁 Project Structure

Create this folder structure:

```
interview-assistant/
├── .env
├── requirements.txt
├── config/
│   └── config.py
├── src/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── chunking.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── qa_chain.py
│   ├── evaluator.py
│   ├── interview_assistant.py    ← NEW
│   └── prepare_answers.py        ← NEW
├── data/
│   ├── raw/
│   ├── prepared_answers.txt      ← NEW
│   └── prepared_answers/
├── interview_main.py              ← NEW
├── main.py
└── APPROACHES.md                  ← NEW
```

## 🚀 Quick Start (5 Steps)

### Step 1: Create Project Folder
```bash
mkdir interview-assistant
cd interview-assistant
```

### Step 2: Create All Folders
```bash
mkdir -p config src data/raw data/prepared_answers
```

### Step 3: Get the Files

#### Method A: Copy from Chat Above ⭐ EASIEST

1. Scroll up in this chat
2. Find each artifact (they have titles like "interview_assistant.py")
3. Click the copy button in the artifact
4. Paste into a file with the same name

**Files to copy from artifacts above:**
- `config.py` → save to `config/config.py`
- `document_loader.py` → save to `src/document_loader.py`
- `chunking.py` → save to `src/chunking.py`
- `vectorstore.py` → save to `src/vectorstore.py`
- `retriever.py` → save to `src/retriever.py`
- `qa_chain.py` → save to `src/qa_chain.py`
- `evaluator.py` → save to `src/evaluator.py`
- **`interview_assistant.py`** → save to `src/interview_assistant.py`
- **`prepare_answers.py`** → save to `src/prepare_answers.py`
- **`interview_main.py`** → save to root folder
- `requirements.txt` → save to root folder
- **`APPROACHES.md`** → save to root folder

#### Method B: Copy Code Sections Below

I'll provide the key new files below for easy copying.

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Set API Key
Create `.env` file:
```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### Step 6: Run!
```bash
python interview_main.py
```

---

## 📄 File Contents (Copy These)

### Create `src/__init__.py` (empty file)
```bash
touch src/__init__.py
```

### The 4 NEW Files (Full Code Below)

---

## 📝 Quick Copy: interview_assistant.py

**File:** `src/interview_assistant.py`

```python
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from config.config import config
import json

class InterviewAssistant:
    """
    Smart interview assistant with LLM-as-judge
    No brittle similarity thresholds!
    """
    
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(
            temperature=0,
            model=config.LLM_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )
        
        self.creative_llm = ChatOpenAI(
            temperature=0.3,
            model=config.LLM_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )
    
    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        min_relevance_score: int = 7
    ) -> Dict[str, Any]:
        """Main method: Answer interview question intelligently"""
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}\n")
        
        # Step 1: Retrieve candidates
        candidates = self._retrieve_candidates(question, top_k)
        print(f"Retrieved {len(candidates)} candidate answers")
        
        # Step 2: LLM evaluates each
        evaluations = self._evaluate_candidates(question, candidates)
        
        # Step 3: Find best match
        best_match = max(evaluations, key=lambda x: x['relevance_score'])
        
        print(f"\nBest match score: {best_match['relevance_score']}/10")
        print(f"Reasoning: {best_match['reasoning']}\n")
        
        # Step 4: Decide
        if best_match['relevance_score'] >= min_relevance_score:
            return {
                'answer': best_match['answer'],
                'source': 'PREPARED',
                'confidence': best_match['relevance_score'],
                'reasoning': best_match['reasoning'],
                'candidates_evaluated': len(candidates),
                'metadata': best_match.get('metadata', {})
            }
        else:
            generated = self._generate_answer(question, candidates)
            return {
                'answer': generated['answer'],
                'source': 'GENERATED',
                'confidence': best_match['relevance_score'],
                'reasoning': f"No perfect match (best: {best_match['relevance_score']}/10). Generated from context.",
                'candidates_evaluated': len(candidates),
                'used_contexts': generated['used_contexts']
            }
    
    def _retrieve_candidates(self, question: str, k: int) -> List[Dict]:
        docs = self.vectorstore.similarity_search(question, k=k)
        return [{'answer': doc.page_content, 'metadata': doc.metadata} for doc in docs]
    
    def _evaluate_candidates(self, question: str, candidates: List[Dict]) -> List[Dict]:
        evaluation_prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template="""You are evaluating if a prepared answer suits an interview question.

Interview Question: {question}

Prepared Answer: {answer}

Rate 1-10:
- 9-10: Perfect match
- 7-8: Good match, relevant
- 5-6: Partially relevant
- 3-4: Tangentially related
- 1-2: Not relevant

JSON format:
{{
    "relevance_score": <1-10>,
    "reasoning": "<brief explanation>",
    "should_use": <true/false>
}}"""
        )
        
        evaluations = []
        for i, candidate in enumerate(candidates):
            print(f"Evaluating candidate {i+1}...")
            prompt = evaluation_prompt.format(question=question, answer=candidate['answer'])
            response = self.llm.invoke(prompt)
            
            try:
                eval_result = json.loads(response.content)
                evaluations.append({
                    'answer': candidate['answer'],
                    'metadata': candidate.get('metadata', {}),
                    'relevance_score': eval_result['relevance_score'],
                    'reasoning': eval_result['reasoning'],
                    'should_use': eval_result['should_use']
                })
                print(f"  Score: {eval_result['relevance_score']}/10")
                print(f"  {eval_result['reasoning']}\n")
            except:
                evaluations.append({
                    'answer': candidate['answer'],
                    'metadata': candidate.get('metadata', {}),
                    'relevance_score': 0,
                    'reasoning': 'Evaluation failed',
                    'should_use': False
                })
        
        return evaluations
    
    def _generate_answer(self, question: str, candidates: List[Dict]) -> Dict[str, Any]:
        generation_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""Using prepared materials as context, generate a strong STAR-format interview answer.

Question: {question}

Context: {context}

Generate answer:"""
        )
        
        context = "\n\n---\n\n".join([f"Experience {i+1}:\n{c['answer']}" for i, c in enumerate(candidates[:3])])
        prompt = generation_prompt.format(question=question, context=context)
        response = self.creative_llm.invoke(prompt)
        
        return {'answer': response.content, 'used_contexts': len(candidates[:3])}
    
    def explain_decision(self, question: str, result: Dict[str, Any]) -> str:
        if result['source'] == 'PREPARED':
            return f"""Decision: Using Your Prepared Answer
Confidence: {result['confidence']}/10
Reasoning: {result['reasoning']}"""
        else:
            return f"""Decision: Generated New Answer
Best Match: {result['confidence']}/10
Reasoning: {result['reasoning']}"""

class InterviewPracticeSession:
    def __init__(self, assistant: InterviewAssistant):
        self.assistant = assistant
        self.session_history = []
    
    def practice_question(self, question: str, show_candidates: bool = True):
        result = self.assistant.answer_question(question)
        self.session_history.append({'question': question, 'result': result})
        
        print(f"\n{'='*60}")
        print(f"ANSWER ({result['source']})")
        print(f"{'='*60}\n")
        print(result['answer'])
        print(f"\n{'-'*60}")
        print(self.assistant.explain_decision(question, result))
        
        return result
    
    def get_session_summary(self) -> Dict[str, Any]:
        total = len(self.session_history)
        prepared = sum(1 for q in self.session_history if q['result']['source'] == 'PREPARED')
        avg_conf = sum(q['result']['confidence'] for q in self.session_history) / total if total > 0 else 0
        
        return {
            'total_questions': total,
            'prepared_answers_used': prepared,
            'generated_answers': total - prepared,
            'avg_confidence': avg_conf,
            'prepared_percentage': (prepared / total * 100) if total > 0 else 0
        }
```

**Save as:** `src/interview_assistant.py`

---

## 📝 Quick Copy: prepare_answers.py

**File:** `src/prepare_answers.py`

```python
from typing import List, Dict
from langchain.schema import Document

class AnswerFormatter:
    @staticmethod
    def create_qa_pair(question: str, answer: str, tags: List[str] = None, metadata: Dict = None) -> Document:
        content = f"""QUESTION: {question}

ANSWER: {answer}"""
        
        meta = metadata or {}
        meta['type'] = 'qa_pair'
        meta['question'] = question
        if tags:
            meta['tags'] = tags
        
        return Document(page_content=content, metadata=meta)
    
    @staticmethod
    def create_from_text_file(filepath: str) -> List[Document]:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        documents = []
        sections = content.split('===')
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            if section.startswith('Q&A'):
                lines = section.split('\n', 1)
                if len(lines) < 2:
                    continue
                
                qa_content = lines[1].strip()
                if 'Q:' in qa_content and 'A:' in qa_content:
                    parts = qa_content.split('A:', 1)
                    question = parts[0].replace('Q:', '').strip()
                    answer = parts[1].strip()
                    documents.append(AnswerFormatter.create_qa_pair(question, answer))
            
            elif section.startswith('STORY'):
                documents.append(Document(page_content=section, metadata={'type': 'story'}))
        
        return documents

def create_sample_answers_file(output_path: str = "./data/prepared_answers.txt"):
    template = """=== Q&A ===
Q: Tell me about a database optimization challenge
A: At TechCorp, I optimized a query from 45s to 1.8s by adding composite indexes on (user_id, timestamp) and refactoring with CTEs. Measured using SQL EXPLAIN ANALYZE. User satisfaction improved from 3.2 to 4.5 stars.

=== Q&A ===
Q: How do you measure project success?
A: I define clear metrics before starting. For my database project, I tracked query time (45s → 1.8s) and user satisfaction (3.2 → 4.5). Created dashboards to monitor improvements over one month.

=== Q&A ===
Q: Tell me about debugging a difficult problem
A: Deployed ML model with 92% test accuracy but only 65% in production. Found training data mismatch - trained on formal emails, production had casual social media text. Fixed by normalizing text and fine-tuning on mixed dataset. Accuracy improved to 88%."""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template.strip())
    
    print(f"Created template: {output_path}")
```

**Save as:** `src/prepare_answers.py`

---

## 📝 Quick Copy: interview_main.py

**File:** `interview_main.py`

```python
import os
from config.config import config
from src.vectorstore import VectorStoreManager
from src.interview_assistant import InterviewAssistant, InterviewPracticeSession
from src.prepare_answers import AnswerFormatter, create_sample_answers_file

def setup_interview_system():
    print("="*60)
    print("Setting up Interview Assistant")
    print("="*60)
    
    sample_file = "./data/prepared_answers.txt"
    if not os.path.exists(sample_file):
        print("\nCreating template...")
        create_sample_answers_file(sample_file)
        print(f"✓ Created: {sample_file}")
        print("Edit this file with your answers!\n")
    
    print("\n[Step 1] Loading prepared answers...")
    formatter = AnswerFormatter()
    documents = formatter.create_from_text_file(sample_file)
    print(f"Loaded {len(documents)} prepared answers")
    
    if not documents:
        print("⚠ No documents found!")
        return None, None
    
    print("\n[Step 2] Creating vector store...")
    vs_manager = VectorStoreManager(
        persist_directory="./data/interview_vectorstore",
        collection_name="prepared_answers"
    )
    
    if os.path.exists("./data/interview_vectorstore"):
        vectorstore = vs_manager.load_vectorstore()
    else:
        vectorstore = vs_manager.create_vectorstore(documents)
    
    print("\n[Step 3] Creating assistant...")
    assistant = InterviewAssistant(vectorstore)
    
    print("\n✓ Setup complete!")
    return assistant, documents

def main():
    if not config.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found")
        print("Create .env file with: OPENAI_API_KEY=sk-...")
        return
    
    assistant, _ = setup_interview_system()
    if not assistant:
        return
    
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Ask interview questions. Type 'quit' to exit\n")
    
    session = InterviewPracticeSession(assistant)
    
    while True:
        question = input("\nQuestion: ").strip()
        
        if question.lower() == 'quit':
            break
        
        if not question:
            continue
        
        try:
            session.practice_question(question, show_candidates=False)
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
```

**Save as:** `interview_main.py` (root folder)

---

## ✅ Verification Checklist

After copying all files:

```bash
# Check structure
ls -R

# Should see:
# config/config.py
# src/interview_assistant.py
# src/prepare_answers.py
# interview_main.py
# requirements.txt
# .env

# Test import
python -c "from src.interview_assistant import InterviewAssistant; print('✓ Import works')"

# Run
python interview_main.py
```

---

## 🎯 Next Steps

1. **Edit** `data/prepared_answers.txt` with YOUR stories
2. **Run** `python interview_main.py`
3. **Practice** asking questions
4. **Adjust** `min_relevance_score` in interview_assistant.py if needed

---

## 📞 Need Help?

**Common Issues:**

1. **Import errors**: Make sure `src/__init__.py` exists (can be empty)
2. **API errors**: Check `.env` has correct OpenAI key
3. **No answers found**: Edit `data/prepared_answers.txt`

**File locations:**
- All Python files → see artifacts in chat above
- Just copy each artifact's code to the right file
- Each artifact has a TITLE telling you the filename

---

## 💡 Pro Tip

The **easiest way**: 

1. Find artifact in chat (scroll up)
2. Click copy button
3. Paste into file with same name
4. Done!

All 4 new files are in artifacts above with exact code.