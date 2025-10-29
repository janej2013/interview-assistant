# Quick Start Guide

## Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key
Create `.env` file:
```bash
OPENAI_API_KEY=sk-your-key-here
```

### 3. Add Your Data
Put interview stories in `./data/raw/`:
```bash
mkdir -p data/raw
# Add your .txt or .pdf files here
```

Example story format (save as `story1.txt`):
```
Interview Story: [Title]

Context: [Brief context about the situation]

Challenge: [What problem you faced]

Solution: [How you solved it]

Results: [Measurable outcomes]

Learning: [What you learned]
```

### 4. Run the System
```bash
python main.py
```

---

## What Happens

1. **Loads documents** from `./data/raw/`
2. **Chunks** them into optimal sizes
3. **Creates embeddings** and stores in vector DB
4. **Runs example queries**
5. **Evaluates performance**
6. **Compares different strategies**

First run takes ~2-3 minutes (creating embeddings).
Subsequent runs take ~3 seconds (loads from disk).

---

## Example Interview Stories

### Story 1: Database Optimization
```txt
Interview Story: Database Performance Crisis

During my internship at TechCorp, our main dashboard was timing out. 
Queries were taking 45+ seconds.

Challenge: The user_activity table had 10M rows with complex joins. 
No indexes on foreign keys. Query scanned entire table.

Solution: 
- Analyzed execution plan using EXPLAIN ANALYZE
- Added composite indexes on (user_id, timestamp)
- Refactored query to use CTEs instead of subqueries
- Implemented query result caching (5 min TTL)

Results:
- Query time: 45s → 1.8s (96% improvement)
- Dashboard load time: 60s → 3s
- User satisfaction: 2.8/5 → 4.5/5 stars
- Reduced database CPU by 40%

How I Measured:
- SQL EXPLAIN ANALYZE for execution time
- Application Performance Monitoring (New Relic)
- User feedback surveys
- Database metrics dashboard

Learning: Always profile before optimizing. Spent 2 hours analyzing, 
30 minutes implementing. The 80/20 rule is real.
```

### Story 2: API Rate Limiting
```txt
Interview Story: Scaling a Third-Party Integration

Our app integrated with a weather API but kept hitting rate limits 
during peak hours (500 req/min limit, we were hitting 1200 req/min).

Challenge: 
- Users were getting errors during peak traffic
- No visibility into usage patterns
- Simple caching wasn't enough (weather changes)

Solution:
- Implemented Redis caching with smart TTL (5 min for current, 1hr for forecasts)
- Added request queuing with exponential backoff
- Built usage dashboard to track patterns
- Implemented request coalescing (batch similar requests)

Results:
- Error rate: 15% → 0.3%
- Cache hit rate: 60%
- API calls reduced: 1200/min → 480/min
- Saved $800/month in API costs

How I Measured:
- Built Grafana dashboard tracking:
  * API calls per minute
  * Cache hit/miss ratio
  * Error rates by endpoint
  * P95/P99 latency
- A/B tested caching strategies over 2 weeks
- Monitored for 1 month post-deployment

Learning: Monitor first, optimize second. The dashboard revealed 
that 70% of requests were for the same 5 cities. This insight 
drove the coalescing strategy.
```

### Story 3: Machine Learning Model Debugging
```txt
Interview Story: ML Model Performing Worse in Production

Deployed a sentiment analysis model with 92% accuracy in testing, 
but only 65% in production. Users complained about wrong predictions.

Challenge:
- Training data: customer service emails
- Production data: social media comments (different language style)
- No clear debugging methodology
- Time pressure to fix

Solution:
1. Data Analysis:
   - Collected 1000 production samples
   - Found emoji usage, slang, abbreviations (not in training data)
   - Training data was formal, production was casual

2. Quick Fixes:
   - Added text normalization (expand contractions, handle emojis)
   - Implemented confidence thresholding (only predict when >80% confident)
   - Added "uncertain" category for low-confidence cases

3. Long-term Solution:
   - Collected 5000 production examples
   - Fine-tuned model on mixed dataset
   - Set up continuous evaluation pipeline

Results:
- Production accuracy: 65% → 88% (after fine-tuning)
- Uncertain rate: 12% (acceptable)
- False positive rate: 15% → 3%
- User complaints: -85%

How I Measured:
- Created evaluation dataset from production samples
- Tracked precision/recall by category
- Monitored confidence score distribution
- Weekly manual audit of 100 random predictions
- User feedback loop via thumbs up/down

Learning: 
- Training/production data mismatch is real
- Always validate on production-like data
- Confidence scores are valuable for UX
- Continuous evaluation > one-time testing
```

---

## Using the System

### Ask Questions
```python
from main import setup_qa_system

qa_system, _, _ = setup_qa_system()

result = qa_system.ask("What database challenges did the engineer face?")
print(result['answer'])
print(result['metadata'])  # Shows sources
```

### Try Different Retrieval Strategies
```python
from src.retriever import AdvancedRetriever

retriever = AdvancedRetriever(vectorstore)

# Basic retrieval
docs = retriever.basic_retrieval("database optimization", k=4)

# MMR (reduces redundancy)
docs = retriever.mmr_retrieval("database optimization", k=4)

# Compressed (extracts only relevant parts)
docs = retriever.compressed_retrieval("database optimization", k=4)
```

### Compare Chunking Strategies
```python
from src.chunking import ChunkingStrategy

chunker = ChunkingStrategy(chunk_size=1000, chunk_overlap=200)
stats = chunker.compare_strategies(documents)
print(stats)
```

### Run Evaluation
```python
from src.evaluator import QAEvaluator

evaluator = QAEvaluator()
metrics = evaluator.evaluate_system_performance(results, eval_dataset)
print(metrics['avg_retrieval'])
```

---

## Experimentation Workflow

### 1. Baseline
```bash
python main.py  # Note the metrics
```

### 2. Experiment with Chunk Size
Edit `config/config.py`:
```python
CHUNK_SIZE = 1500  # Try different values
CHUNK_OVERLAP = 300
```

Run again and compare metrics.

### 3. Try Different Retrieval
Edit `main.py`, change retrieval strategy:
```python
retriever = AdvancedRetriever(vectorstore)
docs = retriever.mmr_retrieval(query, k=4)  # Instead of basic
```

### 4. A/B Test Prompts
Edit prompts in `src/qa_chain.py` and compare:
- Answer quality
- Length
- Citation behavior
- Hallucination rate

---

## Directory Structure After Running

```
qa-system/
├── data/
│   ├── raw/
│   │   ├── story1.txt
│   │   └── story2.txt
│   ├── vectorstore/          # Created by ChromaDB
│   │   └── chroma.sqlite3
│   └── evaluations/          # Evaluation results
│       └── eval_20240115_143022.json
├── config/
│   └── config.py
├── src/
│   ├── document_loader.py
│   ├── chunking.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── qa_chain.py
│   └── evaluator.py
├── .env                      # Your API keys
├── requirements.txt
└── main.py
```

---

## Common Issues

### "No documents found"
- Add .txt or .pdf files to `./data/raw/`
- Check file encoding (should be UTF-8)

### "OpenAI API error"
- Check your API key in `.env`
- Ensure you have credits in your OpenAI account

### "ChromaDB error"
- Delete `./data/vectorstore/` and rebuild
- Run with `rebuild=True` in main.py

### Slow performance
- First run creates embeddings (takes time)
- Subsequent runs use cached embeddings (fast)
- Reduce chunk_size or number of documents for testing

---

## Next Steps

1. **Add Real Data:** Replace sample stories with your actual interview experiences
2. **Create Eval Dataset:** Build 20-50 Q&A pairs for testing
3. **Run Experiments:** Try different configurations, measure improvements
4. **Practice Explaining:** Use INTERVIEW_GUIDE.md to prepare answers

---

## Pro Tips

1. **Start Small:** Test with 3-5 stories first
2. **Iterate:** Change one thing at a time, measure impact
3. **Document:** Keep notes on what worked and what didn't
4. **Real Metrics:** Create a real evaluation dataset, don't just eyeball
5. **Be Specific:** In interviews, use exact numbers (45s → 1.8s, not "made it faster")

---

## Questions?

Common things to try:
- Chunk size: 500, 800, 1000, 1500 tokens
- Top-K: 2, 4, 6, 8 documents
- Retrieval: basic vs MMR vs compressed
- Temperature: 0.0 (deterministic) vs 0.3 (creative)
- Models: gpt-3.5-turbo vs gpt-4

Each change will affect:
- Answer quality
- Response time
- Cost per query
- Accuracy metrics

Measure everything!