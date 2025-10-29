"""
Helper module to structure your prepared interview answers

Two formats supported:
1. Q&A pairs - When you have specific questions prepared
2. Story format - When you have experiences to draw from
"""

from typing import List, Dict
from langchain.schema import Document

class AnswerFormatter:
    """
    Helps you structure your prepared answers for optimal retrieval
    """
    
    @staticmethod
    def create_qa_pair(
        question: str,
        answer: str,
        tags: List[str] = None,
        metadata: Dict = None
    ) -> Document:
        """
        Create a Q&A pair document
        
        Example:
        ```
        qa = create_qa_pair(
            question="Tell me about a time you optimized performance",
            answer="At TechCorp, I optimized a database query from 45s to 1.8s...",
            tags=["database", "optimization", "performance"],
            metadata={"project": "Dashboard", "year": 2023}
        )
        ```
        """
        # Format as structured Q&A
        content = f"""QUESTION: {question}

ANSWER: {answer}"""
        
        meta = metadata or {}
        meta['type'] = 'qa_pair'
        meta['question'] = question
        if tags:
            meta['tags'] = tags
        
        return Document(page_content=content, metadata=meta)
    
    @staticmethod
    def create_story(
        title: str,
        situation: str,
        task: str,
        action: str,
        result: str,
        learning: str = None,
        measurement: str = None,
        tags: List[str] = None
    ) -> Document:
        """
        Create a STAR-formatted story
        
        Example:
        ```
        story = create_story(
            title="Database Optimization Project",
            situation="Dashboard was timing out, queries taking 45+ seconds",
            task="Optimize query performance without changing functionality",
            action="Added composite indexes, refactored with CTEs, implemented caching",
            result="Query time reduced from 45s to 1.8s (96% improvement)",
            measurement="Used SQL EXPLAIN ANALYZE, monitored for 1 month",
            learning="Always profile before optimizing",
            tags=["database", "performance", "sql"]
        )
        ```
        """
        content = f"""STORY: {title}

SITUATION: {situation}

TASK: {task}

ACTION: {action}

RESULT: {result}"""
        
        if measurement:
            content += f"\n\nMEASUREMENT: {measurement}"
        
        if learning:
            content += f"\n\nLEARNING: {learning}"
        
        meta = {
            'type': 'story',
            'title': title
        }
        if tags:
            meta['tags'] = tags
        
        return Document(page_content=content, metadata=meta)
    
    @staticmethod
    def create_from_text_file(filepath: str) -> List[Document]:
        """
        Parse a text file with multiple prepared answers
        
        Format:
        ```
        === Q&A ===
        Q: Question here?
        A: Answer here...
        
        === Q&A ===
        Q: Another question?
        A: Another answer...
        
        === STORY ===
        Title: Story title
        Situation: ...
        Task: ...
        Action: ...
        Result: ...
        ```
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        documents = []
        sections = content.split('===')
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            if section.startswith('Q&A'):
                # Parse Q&A format
                lines = section.split('\n', 1)
                if len(lines) < 2:
                    continue
                
                qa_content = lines[1].strip()
                if 'Q:' in qa_content and 'A:' in qa_content:
                    parts = qa_content.split('A:', 1)
                    question = parts[0].replace('Q:', '').strip()
                    answer = parts[1].strip()
                    
                    documents.append(
                        AnswerFormatter.create_qa_pair(question, answer)
                    )
            
            elif section.startswith('STORY'):
                # Parse STORY format
                # Simple parser - you can make this more robust
                doc = Document(
                    page_content=section,
                    metadata={'type': 'story'}
                )
                documents.append(doc)
        
        return documents


# Template for creating your prepared_answers.txt file
PREPARED_ANSWERS_TEMPLATE = """
=== Q&A ===
Q: Tell me about a time you faced a technical challenge
A: During my internship at TechCorp, I was tasked with optimizing a slow database query that was affecting user experience. The query was taking 45 seconds to return results for our main dashboard.

I analyzed the execution plan using EXPLAIN ANALYZE and found that the issue was missing indexes on our 10-million-row user_activity table. I added composite indexes on (user_id, timestamp) and refactored the query to use CTEs instead of nested subqueries.

The result was a 96% improvement - query time went from 45 seconds to 1.8 seconds. I measured this using SQL EXPLAIN ANALYZE and monitored the improvement over one month. User satisfaction scores increased from 3.2 to 4.5 stars.

The key learning was to always profile before optimizing. I spent 2 hours analyzing and only 30 minutes implementing the fix.

=== Q&A ===
Q: How do you measure success in your projects?
A: I believe in defining clear metrics before starting any project. For example, in my database optimization project, I established baseline measurements using SQL EXPLAIN ANALYZE (45s query time) and user satisfaction surveys (3.2/5 stars).

After implementing changes, I monitored these metrics for one month to ensure improvements were sustained. I also tracked secondary metrics like database CPU usage, which decreased by 40%.

I create dashboards to visualize these metrics and share them with stakeholders. This data-driven approach helps demonstrate impact and identify areas for further improvement.

=== STORY ===
Title: API Rate Limiting Challenge
Situation: Our application was hitting rate limits on a third-party weather API during peak hours (we were making 1200 requests/min against a 500 req/min limit)
Task: Reduce API calls while maintaining data freshness
Action: Implemented Redis caching with smart TTL (5 min for current weather, 1 hour for forecasts), added request queuing with exponential backoff, and implemented request coalescing for similar requests
Result: Reduced API calls from 1200/min to 480/min (60% reduction), achieved 60% cache hit rate, error rate dropped from 15% to 0.3%, saved $800/month
Measurement: Built Grafana dashboard tracking API calls/min, cache hit ratio, error rates, and P95/P99 latency. A/B tested caching strategies over 2 weeks.

=== Q&A ===
Q: Describe a time you had to debug a difficult problem
A: I deployed a sentiment analysis model with 92% accuracy in testing, but it only achieved 65% in production. Users were complaining about incorrect predictions.

I collected 1000 production samples and discovered that our training data (formal customer service emails) didn't match production data (casual social media comments with emojis and slang). This was a classic training/production data mismatch.

I implemented quick fixes: text normalization, confidence thresholding (only predict when >80% confident), and added an "uncertain" category. Long-term, I collected 5000 production examples and fine-tuned the model on mixed data.

Results: Production accuracy improved from 65% to 88%, false positive rate dropped from 15% to 3%, and user complaints decreased by 85%.

I learned to always validate on production-like data and that confidence scores are valuable for UX.
"""


def create_sample_answers_file(output_path: str = "./data/prepared_answers.txt"):
    """
    Creates a sample template file for you to fill in
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(PREPARED_ANSWERS_TEMPLATE.strip())
    
    print(f"Created template file: {output_path}")
    print("Edit this file with your own prepared answers!")


if __name__ == "__main__":
    # Create sample file
    create_sample_answers_file()
    
    # Example: Create structured documents programmatically
    formatter = AnswerFormatter()
    
    # Example 1: Q&A pair
    qa_doc = formatter.create_qa_pair(
        question="Tell me about a database optimization challenge",
        answer="I optimized a query from 45s to 1.8s by adding indexes...",
        tags=["database", "performance", "sql"]
    )
    
    # Example 2: STAR story
    story_doc = formatter.create_story(
        title="Database Performance Crisis",
        situation="Dashboard timing out, queries taking 45+ seconds",
        task="Optimize without changing functionality",
        action="Added composite indexes, refactored queries, implemented caching",
        result="Query time: 45s → 1.8s (96% improvement)",
        measurement="SQL EXPLAIN ANALYZE, user satisfaction surveys",
        learning="Always profile before optimizing",
        tags=["database", "optimization"]
    )
    
    print("\n✓ Created sample documents")
    print(f"Q&A: {len(qa_doc.page_content)} characters")
    print(f"Story: {len(story_doc.page_content)} characters")