from typing import Dict, Any, List
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import json
from src.llm_factory import create_chat_model, extract_response_content

class InterviewAssistant:
    """
    Smart interview assistant that:
    1. Retrieves candidate answers from your prepared materials
    2. Uses LLM to judge if they're good matches
    3. Returns original answer if good match, generates new one if not
    
    No brittle similarity thresholds - the LLM decides intelligently!
    """
    
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.llm = create_chat_model(temperature=0)

        self.creative_llm = create_chat_model(temperature=0.3)
    
    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        min_relevance_score: int = 7
    ) -> Dict[str, Any]:
        """
        Main method: Answer interview question intelligently
        
        Args:
            question: The interviewer's question
            top_k: Number of candidate answers to evaluate
            min_relevance_score: Minimum score (1-10) to use prepared answer
        
        Returns:
            {
                'answer': str,
                'source': 'PREPARED' | 'GENERATED',
                'confidence': int (1-10),
                'reasoning': str,
                'candidates_evaluated': int
            }
        """
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}\n")
        
        # Step 1: Retrieve candidate answers
        candidates = self._retrieve_candidates(question, top_k)
        print(f"Retrieved {len(candidates)} candidate answers")
        
        # Step 2: LLM evaluates each candidate
        evaluations = self._evaluate_candidates(question, candidates)
        
        # Step 3: Find best match
        best_match = max(evaluations, key=lambda x: x['relevance_score'])
        
        print(f"\nBest match score: {best_match['relevance_score']}/10")
        print(f"Reasoning: {best_match['reasoning']}\n")
        
        # Step 4: Decide whether to use prepared answer or generate
        if best_match['relevance_score'] >= min_relevance_score:
            # Use prepared answer
            return {
                'answer': best_match['answer'],
                'source': 'PREPARED',
                'confidence': best_match['relevance_score'],
                'reasoning': best_match['reasoning'],
                'candidates_evaluated': len(candidates),
                'metadata': best_match.get('metadata', {})
            }
        else:
            # Generate new answer using all candidates as context
            generated = self._generate_answer(question, candidates)
            return {
                'answer': generated['answer'],
                'source': 'GENERATED',
                'confidence': best_match['relevance_score'],
                'reasoning': f"No perfect match found (best: {best_match['relevance_score']}/10). Generated answer using your prepared materials as context.",
                'candidates_evaluated': len(candidates),
                'used_contexts': generated['used_contexts']
            }
    
    def _retrieve_candidates(self, question: str, k: int) -> List[Dict]:
        """Retrieve top-k candidate answers"""
        docs = self.vectorstore.similarity_search(question, k=k)
        
        candidates = []
        for doc in docs:
            candidates.append({
                'answer': doc.page_content,
                'metadata': doc.metadata
            })
        
        return candidates
    
    def _evaluate_candidates(
        self,
        question: str,
        candidates: List[Dict]
    ) -> List[Dict]:
        """
        Use LLM to evaluate how well each candidate answers the question
        Returns list of evaluations with scores
        """
        evaluation_prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template="""You are evaluating whether a prepared answer is suitable for an interview question.

Interview Question: {question}

Prepared Answer: {answer}

Evaluate this answer on a scale of 1-10:
- 9-10: Perfect match, directly answers the question
- 7-8: Good match, relevant but might need minor adjustments
- 5-6: Partially relevant, covers some aspects
- 3-4: Tangentially related but not a good answer
- 1-2: Not relevant at all

Consider:
- Does it address the core intent of the question?
- Is it specific enough or too generic?
- Would the interviewer be satisfied with this answer?
- Does it cover the right time period/context?

Respond in JSON format:
{{
    "relevance_score": <1-10>,
    "reasoning": "<brief explanation>",
    "should_use": <true/false>
}}
"""
        )
        
        evaluations = []
        for i, candidate in enumerate(candidates):
            print(f"Evaluating candidate {i+1}...")
            
            # Create prompt
            prompt = evaluation_prompt.format(
                question=question,
                answer=candidate['answer']
            )
            
            # Get LLM evaluation
            response = self.llm.invoke(prompt)
            raw_output = extract_response_content(response)

            try:
                # Parse JSON response
                eval_result = json.loads(raw_output)
                
                evaluations.append({
                    'answer': candidate['answer'],
                    'metadata': candidate.get('metadata', {}),
                    'relevance_score': eval_result['relevance_score'],
                    'reasoning': eval_result['reasoning'],
                    'should_use': eval_result['should_use']
                })
                
                print(f"  Score: {eval_result['relevance_score']}/10")
                print(f"  Reasoning: {eval_result['reasoning']}\n")
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                print(
                    f"  Warning: Could not parse evaluation for candidate {i+1}."
                    f" Raw response: {raw_output!r}"
                )
                evaluations.append({
                    'answer': candidate['answer'],
                    'metadata': candidate.get('metadata', {}),
                    'relevance_score': 0,
                    'reasoning': 'Evaluation failed',
                    'should_use': False
                })
        
        return evaluations
    
    def _generate_answer(
        self,
        question: str,
        candidates: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate new answer using candidates as context
        This is used when no prepared answer is good enough
        """
        generation_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are helping someone prepare for an interview. They have prepared some materials, but none directly answer this specific question.

Using their prepared materials as context, generate a strong interview answer that:
1. Draws from their actual experiences mentioned in the context
2. Follows the STAR format (Situation, Task, Action, Result)
3. Includes specific metrics and details when available
4. Sounds natural and authentic
5. Is concise (2-3 minutes speaking time)

Interview Question: {question}

Prepared Materials (for context):
{context}

Generate a strong interview answer:"""
        )
        
        # Combine all candidate answers as context
        context = "\n\n---\n\n".join([
            f"Experience {i+1}:\n{c['answer']}"
            for i, c in enumerate(candidates[:3])  # Use top 3
        ])
        
        prompt = generation_prompt.format(
            question=question,
            context=context
        )
        
        response = self.creative_llm.invoke(prompt)

        return {
            'answer': extract_response_content(response),
            'used_contexts': len(candidates[:3])
        }
    
    def explain_decision(
        self,
        question: str,
        result: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable explanation of the decision
        Useful for understanding why it chose prepared vs generated
        """
        if result['source'] == 'PREPARED':
            return f"""
Decision: Using Your Prepared Answer
Confidence: {result['confidence']}/10
Reasoning: {result['reasoning']}

The AI found a good match in your prepared materials. This is your original answer with no modifications.
"""
        else:
            return f"""
Decision: Generated New Answer
Best Match Score: {result['confidence']}/10
Reasoning: {result['reasoning']}

The AI didn't find a perfect match, so it generated a new answer by synthesizing information from {result['candidates_evaluated']} of your prepared experiences.
"""


class InterviewPracticeSession:
    """
    Helper class for running practice interview sessions
    Shows side-by-side comparison and builds confidence
    """
    
    def __init__(self, assistant: InterviewAssistant):
        self.assistant = assistant
        self.session_history = []
    
    def practice_question(
        self,
        question: str,
        show_candidates: bool = True
    ):
        """
        Practice a single question and show detailed breakdown
        """
        result = self.assistant.answer_question(question)
        
        self.session_history.append({
            'question': question,
            'result': result
        })
        
        # Display result
        print(f"\n{'='*60}")
        print(f"ANSWER ({result['source']})")
        print(f"{'='*60}\n")
        print(result['answer'])
        print(f"\n{'-'*60}")
        print(self.assistant.explain_decision(question, result))
        
        if show_candidates:
            print(f"\nEvaluated {result['candidates_evaluated']} candidates")
        
        return result
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the practice session"""
        total = len(self.session_history)
        prepared = sum(1 for q in self.session_history if q['result']['source'] == 'PREPARED')
        generated = total - prepared
        
        avg_confidence = sum(
            q['result']['confidence'] for q in self.session_history
        ) / total if total > 0 else 0
        
        return {
            'total_questions': total,
            'prepared_answers_used': prepared,
            'generated_answers': generated,
            'avg_confidence': avg_confidence,
            'prepared_percentage': (prepared / total * 100) if total > 0 else 0
        }