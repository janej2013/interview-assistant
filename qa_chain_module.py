from typing import Dict, Any, List
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from config.config import config

class QASystem:
    """
    Challenge #5: Prompt engineering and response quality
    Interview answer: "Initial responses were too verbose and sometimes hallucinated.
    I iterated on prompts to:
    - Explicitly require citations from context
    - Instruct model to say 'I don't know' when uncertain
    - Format answers in a structured way
    
    A/B tested different prompts using user feedback scores"
    """
    
    def __init__(self, vectorstore: Chroma, retrieval_strategy: str = "basic"):
        self.vectorstore = vectorstore
        self.retrieval_strategy = retrieval_strategy
        
        self.llm = ChatOpenAI(
            temperature=config.LLM_TEMPERATURE,
            model=config.LLM_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )
        
        # Custom prompt template
        self.prompt_template = self._create_prompt_template()
        self.qa_chain = self._create_qa_chain()
    
    def _create_prompt_template(self) -> PromptTemplate:
        """
        Create a well-engineered prompt template
        Key improvements:
        - Clear instructions
        - Citation requirements
        - Handling uncertainty
        - Structured output
        """
        template = """You are a helpful assistant answering questions about interview experiences and stories.

Use the following pieces of context to answer the question at the end. 

IMPORTANT INSTRUCTIONS:
1. Only use information from the provided context
2. If the answer is not in the context, say "I don't have enough information to answer that question."
3. Cite specific parts of the context in your answer
4. Be concise but complete
5. If multiple relevant stories exist, mention them

Context:
{context}

Question: {question}

Answer: """
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Create the QA chain with retriever"""
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.TOP_K}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Stuff all context into prompt
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )
        
        return qa_chain
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources
        Returns: {
            'question': str,
            'answer': str,
            'source_documents': List[Document],
            'metadata': dict
        }
        """
        result = self.qa_chain.invoke({"query": question})
        
        # Format response with metadata
        response = {
            'question': question,
            'answer': result['result'],
            'source_documents': result['source_documents'],
            'metadata': {
                'num_sources': len(result['source_documents']),
                'sources': [
                    {
                        'content_preview': doc.page_content[:200] + "...",
                        'metadata': doc.metadata
                    }
                    for doc in result['source_documents']
                ]
            }
        }
        
        return response
    
    def batch_ask(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple questions"""
        return [self.ask(q) for q in questions]
    
    def ask_with_custom_prompt(
        self,
        question: str,
        custom_template: str
    ) -> Dict[str, Any]:
        """
        Ask with a custom prompt template for experimentation
        Useful for A/B testing prompts
        """
        custom_prompt = PromptTemplate(
            template=custom_template,
            input_variables=["context", "question"]
        )
        
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.TOP_K}
        )
        
        custom_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": custom_prompt}
        )
        
        result = custom_chain.invoke({"query": question})
        
        return {
            'question': question,
            'answer': result['result'],
            'source_documents': result['source_documents']
        }