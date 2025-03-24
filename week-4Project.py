import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# Load environment variables
load_dotenv()

class ContextualChatSystem:
    def __init__(self):
        # Initialize both models
        self.selector_model = ChatGroq(
            temperature=0.3,
            model_name="llama3-8b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        self.generator_model = ChatGroq(
            temperature=0.7,
            model_name="llama3-70b-8192", 
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Define the context selector prompt
        self.selector_prompt = ChatPromptTemplate.from_template(
            """You are a context selection assistant. Your task is to analyze the user's question and select the most relevant context from the provided options that would help answer the question.

            Available contexts:
            {contexts}

            User question: {question}

            Analyze each context and select ONLY THE MOST RELEVANT ONE by returning its ID. If no context is relevant, return "none".

            Your response should be ONLY the context ID or "none"."""
        )
        
        # Define the response generator prompt
        self.generator_prompt = ChatPromptTemplate.from_template(
            """You are an expert assistant that provides accurate, helpful answers based strictly on the provided context. 

            Relevant context:
            {selected_context}

            User question: {question}

            Instructions:
            1. Answer the question using ONLY the provided context
            2. If the context doesn't contain the answer, say "I don't have that information in my knowledge base"
            3. Keep answers concise and accurate
            4. Never make up information

            Answer:"""
        )
        
        # Create the context selection chain
        self.context_selector = (
            {"contexts": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.selector_prompt
            | self.selector_model
            | StrOutputParser()
        )
        
        # Create the response generation chain
        self.response_generator = (
            {"selected_context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.generator_prompt
            | self.generator_model
            | StrOutputParser()
        )
    
    def format_contexts(self, contexts):
        """Format contexts as a string with IDs for selection"""
        return "\n".join(f"ID: {i}\n{ctx}" for i, ctx in enumerate(contexts))
    
    def get_response(self, contexts, question):
        """Process a question through both models"""
        # First select the most relevant context
        formatted_contexts = self.format_contexts(contexts)
        selected_context_id = self.context_selector.invoke({
            "contexts": formatted_contexts,
            "question": question
        })
        
        # Get the selected context text
        if selected_context_id.lower() == "none":
            selected_context = "No relevant context available"
        else:
            try:
                selected_context = contexts[int(selected_context_id)]
            except (ValueError, IndexError):
                selected_context = "No relevant context available"
        
        # Generate response based on selected context
        response = self.response_generator.invoke({
            "selected_context": selected_context,
            "question": question
        })
        
        return response


if __name__ == "__main__":
    
    contexts = [
        "The capital of France is Paris. France is located in Western Europe and is known for its wine and cheese.",
        "The Python programming language was created by Guido van Rossum and first released in 1991. It emphasizes code readability.",
        "The human heart has four chambers: two atria and two ventricles. It pumps blood throughout the body."
    ]
    
    chat_system = ContextualChatSystem()
    
    while True:
        question = input("\nUser question (or 'quit' to exit): ")
        if question.lower() == 'quit' or question.lower() == 'exit':
            break
            
        response = chat_system.get_response(contexts, question)
        print("\nAI:", response)