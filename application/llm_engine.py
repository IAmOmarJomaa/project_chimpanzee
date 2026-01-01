import ollama
from typing import List, Dict, Optional

class LLMEngine:
    def __init__(self):
        self.model = "llama3.2" 
        # Simple connection to local WSL
        self.client = ollama.Client(host="http://127.0.0.1:11434")
        
        try:
            self.client.list()
            print("   > [LLM] Connected to Ollama (Local)")
        except Exception as e:
            print(f"   > [LLM] Error: Could not connect to Ollama. Is 'ollama serve' running? {e}")
            raise e

    def chat(self, user_query: str, context: List[Dict], chat_history: List[Dict], system_override: str = None):
        """
        Generates a response from Llama 3.2.
        
        Args:
            user_query: The user's question.
            context: List of retrieved documents (used for RAG).
            chat_history: Previous conversation turns.
            system_override: (Optional) If set, replaces the 'Joe Rogan' persona. 
                             Used for tool tasks like grading or writing Cypher queries.
        """
        
        # 1. Determine System Prompt
        if system_override:
            # Task-specific mode (e.g., "Write Cypher", "Grade this")
            system_prompt = system_override
        else:
            # Default Persona: Joe Rogan
            context_str = ""
            for item in context:
                context_str += f"- [{item.get('chunk_id', 'Unknown')}]: {item.get('text', '')}\n"

            system_prompt = f"""
            You are a virtual clone of Joe Rogan. 
            CONTEXT:
            {context_str}
            
            INSTRUCTIONS:
            Answer the user's question based on the context. Speak like Joe Rogan.
            """

        # 2. Build Messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # We generally only include chat history for the Persona mode, not for technical tasks
        if not system_override:
            for msg in chat_history[-2:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": user_query})

        # 3. Generate
        try:
            response = self.client.chat(model=self.model, messages=messages)
            return response['message']['content']
        except Exception as e:
            return f"[Error: {e}]"