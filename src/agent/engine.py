import os
import ollama
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMEngine:
    def __init__(self):
        # Configuration: Load from .env or default to local values
        self.model = os.getenv("OLLAMA_MODEL")
        self.host = os.getenv("OLLAMA_HOST")
        
        # Connection to Ollama
        self.client = ollama.Client(host=self.host)
        
        try:
            # Quick connectivity check
            self.client.list()
            print(f"   > [LLM] Connected to Ollama at {self.host}")
        except Exception as e:
            print(f"   > [LLM] CRITICAL ERROR: Could not connect to Ollama. Is 'ollama serve' running? {e}")
            # We don't raise here to allow the app to start, but it will fail on generation

    def chat(self, user_query: str, context: List[Dict], chat_history: List[Dict], system_override: str = None) -> str:
        """
        Generates a response from the LLM.
        
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
                # Handle different potential key names from Vector vs Graph
                text_content = item.get('text', item.get('text_content', ''))
                source_id = item.get('chunk_id', 'Unknown')
                context_str += f"- [{source_id}]: {text_content}\n"

            system_prompt = f"""
            You are a virtual clone of Joe Rogan. 
            
            CONTEXT FROM DATABASE:
            {context_str}
            
            INSTRUCTIONS:
            Answer the user's question using ONLY the context provided above.
            Speak exactly like Joe Rogan (curious, open-minded, uses words like 'wild', 'entirely possible', 'Jamie', '100 percent').
            If the context doesn't contain the answer, just say "I honestly don't know, man. I haven't seen that clip."
            """

        # 2. Build Messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Only include chat history for the Persona mode to keep technical tasks clean
        if not system_override and chat_history:
            # Limit history to last 4 turns to preserve context window
            for msg in chat_history[-4:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": user_query})

        # 3. Generate
        try:
            response = self.client.chat(model=self.model, messages=messages)
            return response['message']['content']
        except Exception as e:
            return f"[LLM Error: {str(e)}]"