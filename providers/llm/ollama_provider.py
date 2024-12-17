from typing import List, Optional
from langchain.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.chains import ConversationChain
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts.chat import BaseMessagePromptTemplate
from ..base import LLMProvider

class OllamaLLM(LLMProvider):
    def __init__(self, model_name: str = "llama2"):
        # Inizializza il modello chat
        self.chat = ChatOllama(
            model=model_name,
            base_url="http://localhost:11434"
        )

        # Inizializza la memoria
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )

        # Crea il template del prompt
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful voice assistant. Be concise and natural in your responses."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Crea la catena di conversazione
        self.conversation = ConversationChain(
            memory=self.memory,
            prompt=self.prompt,
            llm=self.chat
        )

    def get_response(self, text: str, system_prompt: str) -> str:
        try:
            # Aggiorna il system prompt se necessario
            self.prompt.messages[0] = SystemMessage(content=system_prompt)

            # Ottieni la risposta
            response = self.conversation.predict(input=text)

            if response is None:
                return "I apologize, I couldn't generate a response."

            return response.strip()

        except Exception as e:
            print(f"Error getting LLM response: {e}")
            raise

    def cleanup(self) -> None:
        if self.memory is not None:
            self.memory.clear()
