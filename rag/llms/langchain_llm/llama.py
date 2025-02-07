from langchain_ollama import OllamaLLM
from langchain_core.language_models import BaseLanguageModel


class LlamaLLM(OllamaLLM, BaseLanguageModel):
    """Llama LLM implementation that extends LlamaIndex's FunctionCallingLLM."""

    def __init__(self):
        super().__init__(
            temperature=0,
            model="llama3:8b",
        )
