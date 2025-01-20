from langchain_ollama.llms import OllamaLLM
from langchain_core.language_models import BaseLanguageModel


class LlamaLLM(OllamaLLM, BaseLanguageModel):
    """Llama LLM implementation that extends LangChain's BaseLanguageModel."""

    def __init__(self):
        super().__init__(
            temperature=0,
            model="llama3.1",
            base_url="http://localhost:11434"
        )
