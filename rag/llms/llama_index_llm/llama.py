from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.llms.ollama import Ollama


class LlamaLLM(Ollama, FunctionCallingLLM):
    """Llama LLM implementation that extends LlamaIndex's FunctionCallingLLM."""

    def __init__(self):
        super().__init__(
            temperature=0,
            model="llama3:8b",
        )
