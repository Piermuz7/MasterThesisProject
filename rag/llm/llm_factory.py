from langchain_core.language_models import BaseLanguageModel
from rag.llm.claude import ClaudeLLM
from rag.llm.llama import LlamaLLM


class LLMFactory:
    @staticmethod
    def get_llm(model_type: str, api_key: str = None) -> BaseLanguageModel:
        """Factory method to return the appropriate LLM instance."""
        if model_type == "claude":
            return ClaudeLLM(api_key)
        elif model_type == "llama":
            return LlamaLLM()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
