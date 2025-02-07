from llama_index.core.llms.function_calling import FunctionCallingLLM

from rag.llms.llama_index_llm.claude import ClaudeLLM
from rag.llms.llama_index_llm.llama import LlamaLLM


class LLMFactory:
    @staticmethod
    def get_llm(model_type: str, api_key=None) -> FunctionCallingLLM:
        """Factory method to return the appropriate LLM instance."""
        if model_type == "claude":
            return ClaudeLLM(api_key)
        elif model_type == "llama":
            return LlamaLLM()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
