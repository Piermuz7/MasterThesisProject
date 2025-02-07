from abc import ABC

from llama_index.core.llms.function_calling import FunctionCallingLLM


class LLMBaseModel(FunctionCallingLLM, ABC):
    """Base class for all LLM implementations."""
    pass
