from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.llms.anthropic import Anthropic


class ClaudeLLM(Anthropic, FunctionCallingLLM):
    """Claude LLM implementation that extends LlamaIndex's FunctionCallingLLM."""

    def __init__(self, api_key: str):
        super().__init__(
            temperature=0,
            api_key=api_key,
            model="claude-3-5-sonnet-20241022",
            max_tokens=8192,
        )
