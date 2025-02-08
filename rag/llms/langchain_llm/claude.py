from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel


class ClaudeLLM(ChatAnthropic, BaseChatModel):
    """Claude LLM implementation that extends LlamaIndex's FunctionCallingLLM."""

    def __init__(self, api_key: str):
        super().__init__(
            temperature=0,
            model_name="claude-3-5-sonnet-20241022",
            max_tokens_to_sample=8192,
            api_key=api_key,
            max_retries=3
        )
