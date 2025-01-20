from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseLanguageModel


class ClaudeLLM(ChatAnthropic, BaseLanguageModel):
    """Claude LLM implementation that extends LangChain's BaseLanguageModel."""

    def __init__(self, api_key: str):
        super().__init__(
            temperature=0,
            anthropic_api_key=api_key,
            model="claude-3-5-sonnet-20240620"
        )
