from abc import ABC

from langchain_core.language_models import BaseLanguageModel


class LLMBaseModel(BaseLanguageModel, ABC):
    """Base class for all LLM implementations."""
    pass
