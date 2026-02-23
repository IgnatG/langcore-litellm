"""LangCore provider plugin for LiteLLM.

Supports native async inference via ``litellm.acompletion`` when
used with LangCore's ``async_extract`` / ``async_infer`` API.
"""

from langcore_litellm.provider import LiteLLMLanguageModel, UsageStats

__all__ = ["LiteLLMLanguageModel", "UsageStats"]
__version__ = "0.1.0"
