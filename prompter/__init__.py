"""
Module that abstracts submitting the prompt to the LLM half of Bouzyges
"""

from .base import Prompter

from .openai import OpenAIPrompter, OpenAIAzurePrompter
from .human import HumanPrompter


__all__ = [
    "Prompter",
    "OpenAIPrompter",
    "OpenAIAzurePrompter",
    "HumanPrompter",
]
