import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from frozendict import frozendict

from utils.logger import LOGGER
from utils.types import (
    EscapeHatch,
    JsonDict,
    JsonPrimitive,
    OpenAIMessages,
    SCTDescription,
)


@dataclass(frozen=True, slots=True)
class Prompt:
    """\
Represents a prompt for the LLM agent to answer.

Has option to store API parameters for the answer.
"""

    prompt_message: str | OpenAIMessages
    options: frozenset[SCTDescription] | None = None
    escape_hatch: SCTDescription | None = None
    api_options: frozendict[str, JsonPrimitive] | None = None

    def to_json(self) -> JsonDict:
        """Convert the prompt to a JSON-serializable format for caching."""
        if isinstance(self.prompt_message, str):
            message = self.prompt_message
        else:
            message = [
                {role: text for role, text in message.items()}
                for message in self.prompt_message
            ]

        api_options = dict(self.api_options) if self.api_options else None

        return {
            "prompt_text": json.dumps(message, sort_keys=True),
            "prompt_is_json": not isinstance(self.prompt_message, str),
            "api_options": json.dumps(api_options),
        }  # pyright: ignore[reportReturnType]  # Ruff says it's okay


class PromptFormat(ABC):
    """\
Abstract class for formatting prompts for the LLM agent.
"""

    ROLE = (
        "a domain expert system in clinical terminology who is helping to "
        "build a semantic representation of a concept in a clinical ontology "
        "by providing information about the concept's relationships to other "
        "concepts in the ontology"
    )
    TASK = (
        "to provide information about the given term supertypes, "
        "attributes, attribute values, and other relevant information as "
        "requested, inferring them only from the term meaning and the provided "
        "context"
    )

    REQUIREMENTS = (
        "in addition to providing accurate factually correct information, "
        "it is critically important that you provide answer in a "
        "format that is requested by the system, as answers will "
        "be parsed by a machine. Your answer should ALWAYS end with a line "
        "that says 'The answer is ' and the chosen option. This is the second "
        "time you are being asked the question, as the first time you failed "
        "to adhere to the format. Please make sure to follow the instructions."
    )
    INSTRUCTIONS = (
        "Options that speculate about details not explicitly included in the"
        "term meaning are to be avoided, e.g. term 'operation on abdominal "
        "region' should NOT be assumed to be a laparoscopic operation, as "
        "access method is not specified in the term. It absolutely required to "
        "explain your reasoning when providing answers. The automated system "
        "will look for the last answer surrounded by square brackets, e.g. "
        "[answer], so only one of the options should be selected and returned "
        "in this format. If the question looks like 'What is the topography of "
        "the pulmonary tuberculosis?', and the options are [Lung structure], "
        "[Heart structure], [Kidney structure], the good answer would end with"
        "[Lung structure].' Answers that do not include reasoning are "
        "unacceptable. Incorrect answers will be penalized: if a source term "
        "does contain a specific attribute, you must answer so."
    )

    ESCAPE_INSTRUCTIONS = (
        f" If all provided options are incorrect, or imply extra information "
        f"not present explicitly and unambiguously in the term, you must "
        f"explain why each option is incorrect, and finalize the answer with "
        f"the word {EscapeHatch.WORD}. However, if any of the offered terms "
        f"matches the question, you must select it."
    )

    def __init__(self):
        self.logger = LOGGER.getChild("PromptFormat")

    @staticmethod
    def wrap_term(term: str) -> str:
        """Wrap a term in square brackets."""
        return f"[{term}]"

    @abstractmethod
    def form_supertype(
        self,
        term: str,
        options: Iterable[SCTDescription],
        allow_escape: bool = True,
        term_context: str | None = None,
        options_context: dict[SCTDescription, str] | None = None,
    ) -> Prompt:
        """\
Format a prompt for the LLM agent to choose the best matching proximal ancestor
for a term.
"""

    @abstractmethod
    def form_attr_presence(
        self,
        term: str,
        attribute: SCTDescription,
        term_context: str | None = None,
        attribute_context: str | None = None,
    ) -> Prompt:
        """\
Format a prompt for the LLM agent to decide if an attribute is present in a
term.
"""

    @abstractmethod
    def form_attr_value(
        self,
        term: str,
        attribute: SCTDescription,
        options: Iterable[SCTDescription],
        term_context: str | None = None,
        attribute_context: str | None = None,
        options_context: dict[SCTDescription, str] | None = None,
        allow_escape: bool = True,
    ) -> Prompt:
        """\
Format a prompt for the LLM agent to choose the value of an attribute in a term.
"""

    @abstractmethod
    def form_subsumption(
        self,
        term: str,
        prospective_supertype: SCTDescription,
        term_context: str | None = None,
        supertype_context: str | None = None,
    ) -> Prompt:
        """\
Format a prompt for the LLM agent to decide if a term is a subtype of an
existing (primitive) concept.
"""
