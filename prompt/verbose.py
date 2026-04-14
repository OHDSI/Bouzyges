from __future__ import annotations

from typing import Iterable

from prompt.base import Prompt, PromptFormat
from utils.types import (
    BooleanAnswer,
    EscapeHatch,
    SCTDescription,
)


class VerbosePromptFormat(PromptFormat):
    """\
Default simple verbose prompt format for the LLM agent.

Contains no API options and only string prompts, intended for human prompters.
    """

    def _form_shared_header(self, allow_escape) -> str:
        prompt = ""
        prompt += "You are " + self.ROLE + ".\n\n"
        prompt += "Your assignment is " + self.TASK + ".\n\n"
        prompt += "Please note that " + self.REQUIREMENTS + ".\n\n"
        prompt += "Your exact instructions are:\n\n"
        prompt += self.INSTRUCTIONS + (allow_escape * self.ESCAPE_INSTRUCTIONS)
        prompt += "\n\n"
        return prompt

    def form_supertype(
        self,
        term: str,
        options: Iterable[SCTDescription],
        allow_escape: bool = True,
        term_context: str | None = None,
        options_context: dict[SCTDescription, str] | None = None,
    ) -> Prompt:
        prompt = self._form_shared_header(allow_escape)

        prompt += (
            f"Given the term '{term}', what is the closest supertype or exact "
            "equivalent of it's meaning from the following options?"
        )
        if term_context:
            prompt += " Following information is provided about the term: "
            prompt += term_context

        prompt += "\n\nOptions, in no particular order:\n"
        for option in sorted(options):  # Sort for cache consistency
            prompt += f" - {self.wrap_term(option)}"
            if options_context and option in options_context:
                prompt += f": {options_context[option]}"
            prompt += "\n"
        # Remind of the escape hatch, just in case
        if allow_escape:
            prompt += f" - {EscapeHatch.WORD}: None of the above\n"
        return Prompt(
            prompt,
            frozenset(options),
            EscapeHatch.WORD if allow_escape else None,
        )

    def form_attr_presence(
        self,
        term: str,
        attribute: SCTDescription,
        term_context: str | None = None,
        attribute_context: str | None = None,
    ) -> Prompt:
        prompt = self._form_shared_header(allow_escape=False)
        prompt += (
            f"Is the attribute '{attribute}' present in the term '{term}'?"
        )
        if term_context:
            prompt += " Following information is provided about the term: "
            prompt += term_context
        if attribute_context:
            prompt += " Attribute is defined as follows: "
            prompt += attribute_context

        prompt += "\n\n"
        prompt += f"""Options are:
 - {BooleanAnswer.YES}: The attribute is guaranteed present.
 - {BooleanAnswer.NO}: The attribute is absent or not guaranteed present.
"""
        return Prompt(prompt, frozenset((BooleanAnswer.YES, BooleanAnswer.NO)))

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
        prompt = self._form_shared_header(allow_escape=allow_escape)
        prompt += (
            f"Choose the value of the attribute '{attribute}' in the term "
            f"'{term}'. The answer should be the best matching option, either "
            f"exact, or a likely supertype of an actual value."
        )
        if term_context:
            prompt += " Following information is provided about the term: "
            prompt += term_context
        if attribute_context:
            prompt += " Attribute is defined as follows: "
            prompt += attribute_context

        prompt += "\n\nOptions, in no particular order:\n"
        for option in sorted(options):  # Sort for cache consistency
            prompt += f" - {self.wrap_term(option)}"
            if options_context and option in options_context:
                prompt += f": {options_context[option]}"
            prompt += "\n"
        # Remind of the escape hatch, just in case
        prompt += f" - {EscapeHatch.WORD}: None of the above\n"
        return Prompt(prompt, frozenset(options), EscapeHatch.WORD)

    def form_subsumption(
        self,
        term: str,
        prospective_supertype: SCTDescription,
        term_context: str | None = None,
        supertype_context: str | None = None,
    ) -> Prompt:
        prompt = self._form_shared_header(allow_escape=False)
        prompt += (
            f"Is the term '{term}' a STRICT SUBTYPE or EXACT MATCH of the "
            f"concept '{prospective_supertype}'? If the term has any "
            f"additional meaning not present in the concept, it can not be "
            f"considered a subtype."
        )

        if term_context:
            prompt += " Following information is provided about the term: "
            prompt += term_context

        if supertype_context:
            prompt += " Following information is provided about the concept: "
            prompt += supertype_context

        prompt += "\n\n"
        prompt += f"""Options are:
 - {BooleanAnswer.YES}: The term is a subtype of the concept.
 - {BooleanAnswer.NO}: The term is not a subtype of the concept.
"""
        return Prompt(prompt, frozenset((BooleanAnswer.YES, BooleanAnswer.NO)))
