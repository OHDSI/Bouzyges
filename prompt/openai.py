from __future__ import annotations

import json
from typing import Iterable

from frozendict import frozendict

from prompt.base import Prompt, PromptFormat
from utils.types import (
    BooleanAnswer,
    EscapeHatch,
    OpenAIPromptRole,
    SCTDescription,
)


class OpenAIPromptFormat(PromptFormat):
    """\
Default prompt format for the OpenAI API.

Outputs prompts as JSONs and contains sensible API option defaults.
"""

    default_api_options = frozendict(
        # We want responses to use tokens we provide
        presence_penalty=-0.25,
        # max_tokens=1024,
        timeout=15,
    )
    _UnfinishedPrompt = list[tuple[OpenAIPromptRole, str]]

    def __finalise_prompt(
        self,
        prompt: _UnfinishedPrompt,
        options: frozenset[SCTDescription] | None,
        escape_hatch: SCTDescription | None = None,
    ) -> Prompt:
        history = tuple(
            frozendict({"role": r, "content": t}) for r, t in prompt
        )

        return Prompt(
            history,  # pyright: ignore[reportArgumentType]
            options,
            escape_hatch,
            self.default_api_options,
        )

    def _form_shared_history(self, allow_escape) -> _UnfinishedPrompt:
        prompt = [
            (
                "system",
                "You are " + self.ROLE + ".",
            ),
            (
                "system",
                "Your assignment is " + self.TASK + ".",
            ),
            (
                "system",
                "Please note that " + self.REQUIREMENTS + ".",
            ),
            (
                "system",
                "Your exact instructions are:\n"
                + self.INSTRUCTIONS
                + (allow_escape * self.ESCAPE_INSTRUCTIONS),
            ),
        ]
        return prompt

    def form_supertype(
        self,
        term: str,
        options: Iterable[SCTDescription],
        allow_escape: bool = True,
        term_context: str | None = None,
        options_context: dict[SCTDescription, str] | None = None,
    ) -> Prompt:
        if not options:
            self.logger.error("No options provided for supertype prompt")
            self.logger.debug(
                f"""Prompt context: {term_context}
  - Term: {term}
  - Options: {options}
  - Allowed escape: {allow_escape}
  - Term context: {term_context}
  - Options context: {json.dumps(options_context)}"""
            )
            raise ValueError("No options provided for supertype prompt")

        prompt = self._form_shared_history(allow_escape)
        prompt.append(
            (
                "user",
                (
                    f"Given the term '{term}', what is the closest supertype or "
                    "exact equivalent of it's meaning from the following options?"
                ),
            )
        )
        if term_context:
            prompt.append(
                (
                    "user",
                    (
                        "Following information is provided about the term: "
                        + term_context
                    ),
                )
            )

        options_text = "Options, in no particular order:\n"
        for option in sorted(options):  # Sort for cache consistency
            options_text += f" - {self.wrap_term(option)}"
            if options_context and option in options_context:
                options_text += f": {options_context[option]}"
            options_text += "\n"
        # Remind of the escape hatch, just in case
        if allow_escape:
            options_text += f" - {EscapeHatch.WORD}: None of the above\n"

        prompt.append(("user", options_text))

        return self.__finalise_prompt(
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
        prompt = self._form_shared_history(allow_escape=False)
        prompt.append(
            (
                "user",
                f"Can the attribute '{attribute}' be safely assumed to be "
                f"present in the term '{term}'? The value of the attribute "
                f"itself is not important, and may be vaguely or ambiguously "
                f"defined, the question is about the presence of the attribute "
                f"itself.",
            )
        )
        if term_context:
            prompt.append(
                (
                    "user",
                    (
                        "Following information is provided about the term: "
                        + term_context
                    ),
                )
            )
        if attribute_context:
            prompt.append(
                (
                    "user",
                    "Attribute is defined as follows: " + attribute_context,
                )
            )

        prompt.append(
            (
                "user",
                f"""Options are:
- {BooleanAnswer.YES}: The attribute is guaranteed present.
- {BooleanAnswer.NO}: The attribute is absent or not guaranteed present.""",
            )
        )

        return self.__finalise_prompt(
            prompt,
            frozenset((BooleanAnswer.YES, BooleanAnswer.NO)),
        )

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
        prompt = self._form_shared_history(allow_escape=allow_escape)

        prompt.append(
            (
                "user",
                (
                    f"Choose the value of the attribute '{attribute}' "
                    f"in the term '{term}'. The answer should be the best "
                    f"matching option, either exact, or a likely supertype "
                    f"of an actual value."
                ),
            )
        )

        if term_context:
            prompt.append(
                (
                    "user",
                    (
                        " Following information is provided about the term: "
                        + term_context
                    ),
                )
            )

        if attribute_context:
            prompt.append(
                (
                    "user",
                    (" Attribute is defined as follows: " + attribute_context),
                )
            )

        options_text = "Options, in no particular order:\n"
        for option in sorted(options):  # Sort for cache consistency
            options_text += f" - {self.wrap_term(option)}"
            if options_context and option in options_context:
                options_text += f": {options_context[option]}"
            options_text += "\n"
        # Remind of the escape hatch, just in case
        if allow_escape:
            options_text += f" - {EscapeHatch.WORD}: None of the above\n"

        prompt.append(("user", options_text))

        return self.__finalise_prompt(
            prompt,
            frozenset(options),
            EscapeHatch.WORD if allow_escape else None,
        )

    def form_subsumption(
        self,
        term: str,
        prospective_supertype: SCTDescription,
        term_context: str | None = None,
        supertype_context: str | None = None,
    ) -> Prompt:
        prompt = self._form_shared_history(allow_escape=False)

        prompt.append(
            (
                "user",
                (
                    f"Is the term '{term}' a subtype of the concept "
                    f"'{prospective_supertype}'?"
                ),
            )
        )

        if term_context:
            prompt.append(
                (
                    "user",
                    (
                        " Following information is provided about the term: "
                        + term_context
                    ),
                )
            )

        if supertype_context:
            prompt.append(
                (
                    "user",
                    (
                        " Following information is provided about the concept: "
                        + supertype_context
                    ),
                )
            )

        prompt.append(
            (
                "user",
                f"""Options are:
 - {BooleanAnswer.YES}: The term is a subtype of the concept.
 - {BooleanAnswer.NO}: The term is not a subtype of the concept.""",
            )
        )

        return self.__finalise_prompt(
            prompt,
            frozenset((BooleanAnswer.YES, BooleanAnswer.NO)),
        )
