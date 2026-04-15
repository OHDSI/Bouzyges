from abc import ABC, abstractmethod

from prompt import Prompt, PromptFormat
from utils.logger import LOGGER
from utils.constants import (
    DEFAULT_REPEAT_PROMPTS,
    NULL_ANSWER,
)
from utils.exceptions import PrompterError, PrompterInitError
from utils.types import (
    SCTDescription,
    EscapeHatch,
    BooleanAnswer,
)
from collections.abc import Iterable
from typing import Callable, Coroutine, TypeVar, Any
from parameters import APIParameters
from functools import wraps
from collections import Counter

from .cache import PromptCache

import math

AttemptCount = int
PromptAnswerT = TypeVar("PromptAnswerT", bool, SCTDescription | EscapeHatch)


def ask_many(method: Callable[..., Coroutine[Any, Any, PromptAnswerT]]):
    """\
Decorator for methods that require prompting. Will repeat the prompt until
enough correct answers are received.
"""

    @wraps(method)
    async def wrapper(self, *args, **kwargs) -> PromptAnswerT:
        attempt = 1
        winning_attempts = math.ceil(self.min_attempts / 2)
        options_count: Counter[SCTDescription | bool | EscapeHatch] = Counter()
        while True:
            kwargs["attempt"] = attempt
            answer = await method(self, *args, **kwargs)
            options_count.update([answer])
            if options_count[answer] >= winning_attempts:
                return answer
            attempt += 1

    return wrapper


class Prompter(ABC):
    """\
Interfaces prompts to the LLM agent and parses answers.
"""

    _model_id: str = "UNKNOWN"
    min_attempts: int = DEFAULT_REPEAT_PROMPTS

    def __init__(
        self,
        api_parameters: APIParameters,
        *args,
        prompt_format: PromptFormat,
        **kwargs,
    ):
        _ = args, kwargs
        self.prompt_format = prompt_format
        self.logger = LOGGER.getChild(self.__class__.__name__)
        if api_parameters.cache_db is not None:
            try:
                self.cache = PromptCache.use_db(api_parameters.cache_db)
            except Exception as e:
                self.logger.error(
                    "Could not connect (create) to the cache DB", exc_info=e
                )
                raise PrompterInitError(e)
        else:
            self.cache = None

    @staticmethod
    def unwrap_class_answer(
        answer: str,
        options: Iterable[SCTDescription] = (),
        escape_hatch: SCTDescription | None = EscapeHatch.WORD,
    ) -> SCTDescription | EscapeHatch:
        """\
Check if answer has exactly one valid option.

Assumes that the answer is a valid option if it is wrapped in brackets.
"""
        last_line = answer.strip().splitlines()[-1]
        # Try to parse the last line, then the answer as a whole
        look_at = [last_line, answer]

        if not options:
            for text in look_at:
                # Return the answer in brackets, if there is one
                if text.count("[") == text.count("]") == 1:
                    start = text.index("[") + 1
                    end = text.index("]")
                    return SCTDescription(text[start:end])

            raise PrompterError(
                "Could not find a unique option in the answer:", last_line
            )

        wrapped_options = {
            PromptFormat.wrap_term(option): option for option in options
        }

        if escape_hatch is not None:
            wrapped_options = {
                **wrapped_options,
                EscapeHatch.WORD: escape_hatch,
            }

        for text in look_at:
            counts = {}
            for option in wrapped_options:
                counts[option] = text.count(option)

            # Check if there is exactly one option present
            if sum(map(bool, counts.values())) == 1:
                for option, count in counts.items():
                    if count:
                        return (
                            SCTDescription(option[1:-1])
                            if option != escape_hatch
                            else NULL_ANSWER
                        )

            # Return the last encountered option in brackets
            indices: dict[SCTDescription | EscapeHatch, int] = {
                option: text.rfind(wrapped)
                for wrapped, option in wrapped_options.items()
            }
            if any(index != -1 for index in indices.values()):
                return max(indices, key=lambda k: indices.get(k, -1))

        raise PrompterError(
            "Could not find a unique option in the answer:", last_line
        )

    @staticmethod
    def unwrap_bool_answer(
        answer: str,
        yes: str = BooleanAnswer.YES,
        no: str = BooleanAnswer.NO,
    ) -> bool:
        """\
Check if the answer contains a yes or no option.
"""
        if yes in answer and no not in answer:
            return True
        elif no in answer and yes not in answer:
            return False
        else:
            raise PrompterError(
                "Could not find an unambiguous boolean answer in the response"
            )

    @ask_many
    async def prompt_supertype(
        self,
        term: str,
        options: Iterable[SCTDescription],
        allow_escape: bool = True,
        term_context: str | None = None,
        options_context: dict[SCTDescription, str] | None = None,
        attempt: int = 1,
    ) -> SCTDescription | EscapeHatch:
        """\
Prompt the model to choose the best matching proximal ancestor for a term.
"""
        # Construct the prompt
        prompt: Prompt = self.prompt_format.form_supertype(
            term, options, allow_escape, term_context, options_context
        )
        self.logger.debug(f"Constructed prompt: f{prompt.prompt_message}")
        self.logger.debug(f"Getting answer #{attempt}")

        if cached_answer := self.cache_get(prompt, attempt):
            answer = self.unwrap_class_answer(
                cached_answer,
                options,
                EscapeHatch.WORD if allow_escape else None,
            )
        else:
            # Get the answer
            answer = await self._prompt_class_answer(
                allow_escape, options, prompt, attempt
            )
        self.logger.info(f"Agent answer: {answer} is a supertype of {term}")
        return answer

    @ask_many
    async def prompt_attr_presence(
        self,
        term: str,
        attribute: SCTDescription,
        term_context: str | None = None,
        attribute_context: str | None = None,
        attempt: int = 1,
    ) -> bool:
        prompt: Prompt = self.prompt_format.form_attr_presence(
            term, attribute, term_context, attribute_context
        )
        self.logger.debug(f"Constructed prompt: f{prompt.prompt_message}")
        self.logger.debug(f"Getting answer #{attempt}")

        if cached_answer := self.cache_get(prompt, attempt):
            answer = self.unwrap_bool_answer(cached_answer)
        else:
            answer = await self._prompt_bool_answer(prompt, attempt)
        self.logger.info(
            f"Agent answer: The attribute '{attribute}' is "
            f"{'present' if answer else 'absent'} in '{term}'"
        )
        return answer

    @ask_many
    async def prompt_attr_value(
        self,
        term: str,
        attribute: SCTDescription,
        options: Iterable[SCTDescription],
        term_context: str | None = None,
        attribute_context: str | None = None,
        options_context: dict[SCTDescription, str] | None = None,
        allow_escape: bool = True,
        attempt: int = 1,
    ) -> SCTDescription | EscapeHatch:
        """\
Prompt the model to choose the value of an attribute in a term.
"""
        prompt: Prompt = self.prompt_format.form_attr_value(
            term,
            attribute,
            options,
            term_context,
            attribute_context,
            options_context,
            allow_escape,
        )
        self.logger.debug(f"Constructed prompt: f{prompt.prompt_message}")
        self.logger.debug(f"Getting answer #{attempt}")

        if cached_answer := self.cache_get(prompt, attempt):
            answer = self.unwrap_class_answer(
                cached_answer,
                options,
                EscapeHatch.WORD if allow_escape else None,
            )
        else:
            answer = await self._prompt_class_answer(
                allow_escape, options, prompt, attempt
            )

        self.logger.info(
            f"Agent answer: The value of the attribute '{attribute}' in "
            f"'{term}' is '{answer}'"
        )
        return answer

    @ask_many
    async def prompt_subsumption(
        self,
        term: str,
        prospective_supertype: SCTDescription,
        term_context: str | None = None,
        supertype_context: str | None = None,
        attempt: int = 1,
    ) -> bool:
        """\
Prompt the model to decide if a term is a subtype of a prospective supertype.

Only meant to be used for Primitive concepts: use Bouzyges.check_subsumption for
Fully Defined concepts.
"""
        prompt: Prompt = self.prompt_format.form_subsumption(
            term, prospective_supertype, term_context, supertype_context
        )
        self.logger.debug(f"Constructed prompt: f{prompt.prompt_message}")
        self.logger.debug(f"Getting answer #{attempt}")

        if cached_answer := self.cache_get(prompt, attempt):
            answer = self.unwrap_bool_answer(cached_answer)
        else:
            answer = await self._prompt_bool_answer(prompt, attempt)

        self.logger.info(
            f"From cache: The term '{term}' is "
            f"{'a subtype' if answer else 'not a subtype'} "
            f"of '{prospective_supertype}'"
        )
        return answer

    def cache_remember(self, prompt: Prompt, answer: str, attempt: int) -> None:
        if self.cache:
            self.cache.remember(self._model_id, prompt, answer, attempt)

    def cache_get(self, prompt: Prompt, attempt: int) -> str | None:
        if self.cache:
            return self.cache.get(self._model_id, prompt, attempt)
        return None

    # Following methods are abstract and represent common queries to the model
    @abstractmethod
    async def _prompt_bool_answer(
        self, prompt: Prompt, record_attempt: int
    ) -> bool:
        """\
Send a prompt to the counterpart agent to obtain the answer
"""

    @abstractmethod
    async def _prompt_class_answer(
        self,
        allow_escape: bool,
        options: Iterable[SCTDescription],
        prompt: Prompt,
        record_attempt: int,
    ) -> SCTDescription | EscapeHatch:
        """\
Send a prompt to the counterpart agent to obtain a single choice answer.
"""

    @abstractmethod
    def ping(self) -> bool:
        """\
Check if the API is available.
"""

    @abstractmethod
    def report_usage(self) -> None:
        """\
Report the usage of the API to the provider.

Form is not specified, as it is provider-specific.
"""
