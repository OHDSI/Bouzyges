import logging  # Does not have to work with global LOGGER
from typing import Callable

from parameters import APIParameters
from prompt import Prompt
from utils.exceptions import PrompterError
from utils.types import EscapeHatch

from .base import Prompter


class HumanPrompter(Prompter):
    """\
A test prompter that interacts with a human to get answers.
"""

    _model_id = "human"
    # Only ask the human once
    min_attempts = 1

    def __init__(
        self,
        api_parameters: APIParameters,
        *args,
        prompt_function: Callable[[str], str],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt_function = prompt_function

    async def _prompt_class_answer(
        self, allow_escape, options, prompt, record_attempt
    ):
        while True:
            brain_answer = self.prompt_function("Answer: ").strip()
            try:
                answer = self.unwrap_class_answer(
                    brain_answer,
                    options,
                    EscapeHatch.WORD if allow_escape else None,
                )
                self.cache_remember(prompt, brain_answer, record_attempt)
                return answer

            except PrompterError as e:
                logging.error("Error: %s", e)

    async def _prompt_bool_answer(
        self, prompt: Prompt, record_attempt: int
    ) -> bool:
        while True:
            brain_answer = self.prompt_function("Answer: ").strip()
            try:
                answer = self.unwrap_bool_answer(brain_answer)
                self.cache_remember(prompt, brain_answer, record_attempt)
                return answer
            except PrompterError as e:
                logging.error("Error: %s", e)

    def ping(self) -> bool:
        self.prompt_function("Press Enter to confirm you are here")
        return True

    def report_usage(self) -> None:
        self.logger.info(
            "No usage to report, as this is a human prompter. Stay "
            "hydrated and have a good day!"
        )
