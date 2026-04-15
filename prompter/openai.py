from typing import Callable
import openai
import asyncio
import json

import httpx
from parameters import APIParameters
from prompt import Prompt
from utils.exceptions import PrompterError, PrompterInitError
from utils.logger import LOGGER
from utils.types import EscapeHatch
from utils.decorators import retry_exponential

from .base import Prompter
import os

## tiktoken
try:
    import tiktoken
except ImportError:
    LOGGER.warning(
        "tiktoken package is not installed. "
        "Will not be able to use track token usage"
    )
    tiktoken = None


class OpenAIPrompter(Prompter):
    """\
A prompter that interfaces with the OpenAI API using.
"""

    def __init__(
        self,
        api_parameters: APIParameters,
        *args,
        http_client: httpx.AsyncClient,
        repeat_prompts: int | None = None,
        model: str,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._model_id = model
        self._init_client(*args, http_client=http_client, **kwargs)

        if repeat_prompts is not None:
            self.min_attempts = repeat_prompts

        self._estimated_token_usage: dict[Prompt, int] = {}
        self._actual_token_usage: dict[Prompt, int] = {}
        if tiktoken is not None:
            try:
                self._estimation_encoding = tiktoken.encoding_for_model(
                    self._model_id
                )
            except KeyError:
                self.logger.warning("Model not found in the tokenizer")
                self._estimation_encoding = None
        else:
            self.logger.warning(
                "Tiktoken not installed, can't track token usage"
            )
            self._estimation_encoding = None

    def _init_client(self, *args, http_client: httpx.AsyncClient, **kwargs):
        self.logger.info("Initializing the OpenAI API client...")
        _ = args, kwargs
        # API Key will be picked up from env variables
        # self._api_key = api_key
        if not os.getenv("OPENAI_API_KEY"):
            raise PrompterInitError(
                f"Can not initialize {self} without OPENAI_API_KEY"
            )

        self._client = openai.AsyncOpenAI(
            http_client=http_client,
            max_retries=1,  # Retry only once, as we have our own retry logic
        )
        self._ping_headers = {}

    async def _ping(self):
        try:
            return await self._client.models.list(
                extra_headers=self._ping_headers, timeout=5
            )
        except Exception as e:
            self.logger.warning(f"Connection timed out: {e}", exc_info=e)
            raise

    def ping(self) -> bool:
        self.logger.info("Pinging the OpenAI API...")
        # Ping by retrieving the list of models
        try:
            models = asyncio.run(self._ping())
        except Exception as e:
            self.logger.warning(f"API is not available: {e}")
            return False

        response: dict = models.model_dump()
        success = response.get("data", []) != []
        if success:
            self.logger.info("API is available")
            self.logger.debug(
                f"Models: {json.dumps(response['data'], indent=2)}"
            )
            if not any(self._model_id == obj["id"] for obj in response["data"]):
                self.logger.warning(
                    f"'{self._model_id}' is not present in the API response!"
                )
                return False
            return True

        self.logger.warning("API is not available")
        return False

    async def _prompt_bool_answer(
        self, prompt: Prompt, record_attempt: int
    ) -> bool:
        return await self._prompt_answer(
            prompt, self.unwrap_bool_answer, record_attempt
        )

    async def _prompt_class_answer(
        self, allow_escape, options, prompt, record_attempt
    ):
        return await self._prompt_answer(
            prompt,
            lambda x: self.unwrap_class_answer(
                x, options, EscapeHatch.WORD if allow_escape else None
            ),
            record_attempt,
        )

    async def _prompt_answer[T](
        self,
        prompt: Prompt,
        parser: Callable[[str], T],
        attempt,
        parse_retries_left=3,
    ) -> T:
        self.logger.info("Trying cache for answer...")
        if cached_answer := self.cache_get(prompt, attempt):
            answer = parser(cached_answer)
            self.logger.info("Cache hit!")
            return answer
        else:
            self.logger.info("Cache miss")

        self.logger.info("Prompting the OpenAI API for an answer...")

        if self._estimation_encoding:
            token_count = len(
                self._estimation_encoding.encode(
                    json.dumps(prompt.prompt_message)
                )
            )
            self._estimated_token_usage[prompt] = token_count
            self.logger.debug(
                f"Estimated token usage for prompt: {token_count}"
            )
        else:
            self.logger.warning(
                "Token usage will not be estimated: unknown model"
            )

        self.logger.debug(
            f"Prompt message {json.dumps(prompt.prompt_message, indent=2)}"
        )

        if isinstance(prompt.prompt_message, str):
            messages = [
                {
                    "role": "system",
                    "message": prompt.prompt_message,
                },
            ]
        else:
            messages = prompt.prompt_message

        try:
            brain_answer = await self._get_completion(
                messages=messages,  # type: ignore
                **(prompt.api_options or {}),
            )
        except openai.APIError as e:
            self.logger.error(f"API error: {e}", exc_info=e)
            raise PrompterError("Failed to get a response from the API")

        response_message = brain_answer.choices[0].message.content
        if self._estimation_encoding:
            token_count = len(
                self._estimation_encoding.encode(response_message)
            )
            self._estimated_token_usage[prompt] = (
                self._estimated_token_usage.get(prompt, 0) + token_count
            )
            self.logger.debug(
                f"Estimated token usage for answer: {token_count}"
            )

        self._actual_token_usage[prompt] = (
            self._actual_token_usage.get(prompt, 0)
            + brain_answer.usage.total_tokens
        )

        self.logger.debug(f"Literal response: {response_message}")
        try:
            answer = parser(response_message)
            self.cache_remember(prompt, response_message, attempt)
            return answer
        except PrompterError as e:
            # Recursively call self if LLM fails to provide a parsable answer
            self.logger.error(f"Error parsing response: {e}")
            if parse_retries_left > 0:
                self.logger.warning(
                    f"Retrying parsing the answer, "
                    f"attempts left: {parse_retries_left}",
                    exc_info=e,
                )
                return await self._prompt_answer(
                    prompt, parser, attempt, parse_retries_left - 1
                )
            raise PrompterError("Failed to parse the answer")

    @retry_exponential
    async def _get_completion(self, messages, **kwargs):
        return await self._client.chat.completions.create(
            messages=messages,  # type: ignore
            model=self._model_id,
            **kwargs,
        )

    def report_usage(self) -> None:
        self.logger.info("Reporting usage to the OpenAI API...")
        if self._estimated_token_usage:
            n_prompts = len(self._estimated_token_usage)
            total_tokens = sum(self._estimated_token_usage.values())
            self.logger.info(
                f"Estimation: reporting {n_prompts} prompts with a total of "
                f"{total_tokens} tokens"
            )
        else:
            self.logger.warning("No estimation of token usage to report")

        n_prompts = len(self._actual_token_usage)
        total_tokens = sum(self._actual_token_usage.values())
        self.logger.info(
            f"Actual: reporting {n_prompts} prompts with a total of "
            f"{total_tokens} tokens"
        )


class OpenAIAzurePrompter(OpenAIPrompter):
    """\
A prompter that interfaces with the OpenAI API using Azure.
"""

    DEFAULT_VERSION = "2024-06-01"

    def _init_client(
        self, http_client: httpx.AsyncClient, api_key: str, azure_endpoint: str
    ):
        self.logger.info("Initializing the Azure API client...")
        self._client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=self.DEFAULT_VERSION,
            http_client=http_client,
        )
        self._ping_headers = {"api-key": self._client.api_key}
