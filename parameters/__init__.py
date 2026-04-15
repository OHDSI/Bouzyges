import json
import os

import pydantic

from utils.logger import LOGGER
from utils.types import OutFormat, PrompterOption, Url


class ProfilingParameters(pydantic.BaseModel):
    """\
Parameters to control profiling of the program.
"""

    enabled: bool
    stop_profiling_after_seconds: int | None


class LoggingParameters(pydantic.BaseModel):
    """\
Parameters to control logging.
"""

    log_to_file: bool
    logging_level: int

    def update(self, level):
        self.logging_level = level
        LOGGER.setLevel(level=self.logging_level)


class APIParameters(pydantic.BaseModel):
    """\
Parameters to control the interface to Snowstorm, cache and LLMs.
"""

    prompter: PrompterOption
    repeat_prompts: int | None
    snowstorm_url: Url
    llm_model_id: str
    cache_db: str | None
    max_concurrent_workers: int


class EnvironmentParameters(pydantic.BaseModel):
    """\
Parameters that reflect the environment variables.
"""

    OPENAI_API_KEY: str | None = None
    AZURE_API_KEY: str | None = None
    AZURE_API_ENDPOINT: str | None = None

    def fill_from_env(self) -> None:
        for env in self.model_fields:
            env_value = os.getenv(env)
            setattr(self, env, env_value or None)


class IOParameters(pydantic.BaseModel):
    """\
Parameters for reading and writing CSV file data.
"""

    file: str
    sep: str
    quotechar: str
    quoting: int


class RunParameters(pydantic.BaseModel):
    """\
Parameters for the run of the program.
"""

    api: APIParameters
    env: EnvironmentParameters = pydantic.Field(
        exclude=True, default_factory=EnvironmentParameters
    )
    log: LoggingParameters
    prof: ProfilingParameters
    read: IOParameters
    write: IOParameters
    format: OutFormat
    out_dir: str = pydantic.Field(default_factory=os.getcwd)

    @classmethod
    def from_file(cls, file: str) -> RunParameters:
        with open(file, "r") as f:
            json_data = json.load(f)
        params = RunParameters(**json_data)
        # Environment variables are not stored in JSON
        params.env.fill_from_env()
        return params

    def update(self, json_data: dict) -> None:
        self.__init__(**json_data)
        self.log.update(self.log.logging_level)

    def save(self, file: str) -> None:
        with open(file, "w") as f:
            json.dump(self.model_dump(), f, indent=2)


# Default
PARAMS = RunParameters.model_validate(
    {
        "api": {
            "prompter": "openai",
            "repeat_prompts": None,
            "snowstorm_url": "http://localhost:8080/",
            "llm_model_id": "gpt-4o-mini",
            "cache_db": "prompt_cache.db",
            "max_concurrent_workers": 8,
        },
        "log": {"log_to_file": True, "logging_level": 10},
        "prof": {"enabled": False, "stop_profiling_after_seconds": None},
        "read": {
            "file": "Test.csv",
            "sep": "Tab",
            "quotechar": '"',
            "quoting": 0,
        },
        "write": {
            "file": "bouzyges_output.json",
            "sep": "Tab",
            "quotechar": '"',
            "quoting": 0,
        },
        "format": "JSON",
    }
)
