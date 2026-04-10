import json
from .logger import LOGGER


## Exceptions
class ProfileMark(Exception):
    """Interrupts flow of the program at arbitrary point for profiling"""


class BouzygesError(Exception):
    """Base class for Bouzyges errors."""


class SnowstormAPIError(Exception):
    """Raised when the Snowstorm API returns a bad response."""


class SnowstormRequestError(SnowstormAPIError):
    """Raised when the Snowstorm API returns a non-200 response"""

    def __init__(self, text, response, *_):
        super().__init__(text)
        self.response = response

    @classmethod
    def from_response(cls, response):
        LOGGER.error(
            f"Request: {response.request.method}, {response.request.url}"
        )
        if response:
            LOGGER.error(f"Response: {json.dumps(response.json(), indent=2)}")
        return cls(
            f"Snowstorm API returned {response.status_code} status code",
            response,
        )


class PrompterError(Exception):
    """Raised when the prompter encounters an error."""


class PrompterInitError(PrompterError):
    """Raised when prompter can not be initialized"""
