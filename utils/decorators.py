import tenacity
from .logger import LOGGER


## Request retrying decorators
def log_retry_error(state: tenacity.RetryCallState) -> None:
    result = state.outcome
    if result and result.failed:
        exception = result.exception()
        LOGGER.error(f"Retry failed: {result}", exc_info=exception)


retry_fixed = tenacity.retry(
    wait=tenacity.wait_fixed(15),
    stop=tenacity.stop_never,
    retry_error_callback=log_retry_error,
)
