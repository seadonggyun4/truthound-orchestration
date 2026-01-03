"""Output handlers for Kestra data quality integration.

This package provides handlers for formatting and sending data quality
results to Kestra's output system and other destinations.

Example:
    >>> from truthound_kestra.outputs import (
    ...     send_check_result,
    ...     KestraOutputHandler,
    ... )
    >>>
    >>> # Simple function
    >>> send_check_result(result)
    >>>
    >>> # Custom handler
    >>> handler = KestraOutputHandler(config=OutputConfig(...))
    >>> handler.send_check_result(result)
"""

from truthound_kestra.outputs.handlers import (
    FileOutputHandler,
    KestraOutputHandler,
    MultiOutputHandler,
    OutputConfig,
    OutputHandlerProtocol,
    send_check_result,
    send_learn_result,
    send_outputs,
    send_profile_result,
)

__all__ = [
    # Protocols
    "OutputHandlerProtocol",
    # Handlers
    "KestraOutputHandler",
    "FileOutputHandler",
    "MultiOutputHandler",
    # Config
    "OutputConfig",
    # Functions
    "send_outputs",
    "send_check_result",
    "send_profile_result",
    "send_learn_result",
]
