"""Public exception hierarchy for kfbatch."""


class KFBatchError(Exception):
    """Base class for expected command-line failures."""


class KFBatchUsageError(KFBatchError):
    """Raised for invalid or unsupported command-line combinations."""


class KFBatchCommandError(KFBatchError):
    """Raised when a scheduler command or captured input cannot be read."""

    def __init__(
        self,
        message,
        *,
        command_name="",
        argv=None,
        returncode=None,
        timed_out=False,
        output_limited=False,
    ):
        super().__init__(message)
        self.command_name = command_name
        self.argv = tuple(argv or ())
        self.returncode = returncode
        self.timed_out = bool(timed_out)
        self.output_limited = bool(output_limited)
