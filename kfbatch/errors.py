"""Public exception hierarchy for kfbatch."""


class KFBatchError(Exception):
    """Base class for expected command-line failures."""


class KFBatchUsageError(KFBatchError):
    """Raised for invalid or unsupported command-line combinations."""


class KFBatchCommandError(KFBatchError):
    """Raised when a scheduler command or captured input cannot be read."""
