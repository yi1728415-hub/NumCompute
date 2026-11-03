"""Custom exceptions shared by all numcompute_stream modules."""


class NotFittedError(RuntimeError):
    """Raised when a model or transformer is used before it has been trained.

    In the streaming setting this happens when ``predict`` is called before the
    first buffered retrain, i.e. before ``buffer_size`` samples have arrived.
    """
