__version__ = "0.1.0"

from .core import Match, Pattern, RegexSyntaxError, compile, fullmatch

__all__ = [
    "__version__",
    "RegexSyntaxError",
    "Match",
    "Pattern",
    "compile",
    "fullmatch",
]
