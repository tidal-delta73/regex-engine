"""Command line entry point: version, help and fullmatch."""
import json
import sys

from . import __version__
from .core import RegexSyntaxError, compile

USAGE = """usage: python3 -m regex_engine <command>

commands:
  version   print the package version
  help      print this message
  fullmatch PATTERN TEXT
            compile PATTERN, fullmatch it against TEXT, print JSON
"""


def _fullmatch(pattern: str, text: str) -> int:
    try:
        compiled = compile(pattern)
    except RegexSyntaxError as exc:
        result = {
            "error": "RegexSyntaxError",
            "message": str(exc),
            "pos": exc.pos,
        }
        print(json.dumps(result, separators=(",", ":")), file=sys.stderr)
        return 2
    match = compiled.fullmatch(text)
    if match is None:
        result = {"matched": False, "groups": None, "spans": None}
        print(json.dumps(result, separators=(",", ":")))
        return 1
    count = len(match.groups()) + 1
    result = {
        "matched": True,
        "groups": [match.group(i) for i in range(count)],
        "spans": [list(match.span(i)) for i in range(count)],
    }
    print(json.dumps(result, separators=(",", ":")))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "help"
    if command == "version":
        print(__version__)
        return 0
    if command in {"help", "-h", "--help"}:
        print(USAGE, end="")
        return 0
    if command == "fullmatch":
        if len(args) != 3:
            print(USAGE, end="", file=sys.stderr)
            return 2
        return _fullmatch(args[1], args[2])
    print(f"unknown command: {command}", file=sys.stderr)
    print(USAGE, end="", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
