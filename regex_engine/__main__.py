"""Command line entry point: ``version``, ``fullmatch`` and ``help``."""
import json
import sys

from . import __version__, RegexSyntaxError, compile

USAGE = """usage: python3 -m regex_engine <command>

commands:
  version                print the package version
  fullmatch PATTERN TEXT match PATTERN against all of TEXT and print JSON
  help                   print this message
"""


def _compact(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))


def _fullmatch(pattern: str, text: str) -> int:
    try:
        match = compile(pattern).fullmatch(text)
    except RegexSyntaxError as exc:
        print(_compact({"error": "RegexSyntaxError",
                        "message": str(exc), "pos": exc.pos}),
              file=sys.stderr)
        return 2
    if match is None:
        print(_compact({"matched": False, "groups": None, "spans": None}))
        return 1
    count = len(match._spans)
    print(_compact({
        "matched": True,
        "groups": [match.group(i) for i in range(count)],
        "spans": [list(match.span(i)) for i in range(count)],
    }))
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
