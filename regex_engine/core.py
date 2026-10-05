"""Compile and match the small regex subset supported by this version.

Supported syntax (fullmatch only):
  * literals (any Unicode character), concatenation and the empty pattern
  * "." matching any single character, newline included
  * backslash escapes of metacharacters; an escaped reserved char is literal
  * character classes: [abc], [a-z], [^...]; ], -, ^ and \\ are escapable
  * postfix quantifiers: ? * + {m} {m,} {m,n}

Groups, alternation and anchors are rejected at compile time.
"""

from __future__ import annotations

__all__ = ["RegexSyntaxError", "Match", "Pattern", "compile", "fullmatch"]

_INF = float("inf")
_DIGITS = frozenset("0123456789")
_QUANTIFIER_STARTS = frozenset("?*+{")
_SIMPLE_BOUNDS = {"?": (0, 1), "*": (0, _INF), "+": (1, _INF)}


class RegexSyntaxError(ValueError):
    """A pattern could not be compiled.

    ``pos`` is the zero-based offset of the first character at which the
    pattern is invalid; the offset is also included in the error message.
    """

    def __init__(self, message: str, pos: int) -> None:
        self.pos = pos
        super().__init__(f"{message} at position {pos}")


def _error(message: str, pos: int):
    raise RegexSyntaxError(message, pos)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_braces(pattern: str, i: int):
    """Parse a {m}, {m,} or {m,n} quantifier starting at pattern[i] == "{"."""
    start = i
    n = len(pattern)
    j = i + 1
    if j >= n or pattern[j] not in _DIGITS:
        _error("invalid quantifier bounds", start)
    while j < n and pattern[j] in _DIGITS:
        j += 1
    lo = int(pattern[i + 1 : j])
    if j >= n:
        _error("unclosed quantifier bounds", start)
    if pattern[j] == "}":
        return lo, lo, j + 1
    if pattern[j] != ",":
        _error("invalid quantifier bounds", start)
    j += 1
    k = j
    while k < n and pattern[k] in _DIGITS:
        k += 1
    if k == j:
        if k < n and pattern[k] == "}":
            return lo, _INF, k + 1
        _error("invalid quantifier bounds", start)
    hi = int(pattern[j:k])
    if k >= n or pattern[k] != "}":
        _error("invalid quantifier bounds", start)
    if hi < lo:
        _error("quantifier upper bound is smaller than lower bound", start)
    return lo, hi, k + 1


def _parse_class_char(pattern: str, i: int):
    """Read one member character; returns (char, next_index, char_offset)."""
    if pattern[i] == "\\":
        if i + 1 >= len(pattern):
            _error("dangling backslash in character class", i)
        return pattern[i + 1], i + 2, i
    return pattern[i], i + 1, i


def _parse_class(pattern: str, i: int):
    """Parse a character class starting at pattern[i] == "["."""
    start = i
    n = len(pattern)
    i += 1  # consume "["
    negated = False
    if i < n and pattern[i] == "^":
        negated = True
        i += 1
    chars: set[str] = set()
    ranges: list[tuple[str, str]] = []
    members = 0
    while True:
        if i >= n:
            _error("unterminated character class", start)
        if pattern[i] == "]":
            if members == 0:
                _error("empty character class", start)
            i += 1
            break
        lo, i, lo_pos = _parse_class_char(pattern, i)
        members += 1
        if i + 1 < n and pattern[i] == "-" and pattern[i + 1] != "]":
            hi, i, _ = _parse_class_char(pattern, i + 1)
            members += 1
            if lo > hi:
                _error(f"reversed character range {lo!r}-{hi!r}", lo_pos)
            ranges.append((lo, hi))
        else:
            chars.add(lo)
    atom = ("cls", (negated, frozenset(chars), tuple(ranges)))
    return atom, i


def _parse_atom(pattern: str, i: int):
    c = pattern[i]
    if c == "\\":
        if i + 1 >= len(pattern):
            _error("dangling backslash", i)
        return ("lit", pattern[i + 1]), i + 2
    if c == ".":
        return ("any", None), i + 1
    if c == "[":
        return _parse_class(pattern, i)
    if c in "?*+":
        _error("quantifier with no preceding atom", i)
    if c == "{":
        # Validate the bounds shape first, but a brace quantifier still
        # needs an atom in front of it.
        _parse_braces(pattern, i)
        _error("quantifier with no preceding atom", i)
    if c in "()|":
        _error(f"groups and alternation are not supported ({c!r})", i)
    if c == "^" or c == "$":
        _error("anchors are not supported", i)
    # Bare "]" and "}" are ordinary literal characters.
    return ("lit", c), i + 1


def _parse(pattern: str):
    pieces = []
    i, n = 0, len(pattern)
    while i < n:
        atom, i = _parse_atom(pattern, i)
        lo, hi = 1, 1
        if i < n and pattern[i] in _SIMPLE_BOUNDS:
            lo, hi = _SIMPLE_BOUNDS[pattern[i]]
            i += 1
            if i < n and pattern[i] in _QUANTIFIER_STARTS:
                _error("multiple quantifiers on one atom", i)
        elif i < n and pattern[i] == "{":
            lo, hi, i = _parse_braces(pattern, i)
            if i < n and pattern[i] in _QUANTIFIER_STARTS:
                _error("multiple quantifiers on one atom", i)
        pieces.append((atom, lo, hi))
    return pieces


# ---------------------------------------------------------------------------
# Matching (greedy backtracking; every atom consumes exactly one character,
# so repetition loops are always bounded by the length of the text)
# ---------------------------------------------------------------------------


def _atom_matches(atom, ch: str) -> bool:
    kind = atom[0]
    if kind == "lit":
        return ch == atom[1]
    if kind == "any":
        return True
    negated, members, ranges = atom[1]
    found = ch in members
    if not found:
        for lo, hi in ranges:
            if lo <= ch <= hi:
                found = True
                break
    return (not found) if negated else found


def _run(pieces, text: str) -> bool:
    """Greedy backtracking over an explicit stack.

    One frame per piece; ``counts[k]`` is the repetition count currently
    chosen for piece ``k``. Every repetition consumes exactly one Unicode
    character, so all counts together never exceed ``len(text)`` and the
    loop cannot hang on zero repetitions or empty text.
    """
    n = len(text)
    k = len(pieces)
    counts = [0] * k
    # 0 while greedily extending, 1 once the rest has been launched at the
    # current count (returning then means trying one fewer repetition).
    launched = bytearray(k)
    pi = 0
    pos = 0
    while True:
        if pi == k:
            if pos == n:
                return True
        else:
            atom, lo, hi = pieces[pi]
            if not launched[pi]:
                c = counts[pi]
                if c < hi and pos < n and _atom_matches(atom, text[pos]):
                    counts[pi] = c + 1
                    pos += 1
                    continue
                launched[pi] = 1  # extension exhausted; settle if allowed
            if counts[pi] >= lo:
                pi += 1
                if pi < k:
                    counts[pi] = 0
                    launched[pi] = 0
                continue
        # Current frame (or the terminal) failed: backtrack.
        while True:
            if pi == k:
                pi -= 1
                if pi < 0:
                    return False
                continue
            c = counts[pi]
            lo = pieces[pi][1]
            if launched[pi] and c - 1 >= lo:
                counts[pi] = c - 1
                pos -= 1
                pi += 1  # relaunch the rest at the smaller count
                if pi < k:
                    counts[pi] = 0
                    launched[pi] = 0
                break
            pos -= c
            counts[pi] = 0
            launched[pi] = 0
            pi -= 1
            if pi < 0:
                return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Match:
    """A successful full match; the matched span is always the whole text."""

    __slots__ = ("string",)

    def __init__(self, text: str) -> None:
        self.string = text

    def group(self, index: int = 0) -> str:
        if index != 0:
            raise IndexError("no such group")
        return self.string

    def start(self) -> int:
        return 0

    def end(self) -> int:
        return len(self.string)

    def span(self) -> tuple[int, int]:
        return (0, len(self.string))

    def __repr__(self) -> str:
        return f"<Match {self.string!r}>"


class Pattern:
    """A compiled pattern, reusable across any number of texts."""

    __slots__ = ("pattern", "_pieces")

    def __init__(self, pattern: str, pieces) -> None:
        self.pattern = pattern
        self._pieces = pieces

    def fullmatch(self, text: str):
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        if _run(self._pieces, text):
            return Match(text)
        return None

    def __repr__(self) -> str:
        return f"<Pattern {self.pattern!r}>"


def compile(pattern: str) -> Pattern:
    """Compile a pattern string into a reusable :class:`Pattern`."""
    if not isinstance(pattern, str):
        raise TypeError("pattern must be a str")
    return Pattern(pattern, _parse(pattern))


def fullmatch(pattern: str, text: str):
    """Compile ``pattern`` and match it against the whole of ``text``."""
    return compile(pattern).fullmatch(text)
