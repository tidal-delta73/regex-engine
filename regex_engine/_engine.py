"""Parsing and matching for the regex_engine vertical slice.

Supported syntax: empty pattern, Unicode literals, concatenation, dot
(matches any single character including newline), backslash escapes of
metacharacters, and character classes (members, ``a-z`` ranges, leading
``^`` negation, escapes for ``]``, ``-``, ``^`` and ``\\``).  Every atom
may be followed by one quantifier: ``?``, ``*``, ``+``, ``{m}``, ``{m,}``
or ``{m,n}``.  Groups, alternation and anchors are rejected at compile
time with :class:`RegexSyntaxError`.
"""

__all__ = ["RegexSyntaxError", "Match", "Pattern", "compile", "fullmatch"]


class RegexSyntaxError(ValueError):
    """Raised when a pattern cannot be compiled.

    The message contains the zero-based offset of the first offending
    character, also available as :attr:`offset`.
    """

    def __init__(self, detail: str, offset: int) -> None:
        super().__init__(f"{detail} at offset {offset}")
        self.offset = offset


class Match:
    """Result of a successful fullmatch."""

    __slots__ = ("_text",)

    def __init__(self, text: str) -> None:
        self._text = text

    def group(self, index: int = 0) -> str:
        if index != 0:
            raise IndexError(f"no such group: {index!r}")
        return self._text

    def start(self) -> int:
        return 0

    def end(self) -> int:
        return len(self._text)

    def span(self) -> tuple[int, int]:
        return (0, len(self._text))

    def __repr__(self) -> str:
        return f"<Match span=(0, {len(self._text)}) match={self._text!r}>"


class Pattern:
    """A compiled pattern, reusable across any number of texts."""

    __slots__ = ("pattern", "_elements")

    def __init__(self, pattern: str, elements: list) -> None:
        self.pattern = pattern
        self._elements = elements

    def fullmatch(self, text: str) -> Match | None:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, not {type(text).__name__}")
        if _run(self._elements, text):
            return Match(text)
        return None

    def __repr__(self) -> str:
        return f"regex_engine.compile({self.pattern!r})"


def compile(pattern: str) -> Pattern:
    """Compile *pattern* into a reusable :class:`Pattern`."""
    if not isinstance(pattern, str):
        raise TypeError(f"pattern must be str, not {type(pattern).__name__}")
    return Pattern(pattern, _parse(pattern))


def fullmatch(pattern: str, text: str) -> Match | None:
    """Compile *pattern* and require it to consume all of *text*."""
    return compile(pattern).fullmatch(text)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_QUANTIFIERS = {"?": (0, 1), "*": (0, None), "+": (1, None)}
_UNSUPPORTED = "()|$^"


def _parse(pattern: str) -> list:
    elements: list = []
    i = 0
    n = len(pattern)
    can_quantify = False
    while i < n:
        c = pattern[i]
        if c in _QUANTIFIERS:
            if not can_quantify:
                if elements:
                    raise RegexSyntaxError("repeated quantifier", i)
                raise RegexSyntaxError("quantifier without preceding atom", i)
            lo, hi = _QUANTIFIERS[c]
            elements[-1] = (elements[-1][0], lo, hi)
            can_quantify = False
            i += 1
        elif c == "{":
            if not can_quantify:
                if elements:
                    raise RegexSyntaxError("repeated quantifier", i)
                raise RegexSyntaxError("quantifier without preceding atom", i)
            lo, hi, i = _parse_braces(pattern, i)
            elements[-1] = (elements[-1][0], lo, hi)
            can_quantify = False
        elif c == "\\":
            if i + 1 >= n:
                raise RegexSyntaxError("dangling backslash", i)
            elements.append((_literal(pattern[i + 1]), 1, 1))
            can_quantify = True
            i += 2
        elif c == ".":
            elements.append((_dot(), 1, 1))
            can_quantify = True
            i += 1
        elif c == "[":
            pred, i = _parse_class(pattern, i)
            elements.append((pred, 1, 1))
            can_quantify = True
        elif c in _UNSUPPORTED:
            raise RegexSyntaxError(f"unsupported metacharacter {c!r}", i)
        else:
            elements.append((_literal(c), 1, 1))
            can_quantify = True
            i += 1
    return elements


def _literal(char: str):
    return lambda c, char=char: c == char


def _dot():
    return lambda c: True


def _is_ascii_digit(c: str) -> bool:
    return "0" <= c <= "9"


def _parse_braces(pattern: str, i: int) -> tuple[int, int | None, int]:
    """Parse ``{m}``, ``{m,}`` or ``{m,n}`` starting at offset *i*."""
    n = len(pattern)
    j = i + 1
    start = j
    while j < n and _is_ascii_digit(pattern[j]):
        j += 1
    if j == start:
        raise RegexSyntaxError("missing lower bound in quantifier", i)
    lo = int(pattern[start:j])
    if j >= n:
        raise RegexSyntaxError("unterminated quantifier", i)
    if pattern[j] == "}":
        return lo, lo, j + 1
    if pattern[j] != ",":
        raise RegexSyntaxError("illegal quantifier bound", i)
    j += 1
    start = j
    while j < n and _is_ascii_digit(pattern[j]):
        j += 1
    if j == start:
        if j >= n:
            raise RegexSyntaxError("unterminated quantifier", i)
        if pattern[j] != "}":
            raise RegexSyntaxError("illegal quantifier bound", i)
        return lo, None, j + 1
    hi = int(pattern[start:j])
    if j >= n:
        raise RegexSyntaxError("unterminated quantifier", i)
    if pattern[j] != "}":
        raise RegexSyntaxError("illegal quantifier bound", i)
    if lo > hi:
        raise RegexSyntaxError("lower bound exceeds upper bound", i)
    return lo, hi, j + 1


def _read_class_char(pattern: str, j: int) -> tuple[str, int]:
    c = pattern[j]
    if c == "\\":
        if j + 1 >= len(pattern):
            raise RegexSyntaxError("dangling backslash", j)
        return pattern[j + 1], j + 2
    return c, j + 1


def _parse_class(pattern: str, i: int) -> tuple[object, int]:
    """Parse a character class starting at ``[`` (offset *i*)."""
    n = len(pattern)
    j = i + 1
    negate = False
    if j < n and pattern[j] == "^":
        negate = True
        j += 1
    ranges: list[tuple[int, int]] = []
    first = True
    while True:
        if j >= n:
            raise RegexSyntaxError("unterminated character class", i)
        if pattern[j] == "]":
            if first:
                raise RegexSyntaxError("empty character class", j)
            j += 1
            break
        first = False
        lo_pos = j
        lo, j = _read_class_char(pattern, j)
        if j < n and pattern[j] == "-" and j + 1 < n and pattern[j + 1] != "]":
            hi, j = _read_class_char(pattern, j + 1)
            if ord(lo) > ord(hi):
                raise RegexSyntaxError("reversed range in character class", lo_pos)
            ranges.append((ord(lo), ord(hi)))
        else:
            ranges.append((ord(lo), ord(lo)))
    ranges = tuple(ranges)
    if negate:
        def pred(c: str, ranges=ranges) -> bool:
            o = ord(c)
            return all(not (a <= o <= b) for a, b in ranges)
    else:
        def pred(c: str, ranges=ranges) -> bool:
            o = ord(c)
            return any(a <= o <= b for a, b in ranges)
    return pred, j


# ---------------------------------------------------------------------------
# Matching (iterative backtracking; every atom consumes exactly one char)
# ---------------------------------------------------------------------------

def _run(elements: list, text: str) -> bool:
    n = len(text)
    m = len(elements)
    if m == 0:
        return n == 0
    counts = [0] * m
    positions = [0] * m
    ei = 0
    pos = 0
    while True:
        pred, lo, hi = elements[ei]
        limit = n - pos if hi is None else min(hi, n - pos)
        k = 0
        while k < limit and pred(text[pos + k]):
            k += 1
        counts[ei] = k
        positions[ei] = pos
        while True:
            c = counts[ei]
            if c >= lo:
                if ei == m - 1:
                    if positions[ei] + c == n:
                        return True
                    counts[ei] = c - 1
                    continue
                ei += 1
                pos = positions[ei - 1] + c
                break
            if ei == 0:
                return False
            ei -= 1
            counts[ei] -= 1
            pos = positions[ei]
