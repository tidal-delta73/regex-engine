"""Compile and match the small regex subset supported by this version.

Supported syntax (fullmatch only):
  * literals (any Unicode character), concatenation and the empty pattern
  * "." matching any single character, newline included
  * backslash escapes of metacharacters; an escaped reserved char is literal
  * character classes: [abc], [a-z], [^...]; ], -, ^ and \\ are escapable
  * postfix quantifiers: ? * + {m} {m,} {m,n}
  * numbered capturing groups "(...)", nestable, quantifiable as a whole;
    group 0 is the whole match, groups are numbered from 1 by their "("

Alternation and anchors are rejected at compile time.
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
    if c == "|":
        _error("alternation is not supported ('|')", i)
    if c == "^" or c == "$":
        _error("anchors are not supported", i)
    # Bare "]" and "}" are ordinary literal characters.
    return ("lit", c), i + 1


def _parse(pattern: str):
    """Parse into (pieces, group_count).

    A piece is ``(node, lo, hi)``; a node is an atom tuple or
    ``("group", number, body_pieces)``.  Groups are numbered from 1 in
    left-parenthesis order.  The scan is iterative so deeply nested
    parentheses cannot exhaust the call stack.
    """
    pieces = []
    # Each open frame is (position of "(", group number, outer pieces).
    open_frames = []
    ngroups = 0
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "(":
            ngroups += 1
            open_frames.append((i, ngroups, pieces))
            pieces = []
            i += 1
            continue
        if c == ")":
            if not open_frames:
                _error("unmatched ')'", i)
            _, number, outer = open_frames.pop()
            atom = ("group", number, pieces)
            pieces = outer
            i += 1
        else:
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
    if open_frames:
        _error("unclosed '('", open_frames[-1][0])
    return pieces, ngroups


# ---------------------------------------------------------------------------
# Compiling the tree to a flat program
# ---------------------------------------------------------------------------
#
# Instructions:
#   ("char", atom)                  consume one character matching the atom
#   ("save", slot)                  record pos into capture slot
#   ("rep_begin", rid, lo, hi, exit_pc)
#   ("rep_end", rid, begin_pc)      counted greedy repetition of the body
#   ("match",)                      accept iff pos == len(text)
#
# ``rid`` identifies a repetition; runtime counters are snapshotted into
# every backtracking entry so abandoned attempts never leak state.


def _build(pieces):
    prog = []
    nreps = [0]

    def emit_node(node):
        if node[0] == "group":
            _, number, body = node
            prog.append(("save", 2 * number))
            emit_seq(body)
            prog.append(("save", 2 * number + 1))
        else:
            prog.append(("char", node))

    def emit_seq(seq):
        for node, lo, hi in seq:
            if lo == 1 and hi == 1:
                emit_node(node)
                continue
            rid = nreps[0]
            nreps[0] += 1
            begin = len(prog)
            prog.append(["rep_begin", rid, lo, hi, None])
            emit_node(node)
            prog.append(("rep_end", rid, begin))
            prog[begin][4] = len(prog)  # exit: instruction after rep_end
            prog[begin] = tuple(prog[begin])

    emit_seq(pieces)
    prog.append(("match",))
    return prog, nreps[0]


# ---------------------------------------------------------------------------
# Matching: a backtracking VM over an explicit stack.  Backtracking entries
# snapshot the capture slots and repetition counters, so state abandoned by
# backtracking is fully restored.  A repetition whose body succeeds without
# advancing keeps its captures, stops expanding and lets the rest run, which
# keeps loops over nullable bodies from hanging.
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


def _run(prog, nreps: int, ngroups: int, text: str):
    """Run the program; return the capture slots on success, else None."""
    n = len(text)
    caps = [None] * (2 * (ngroups + 1))
    counters = [0] * nreps
    entered_at = [0] * nreps
    stack = []
    pc = 0
    pos = 0
    while True:
        op = prog[pc]
        kind = op[0]
        if kind == "char":
            if pos < n and _atom_matches(op[1], text[pos]):
                pos += 1
                pc += 1
                continue
        elif kind == "save":
            caps[op[1]] = pos
            pc += 1
            continue
        elif kind == "rep_begin":
            _, rid, lo, hi, exit_pc = op
            count = counters[rid]
            if count < hi:
                if count >= lo:
                    # Greedy: iterate now, but remember we may also stop
                    # here.  The snapshot resets this loop's counter so a
                    # later re-entry (from an enclosing repetition) starts
                    # at zero.
                    snapshot = list(counters)
                    snapshot[rid] = 0
                    stack.append(
                        (exit_pc, pos, tuple(caps), tuple(snapshot),
                         tuple(entered_at))
                    )
                counters[rid] = count + 1
                entered_at[rid] = pos
                pc += 1
            else:
                counters[rid] = 0
                pc = exit_pc
            continue
        elif kind == "rep_end":
            _, rid, begin_pc = op
            if pos == entered_at[rid]:
                # Zero-width iteration: keep its captures, stop expanding.
                counters[rid] = 0
                pc += 1
            else:
                pc = begin_pc
            continue
        else:  # "match"
            if pos == n:
                return caps
        # Failure: backtrack to the most recent alternative.
        if not stack:
            return None
        pc, pos, caps_t, counters_t, entered_t = stack.pop()
        caps = list(caps_t)
        counters = list(counters_t)
        entered_at = list(entered_t)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Match:
    """A successful full match; group 0 is always the whole text."""

    __slots__ = ("string", "_spans")

    def __init__(self, text: str, spans) -> None:
        self.string = text
        # spans[g] is (start, end), or None when group g did not participate.
        self._spans = spans

    def _span_of(self, index: int):
        if index < 0 or index >= len(self._spans):
            raise IndexError("no such group")
        return self._spans[index]

    def group(self, index: int = 0):
        span = self._span_of(index)
        if span is None:
            return None
        return self.string[span[0] : span[1]]

    def groups(self) -> tuple:
        return tuple(self.group(i) for i in range(1, len(self._spans)))

    def start(self, index: int = 0) -> int:
        span = self._span_of(index)
        return -1 if span is None else span[0]

    def end(self, index: int = 0) -> int:
        span = self._span_of(index)
        return -1 if span is None else span[1]

    def span(self, index: int = 0) -> tuple[int, int]:
        span = self._span_of(index)
        return (-1, -1) if span is None else span

    def __repr__(self) -> str:
        return f"<Match {self.string!r}>"


class Pattern:
    """A compiled pattern, reusable across any number of texts."""

    __slots__ = ("pattern", "_prog", "_nreps", "_ngroups")

    def __init__(self, pattern: str, prog, nreps: int, ngroups: int) -> None:
        self.pattern = pattern
        self._prog = prog
        self._nreps = nreps
        self._ngroups = ngroups

    def fullmatch(self, text: str):
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        caps = _run(self._prog, self._nreps, self._ngroups, text)
        if caps is None:
            return None
        spans = [(0, len(text))]
        for number in range(1, self._ngroups + 1):
            start = caps[2 * number]
            spans.append(None if start is None else (start, caps[2 * number + 1]))
        return Match(text, spans)

    def __repr__(self) -> str:
        return f"<Pattern {self.pattern!r}>"


def compile(pattern: str) -> Pattern:
    """Compile a pattern string into a reusable :class:`Pattern`."""
    if not isinstance(pattern, str):
        raise TypeError("pattern must be a str")
    pieces, ngroups = _parse(pattern)
    prog, nreps = _build(pieces)
    return Pattern(pattern, prog, nreps, ngroups)


def fullmatch(pattern: str, text: str):
    """Compile ``pattern`` and match it against the whole of ``text``."""
    return compile(pattern).fullmatch(text)
