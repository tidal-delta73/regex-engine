"""Compile and match the small regex subset supported by this version.

Supported syntax (fullmatch only):
  * literals (any Unicode character), concatenation and the empty pattern
  * "." matching any single character, newline included
  * backslash escapes of metacharacters; an escaped reserved char is literal
  * character classes: [abc], [a-z], [^...]; ], -, ^ and \\ are escapable
  * numbered capturing groups: (...), nestable; empty groups are legal
  * postfix quantifiers ? * + {m} {m,} {m,n}, on atoms or whole groups

Alternation, anchors, named groups and backreferences are rejected at
compile time.
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


class _GroupCounter:
    """Allocates 1-based group numbers in left-parenthesis order."""

    __slots__ = ("next",)

    def __init__(self) -> None:
        self.next = 1

    def alloc(self) -> int:
        gid = self.next
        self.next += 1
        return gid


def _parse_plain_atom(pattern: str, i: int):
    """Parse one non-group atom (the caller handles "(" and ")")."""
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
        _error("alternation is not supported", i)
    if c == "^" or c == "$":
        _error("anchors are not supported", i)
    # Bare "]" and "}" are ordinary literal characters.
    return ("lit", c), i + 1


def _attach_piece(pattern: str, i: int, atom, pieces: list) -> int:
    """Consume the quantifier following ``atom`` and append the piece."""
    n = len(pattern)
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
    return i


def _parse(pattern: str):
    """Parse with an explicit open-group stack, so nesting depth is bounded
    by heap memory rather than the Python call stack.

    Each stack entry records a group's number, its opening parenthesis and
    the piece list that owns it; the current piece list is the innermost
    open group's, or the root sequence's.
    """
    n = len(pattern)
    groups = _GroupCounter()
    root: list = []
    stack: list[tuple[int, int, list]] = []
    pieces = root
    i = 0
    while i < n:
        c = pattern[i]
        if c == "(":
            gid = groups.alloc()
            stack.append((gid, i, pieces))
            pieces = []
            i += 1
            continue
        if c == ")":
            if not stack:
                _error("unbalanced parenthesis", i)
            gid, _open_pos, parent = stack.pop()
            atom = ("grp", (gid, tuple(pieces)))
            pieces = parent
            i = _attach_piece(pattern, i + 1, atom, pieces)
            continue
        atom, i = _parse_plain_atom(pattern, i)
        i = _attach_piece(pattern, i, atom, pieces)
    if stack:
        # The deepest still-open parenthesis is the one that never closed.
        _error("unbalanced parenthesis", stack[-1][1])
    return tuple(root), groups.next - 1


# ---------------------------------------------------------------------------
# Matching
#
# A greedy backtracking NFA driven by an explicit frame stack, so neither
# pattern nesting nor repetition length is limited by the Python call
# stack.  Captures live in one flat array (caps[g] is None for a group
# that has not participated, else a (start, end) pair; index 0 is unused).
#
# Backtracking correctness for captures relies on whole-array snapshots:
# a group enumerator remembers the capture state at its entry and at each
# sub-sequence yield, and restores the exact snapshot before resuming a
# suspended child, so a resumed enumerator always sees the state it last
# produced and captures of abandoned attempts never leak.  A nested group
# that simply does not participate in a later repetition thereby keeps the
# capture of its most recent earlier participation.
#
# S ("sequence") frames walk a list of pieces.  Every processed piece
# leaves a record: ("A", start, count) for a single-character atom already
# stretched to its greedy count, or ("G", frame) for a suspended group
# enumerator.  Backtracking pops records: atom records are shortened by
# one, group records are resumed to offer their next shorter end.
#
# G ("group") frames are pure enumerators for one piece (sub){lo,hi} at a
# fixed base position.  They yield successive end positions, longest
# first, and (via the snapshots) present the correct captures for each;
# they never consume the pieces following the group -- the parent S frame
# does.  A sub-sequence that matches the empty string stops expansion at
# once, so empty-able repeats terminate instead of looping.
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


_S_RUN, _S_WAIT, _S_END = "run", "wait", "end"
# G phases.  "drive" while in a phase named below means "the position last
# yielded in that phase was rejected upstream -- produce the next one".
_G_SUB = "sub"        # (re)start / resume the sub-sequence enumerator
_G_EXT = "ext"        # the one-more-repetition child enumerator is open
_G_SETTLE = "settle"  # yielded the current repetition's end; shorten sub
_G_EMPTY = "empty"    # yielded one empty match; fall back to no repetition
_G_FINAL = "final"    # yielded "stop without another repetition"; done


class _Runner:
    def __init__(self, program, text: str) -> None:
        self.pieces, self.ngroups = program
        self.text = text
        self.n = len(text)
        self.caps: list = [None] * (self.ngroups + 1)

    def _snap(self):
        return list(self.caps)

    def _restore(self, snapshot) -> None:
        self.caps[:] = snapshot

    def _seq_frame(self, pieces, pos: int):
        # ["S", pieces, pos, records, phase, gframe]
        return ["S", pieces, pos, [], _S_RUN, None]

    def _group_frame(self, pieces, k: int, pos: int, count: int, entry):
        # ["G", pieces, k, base, count, phase, subf, extf,
        #  entry, repstate, sub_end]
        return ["G", pieces, k, pos, count, _G_SUB, None, None,
                entry, entry, pos]

    # -- S frame ------------------------------------------------------------

    def _s_backtrack(self, f):
        records = f[3]
        while records:
            if records[-1][0] == "G":
                gframe = records.pop()[1]
                f[5] = gframe
                f[4] = _S_WAIT
                return ("push", gframe)
            _, start, count = records.pop()
            pieces = f[1]
            j = len(records)
            lo = pieces[j][1]
            count -= 1
            if count >= lo:
                records.append(("A", start, count))
                f[2] = start + count
                f[4] = _S_RUN
                return "again"
        f[4] = _S_END
        return "fail"

    def _step_s(self, f, event):
        _, pieces, pos, records, phase, _gf = f
        if phase == _S_WAIT:
            if event[0] == "yield":
                records.append(("G", event[2]))
                f[2] = event[1]
                f[4] = _S_RUN
                f[5] = None
                return "again"
            return self._s_backtrack(f)
        if phase == _S_END:
            return self._s_backtrack(f)
        # _S_RUN
        while True:
            k = len(records)
            pos = f[2]
            if k == len(pieces):
                f[4] = _S_END
                return ("yield", pos)
            atom, lo, hi = pieces[k]
            if atom[0] != "grp":
                start = pos
                count = 0
                while (
                    count < hi
                    and pos < self.n
                    and _atom_matches(atom, self.text[pos])
                ):
                    pos += 1
                    count += 1
                if count < lo:
                    return self._s_backtrack(f)
                records.append(("A", start, count))
                f[2] = pos
                continue
            f[4] = _S_WAIT
            f[5] = None
            entry = self._snap()
            return ("push",
                    self._group_frame(pieces, k, pos, 0, entry))

    # -- G frame ------------------------------------------------------------

    def _step_g(self, f, event):
        (_, pieces, k, base, count, phase, subf, extf,
         entry, repstate, sub_end) = f
        gid, sub = pieces[k][0][1]
        lo, hi = pieces[k][1], pieces[k][2]

        if event[0] == "drive":
            if phase == _G_SUB:
                # A suspended G frame never keeps phase == sub (it always
                # moves to ext/settle/empty before yielding), so reaching
                # here on "drive" is a fresh entry at repetition count 0.
                # With an upper bound of zero the group never participates.
                if count >= hi:
                    self._restore(entry)
                    f[5] = _G_FINAL
                    return ("yield", base)
                child = self._seq_frame(sub, base)
                f[6] = child
                return ("push", child)
            if phase == _G_EXT:
                return ("push", extf)
            if phase == _G_SETTLE:
                # Give up the currently offered repetition and ask sub for
                # a shorter match, restoring the state sub last produced.
                self._restore(repstate)
                f[5] = _G_SUB
                return ("push", subf)
            if phase == _G_EMPTY:
                # The empty match was rejected; stop without it, if allowed.
                self._restore(entry)
                if count >= lo:
                    f[5] = _G_FINAL
                    return ("yield", base)
                return "fail"
            return "fail"  # _G_FINAL: the settlement has no alternative

        if event[0] == "fail":
            if phase == _G_EXT:
                # No further repetition can follow the current sub match;
                # the failed child already restored its own captures, so the
                # current repetition's capture is intact.  Settle on it if
                # the minimum is met, otherwise shorten the sub match.
                if count + 1 >= lo:
                    f[5] = _G_SETTLE
                    return ("yield", sub_end)
                self._restore(repstate)
                f[5] = _G_SUB
                return ("push", subf)
            # The sub-sequence offers no end position at this base.  Settle
            # by taking no further repetition if the minimum is already met,
            # otherwise the whole enumeration fails; in both cases captures
            # return to the state on entry.
            self._restore(entry)
            if count >= lo:
                f[5] = _G_FINAL
                return ("yield", base)
            return "fail"

        # event == ("yield", end, frame)
        end = event[1]
        child_frame = event[2]
        if phase == _G_EXT:
            f[7] = child_frame
            f[10] = end
            f[5] = _G_EXT
            return ("yield", end)

        # The sub-sequence finished at ``end``.  Remember the exact capture
        # state it produced so a later shortening resumes it consistently,
        # then record this group's repetition.
        f[6] = child_frame
        f[9] = self._snap()
        f[10] = end
        self.caps[gid] = (base, end)
        if end == base:
            # Empty match: keep this capture, stop expanding immediately and
            # offer the position once.  The minimum is waived for an empty
            # match so repeats containing an empty-able sub-expression end.
            f[5] = _G_EMPTY
            return ("yield", end)
        if count + 1 < hi:
            ext = self._group_frame(
                pieces, k, end, count + 1, self._snap())
            f[7] = ext
            f[5] = _G_EXT
            return ("push", ext)
        # Last permitted repetition: settle on it, leaving a shorter sub
        # match available as the next alternative.
        f[5] = _G_SETTLE
        return ("yield", end)

    # -- driver -------------------------------------------------------------

    def run(self):
        stack = [self._seq_frame(self.pieces, 0)]
        event = ("drive",)
        while True:
            top = stack[-1]
            action = (self._step_s if top[0] == "S" else self._step_g)(
                top, event)
            if action == "again":
                event = ("drive",)
                continue
            kind = action[0]
            if kind == "push":
                stack.append(action[1])
                event = ("drive",)
                continue
            if kind == "yield":
                end = action[1]
                yielded = stack.pop()
                if not stack:
                    if end == self.n:
                        return list(self.caps)  # independent copy
                    # Rejected by the full-match boundary; keep searching.
                    stack.append(yielded)
                    event = ("drive",)
                    continue
                event = ("yield", end, yielded)
                continue
            # fail
            failed = stack.pop()
            if failed[0] == "G":
                self._restore(failed[8])
            if not stack:
                return None
            event = ("fail",)


def _run(program, text: str):
    return _Runner(program, text).run()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Match:
    """A successful full match; the matched span is always the whole text."""

    __slots__ = ("string", "_spans")

    def __init__(self, text: str, spans) -> None:
        self.string = text
        # spans[g] is (start, end) for a participating group, else None.
        self._spans = tuple(spans)

    def _check(self, index: int = 0) -> int:
        # Baseline raises IndexError for any index other than 0, including
        # non-integers; keep that channel rather than introducing TypeError.
        if not isinstance(index, int) or index < 0 or index >= len(self._spans):
            raise IndexError("no such group")
        return index

    def group(self, index: int = 0):
        index = self._check(index)
        if index == 0:
            return self.string
        span = self._spans[index]
        return None if span is None else self.string[span[0] : span[1]]

    def groups(self) -> tuple:
        return tuple(
            None if span is None else self.string[span[0] : span[1]]
            for span in self._spans[1:]
        )

    def start(self, index: int = 0) -> int:
        index = self._check(index)
        if index == 0:
            return 0
        span = self._spans[index]
        return -1 if span is None else span[0]

    def end(self, index: int = 0) -> int:
        index = self._check(index)
        if index == 0:
            return len(self.string)
        span = self._spans[index]
        return -1 if span is None else span[1]

    def span(self, index: int = 0) -> tuple[int, int]:
        index = self._check(index)
        if index == 0:
            return (0, len(self.string))
        span = self._spans[index]
        return (-1, -1) if span is None else (span[0], span[1])

    def __repr__(self) -> str:
        return f"<Match {self.string!r}>"


class Pattern:
    """A compiled pattern, reusable across any number of texts."""

    __slots__ = ("pattern", "_program")

    def __init__(self, pattern: str, program) -> None:
        self.pattern = pattern
        self._program = program

    def fullmatch(self, text: str):
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        spans = _run(self._program, text)
        if spans is None:
            return None
        spans[0] = (0, len(text))
        return Match(text, spans)

    def __repr__(self) -> str:
        return f"<Pattern {self.pattern!r}>"


def compile(pattern: str) -> Pattern:
    """Compile a pattern string into a reusable :class:`Pattern`."""
    if not isinstance(pattern, str):
        raise TypeError("pattern must be a str")
    pieces, ngroups = _parse(pattern)
    return Pattern(pattern, (pieces, ngroups))


def fullmatch(pattern: str, text: str):
    """Compile ``pattern`` and match it against the whole of ``text``."""
    return compile(pattern).fullmatch(text)
