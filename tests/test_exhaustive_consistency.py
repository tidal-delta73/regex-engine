"""Exhaustive matching-semantics consistency tests for ``regex_engine``.

This module pins down the *published* matching semantics of the current
executor on short patterns and short texts, so that a second
execution path can later be checked against a stable regression baseline.

Design notes
------------
* Patterns and texts are generated as bounded, deterministic combinations
  of the supported grammar: plain and Unicode literals, ``.`` (which
  matches newlines here), positive/negated character classes, escapes,
  concatenation, empty and nested groups, and the ``?`` ``*`` ``+``
  ``{m}`` ``{m,}`` ``{m,n}`` quantifiers on atoms and on whole groups,
  each in all three modes (greedy, lazy ``?``, possessive ``+``).
  Generation depth, quantifier upper bounds and text length all have
  small explicit caps, so the case set has the same size, order and
  results on every run.  Alternation, anchors, named groups and
  backreferences are never generated: the engine rejects them.
* Expected results come from an *independent* oracle written in this
  file: its own parser plus a generator-based enumerator that walks the
  feasible paths in the documented priority order of each mode (greedy:
  longest first, backtrack by shortening; lazy: fewest repetitions
  first, expanding one at a time; possessive: the greedy maximum,
  committed) and takes the first complete match.  It never
  imports or calls any private parsing/matching helper of
  ``regex_engine``.  The oracle models the documented capture rules:
  groups that never participate stay ``None``; a group that participates
  but matches the empty string keeps its ``(pos, pos)`` span; a repeated
  group keeps the capture of its last participation (a nested group that
  skips a later repetition keeps its most recent earlier capture); and
  captures made by branches that backtracking abandons are rolled back.
* Every (pattern, text) case is observed through the public entry points
  only: ``compile(pattern).fullmatch(text)``, ``fullmatch(pattern, text)``
  and a repeated call on the same ``Pattern`` object.  All three must
  agree with the oracle: ``None`` on failure, and on success the same
  ``group(0)``, ``groups()`` and the same ``start``/``end``/``span`` for
  every group.
* Failure messages quote the pattern and text with ``ascii()`` so that
  newlines and non-ASCII characters stay distinguishable, and name the
  entry point, the expected observation and the actual one.
"""

from __future__ import annotations

import itertools
import unittest

from regex_engine import compile, fullmatch

# ---------------------------------------------------------------------------
# Independent semantic oracle
#
# The oracle has two parts: a parser for exactly the documented grammar
# (written from the module docstring, sharing no code with the engine)
# and a lazy enumerator.  ``_enum_seq``/``_enum_group*`` yield
# ``(end_position, captures)`` pairs in the priority order documented
# for the piece's mode; captures
# are threaded as immutable tuples, so a branch that is abandoned simply
# stops yielding -- its captures can never leak into a sibling branch,
# which is exactly the rollback rule the engine documents.
# ---------------------------------------------------------------------------


class _OracleParser:
    """Recursive-descent parser for the supported pattern subset.

    Produces nested ``pieces`` tuples: each piece is ``(atom, lo, hi,
    mode)`` with ``hi is None`` meaning "unbounded" and ``mode`` one of
    ``"greedy"``, ``"lazy"`` (quantifier followed by ``?``) or
    ``"possessive"`` (quantifier followed by ``+``).  An atom is one of
    ``("lit", ch)``, ``("any",)``,
    ``("cls", negated, chars, ranges)`` or
    ``("grp", group_number, sub_pieces)``.  Group numbers are 1-based in
    left-parenthesis order, as documented.
    """

    def __init__(self, pattern: str) -> None:
        self._p = pattern
        self._i = 0
        self.ngroups = 0

    def parse(self):
        pieces = self._sequence(top=True)
        if self._i != len(self._p):
            raise AssertionError(
                f"oracle parser stuck at {self._i} in {self._p!r}"
            )
        return tuple(pieces), self.ngroups

    def _sequence(self, top: bool):
        pieces = []
        while self._i < len(self._p):
            if self._p[self._i] == ")":
                if top:
                    raise AssertionError(
                        f"unbalanced ')' in generated pattern {self._p!r}"
                    )
                break
            pieces.append(self._piece())
        return pieces

    def _piece(self):
        atom = self._atom()
        lo, hi = 1, 1
        quantified = True
        if self._i < len(self._p):
            c = self._p[self._i]
            if c == "?":
                lo, hi = 0, 1
                self._i += 1
            elif c == "*":
                lo, hi = 0, None
                self._i += 1
            elif c == "+":
                lo, hi = 1, None
                self._i += 1
            elif c == "{":
                j = self._p.index("}", self._i)
                body = self._p[self._i + 1 : j]
                if "," in body:
                    low, _, high = body.partition(",")
                    lo = int(low)
                    hi = int(high) if high else None
                else:
                    lo = hi = int(body)
                self._i = j + 1
            else:
                quantified = False
        else:
            quantified = False
        mode = "greedy"
        if quantified and self._i < len(self._p):
            # A "?" or "+" directly after a complete quantifier is its
            # lazy/possessive modifier, never a new atom.
            c = self._p[self._i]
            if c == "?":
                mode = "lazy"
                self._i += 1
            elif c == "+":
                mode = "possessive"
                self._i += 1
        return (atom, lo, hi, mode)

    def _atom(self):
        c = self._p[self._i]
        if c == "(":
            self.ngroups += 1
            gid = self.ngroups
            self._i += 1
            sub = self._sequence(top=False)
            if self._i >= len(self._p) or self._p[self._i] != ")":
                raise AssertionError(
                    f"unbalanced '(' in generated pattern {self._p!r}"
                )
            self._i += 1
            return ("grp", gid, tuple(sub))
        if c == "\\":
            literal = self._p[self._i + 1]
            self._i += 2
            return ("lit", literal)
        if c == ".":
            self._i += 1
            return ("any",)
        if c == "[":
            return self._class()
        # Bare "]" and "}" are ordinary literals, as documented.
        self._i += 1
        return ("lit", c)

    def _class(self):
        self._i += 1  # consume "["
        negated = False
        if self._p[self._i] == "^":
            negated = True
            self._i += 1
        chars = set()
        ranges = []
        while True:
            c = self._p[self._i]
            if c == "]":
                self._i += 1
                return ("cls", negated, frozenset(chars), tuple(ranges))
            if c == "\\":
                lo = self._p[self._i + 1]
                self._i += 2
            else:
                lo = c
                self._i += 1
            # A "-" only starts a range when another member follows it.
            if (
                self._i + 1 < len(self._p)
                and self._p[self._i] == "-"
                and self._p[self._i + 1] != "]"
            ):
                self._i += 1
                c2 = self._p[self._i]
                if c2 == "\\":
                    hi = self._p[self._i + 1]
                    self._i += 2
                else:
                    hi = c2
                    self._i += 1
                ranges.append((lo, hi))
            else:
                chars.add(lo)


def _atom_matches(atom, ch: str) -> bool:
    kind = atom[0]
    if kind == "lit":
        return ch == atom[1]
    if kind == "any":
        return True  # "." matches every character, newline included
    _, negated, chars, ranges = atom
    found = ch in chars or any(lo <= ch <= hi for lo, hi in ranges)
    return (not found) if negated else found


def _enum_seq(pieces, index, pos, caps, text):
    """Yield ``(end, caps)`` for ``pieces[index:]`` starting at ``pos``.

    ``caps`` is a tuple with one slot per group (slot 0 unused); each
    slot is ``None`` or a ``(start, end)`` pair.  Ends are produced in
    the piece's own priority order: greedy longest-first, lazy
    shortest-first, possessive committed to the greedy maximum.
    """
    if index == len(pieces):
        yield pos, caps
        return
    atom, lo, hi, mode = pieces[index]
    if atom[0] == "grp":
        for mid, caps1 in _enum_group(atom, lo, hi, mode, pos, 0, caps, text):
            yield from _enum_seq(pieces, index + 1, mid, caps1, text)
        return
    # Single-character atom: greedy takes the longest run first and
    # offers every shorter count down to the lower bound; lazy offers
    # counts growing from the lower bound; possessive offers only the
    # longest run.
    limit = len(text) - pos if hi is None else min(hi, len(text) - pos)
    count = 0
    while count < limit and _atom_matches(atom, text[pos + count]):
        count += 1
    if mode == "lazy":
        takes = range(lo, count + 1)
    elif mode == "possessive":
        takes = (count,) if count >= lo else ()
    else:
        takes = range(count, lo - 1, -1)
    for take in takes:
        yield from _enum_seq(pieces, index + 1, pos + take, caps, text)


def _enum_group(atom, lo, hi, mode, base, count, caps, text):
    """Dispatch to the group enumerator for the piece's mode."""
    if mode == "lazy":
        yield from _enum_group_lazy(atom, lo, hi, base, count, caps, text)
        return
    if mode == "possessive":
        # Take the greedy maximum and commit: only the first offer of
        # the greedy enumeration is ever presented.
        for end, caps1 in _enum_group_greedy(
                atom, lo, hi, base, count, caps, text):
            yield end, caps1
            return
        return
    yield from _enum_group_greedy(atom, lo, hi, base, count, caps, text)


def _enum_group_greedy(atom, lo, hi, base, count, caps, text):
    """Yield ``(end, caps)`` for one more stretch of ``(sub){lo,hi}``.

    ``count`` repetitions are already committed; the next one starts at
    ``base``.  Greedy order: for each sub-sequence end (longest first),
    first try to extend with another repetition, then settle on the
    current end.  A sub-match that consumes nothing is offered once and
    stops expansion, so empty-able repeats terminate; the lower bound is
    waived for that empty match, as documented.
    """
    gid, sub = atom[1], atom[2]
    if hi is not None and count >= hi:
        yield base, caps
        return
    for end, caps1 in _enum_seq(sub, 0, base, caps, text):
        caps2 = caps1[:gid] + ((base, end),) + caps1[gid + 1 :]
        if end == base:
            yield end, caps2
            break
        if hi is None or count + 1 < hi:
            yield from _enum_group_greedy(
                atom, lo, hi, end, count + 1, caps2, text)
        if count + 1 >= lo:
            yield end, caps2
    # No (further) repetition: stop here if the minimum is already met.
    # ``caps`` is the entry state, so a group that never participates in
    # this stretch keeps whatever it captured before (or stays None).
    if count >= lo:
        yield base, caps


def _enum_group_lazy(atom, lo, hi, base, count, caps, text):
    """Yield ``(end, caps)`` for a lazy stretch of ``(sub){lo,hi}``.

    Lazy order: the fewest repetitions first -- settle at the committed
    count (once the minimum is met) before trying one more repetition.
    Each repetition's sub-sequence is enumerated in its own priority
    order.  A sub-match that consumes nothing is offered once and stops
    expansion, so empty-able repeats terminate; the lower bound is
    waived for that empty match, as documented.
    """
    gid, sub = atom[1], atom[2]
    if count >= lo:
        yield base, caps
    if hi is not None and count >= hi:
        return
    for end, caps1 in _enum_seq(sub, 0, base, caps, text):
        caps2 = caps1[:gid] + ((base, end),) + caps1[gid + 1 :]
        if end == base:
            yield end, caps2
            return
        yield from _enum_group_lazy(atom, lo, hi, end, count + 1, caps2, text)


def _oracle_caps(pattern, text):
    """First complete match's capture tuple, or None if there is none."""
    pieces, ngroups = _OracleParser(pattern).parse()
    caps0 = (None,) * (ngroups + 1)
    for end, caps in _enum_seq(pieces, 0, 0, caps0, text):
        if end == len(text):
            return caps
    return None


# ---------------------------------------------------------------------------
# Observations: everything the public API can report about a result
# ---------------------------------------------------------------------------


def _observe(match):
    """Public-API observation of a Match: None, or a comparable tuple."""
    if match is None:
        return None
    count = len(match.groups())
    indexes = range(count + 1)
    return (
        match.group(0),
        match.groups(),
        tuple(match.span(i) for i in indexes),
        tuple(match.start(i) for i in indexes),
        tuple(match.end(i) for i in indexes),
    )


def _expected(pattern, text):
    """The same observation, computed by the independent oracle."""
    caps = _oracle_caps(pattern, text)
    if caps is None:
        return None
    spans = ((0, len(text)),) + tuple(
        (-1, -1) if span is None else span for span in caps[1:]
    )
    groups = tuple(
        None if span is None else text[span[0] : span[1]] for span in caps[1:]
    )
    starts = tuple(span[0] for span in spans)
    ends = tuple(span[1] for span in spans)
    return (text, groups, spans, starts, ends)


def _ascii(value):
    return "None" if value is None else ascii(value)


def _format_obs(obs):
    """Render an observation keeping newlines and non-ASCII visible."""
    if obs is None:
        return "None"
    group0, groups, spans, starts, ends = obs
    inner = ", ".join(_ascii(g) for g in groups)
    if len(groups) == 1:
        inner += ","
    return (
        f"group(0)={ascii(group0)} groups=({inner}) "
        f"starts={starts!r} ends={ends!r} spans={spans!r}"
    )


# ---------------------------------------------------------------------------
# Deterministic bounded case generation
#
# Bounds: group nesting depth <= 3 (only in the hand-picked extras),
# quantifier upper bounds <= 2 (plus {2,} for an open upper bound), text
# length <= 3, concatenations of at most 2 pieces (plus short prefixes
# and suffixes around groups).
# ---------------------------------------------------------------------------

ATOMS = ("a", "b", "é", "🙂", ".", "\\.", "\\*", "[ab]", "[^a]", "[a-c]", "[^a-cé]")
QUANTIFIERS = ("", "?", "*", "+", "{2}", "{0,2}", "{1,2}", "{0}", "{2,}",
               "??", "*?", "+?", "{1,2}?", "{2,}?",
               "?+", "*+", "++", "{1,2}+", "{2,}+")
CONCAT_ATOMS = ("a", ".", "[ab]", "[^a]")
CONCAT_QUANTIFIERS = ("", "?", "*", "+", "*?", "*+")
GROUP_BODIES = (
    "", "a", "b", ".", "ab", "a.", "a?", "a*", "ba", "[ab]",
    "é", "a(b)", "(a)b", "(a)(b)", "(a?)", "()",
)
CONTEXT_PREFIXES = ("", "a", "a?")
CONTEXT_BODIES = ("", "a", "ab", "a?", "a*")
CONTEXT_QUANTIFIERS = ("", "?", "*", "+", "*?", "*+")
CONTEXT_SUFFIXES = ("", "b", "b?", ".*")

# Hand-picked shapes the cross products above do not reach: deeper
# nesting, zero-width repeats, competing greedy splits, brace bounds on
# concatenations, Unicode groups and mode modifiers on groups.
EXTRA_PATTERNS = (
    "((a))", "((a)*)", "((a*)*)", "((a*)+)", "(a(b))", "(a(b)?)",
    "(a(b)?)*", "((a+)b)+", "((a)(b))", "(a(b)c)?", "((()))",
    "(()*)", "(()+)", "(a*)+", "(a?)*", "(a*){2}", "(){2}", "()*a",
    "(a*)(a*)", "(a?)(b?)", "(x)?y", "(..)(..)", "(a*).*",
    "([ab]+)(x)?", "(a(b)?)*.", "(a*)(a*)?", "(())*",
    "a{2}b{2}", "a{0,2}b", "a{1,2}b{0,2}", ".{2}a", "(ab){2,}",
    "(a){0}", "(a?){0,2}", "((a)b(c))", "(é)(🙂)?",
    "(a*?)(a*)", "(a*+)(a*)", "(a+?)(a+)", "(a??)(a?)", "(a?+)(a?)",
    "(a){1,3}?a", "(a){1,3}a", "(a){1,3}+a", "a*?a", "a*+a",
    "(a*)*?", "(a*)*+", "(a*)+?", "(a*)++", "()*?", "()*+",
    "(a*?)*", "(a*+)*", "((a*)*?)", "(a(b)?)*?", "(a(b)?)*+",
    "(a?){2,}?", "(a?){2,}+", "(ab){1,}?", "(ab){1,}+ab",
)

TEXT_ALPHABET = ("a", "b", "é", "\n")
EXTRA_TEXTS = (
    "aaa", "aab", "aba", "abb", "baa", "ab\n", "a\nb",
    "ééé", "a.b", "...", "aa\n", "🙂🙂", "é.a", "b\na",
)


def _dedupe(items):
    seen = set()
    unique = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return tuple(unique)


def _generate_patterns():
    patterns = []
    # 1. Every atom under every quantifier.
    for atom in ATOMS:
        for quant in QUANTIFIERS:
            patterns.append(atom + quant)
    # 2. Two-piece concatenations (backtracking must share characters).
    pieces = [a + q for a in CONCAT_ATOMS for q in CONCAT_QUANTIFIERS]
    for first in pieces:
        for second in pieces:
            patterns.append(first + second)
    # 3. Groups -- empty, nested, quantified inside -- under every
    #    quantifier.
    for body in GROUP_BODIES:
        for quant in QUANTIFIERS:
            patterns.append("(" + body + ")" + quant)
    # 4. Groups inside concatenations, so captures interact with
    #    surrounding backtracking.
    for prefix in CONTEXT_PREFIXES:
        for body in CONTEXT_BODIES:
            for quant in CONTEXT_QUANTIFIERS:
                for suffix in CONTEXT_SUFFIXES:
                    patterns.append(prefix + "(" + body + ")" + quant + suffix)
    # 5. Hand-picked boundary shapes.
    patterns.extend(EXTRA_PATTERNS)
    return _dedupe(patterns)


def _generate_texts():
    texts = [""]
    for length in (1, 2):
        for chars in itertools.product(TEXT_ALPHABET, repeat=length):
            texts.append("".join(chars))
    texts.extend(EXTRA_TEXTS)
    return _dedupe(texts)


PATTERNS = _generate_patterns()
TEXTS = _generate_texts()
CASES = tuple((pattern, text) for pattern in PATTERNS for text in TEXTS)
# Patterns whose capture state could leak across calls; used by the
# reuse-after-failure test.
GROUP_PATTERNS = tuple(p for p in PATTERNS if "(" in p)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class OracleSanityTest(unittest.TestCase):
    """Anchor the oracle to the documented semantics before trusting it.

    Every expectation below is taken from the behaviour the existing
    system tests already pin down, so a broken oracle fails here rather
    than silently redefining the exhaustive comparison.
    """

    # (pattern, text, expected per-group (text, span) or None, or None
    # when no full match exists)
    KNOWN = [
        ("(a*)+", "aaa", (("", (3, 3)),)),
        ("(a?)*", "a", (("", (1, 1)),)),
        ("(a(b)?)*", "aba", (("a", (2, 3)), ("b", (1, 2)))),
        ("(a(b)?)*", "aaa", (("a", (2, 3)), None)),
        ("(a*)(a*)", "aa", (("aa", (0, 2)), ("", (2, 2)))),
        ("(ab)*", "", (None,)),
        ("()", "", (("", (0, 0)),)),
        ("(a(b)c)?a*", "aa", (None, None)),
        ("((a)(b))", "ab", (("ab", (0, 2)), ("a", (0, 1)), ("b", (1, 2)))),
        ("(a*){2}", "a", (("", (1, 1)),)),
        ("(){2}", "", (("", (0, 0)),)),
        ("(a)", "b", None),
        ("a*", "aab", None),
        ("a.c", "a\nc", ()),
        # Lazy and possessive modes, anchored to the documented rules.
        ("(a*?)(a*)", "aaa", (("", (0, 0)), ("aaa", (0, 3)))),
        ("(a*+)(a*)", "aaa", (("aaa", (0, 3)), ("", (3, 3)))),
        ("a*?a", "aaa", ()),
        ("a*+a", "aaa", None),
        ("a*+a", "a", None),
        ("(a){1,3}?a", "aaa", (("a", (1, 2)),)),
        ("(a){1,3}a", "aaa", (("a", (1, 2)),)),
        ("(a*)+?", "aaa", (("aaa", (0, 3)),)),
        ("(a*)++a", "aaa", None),
        ("(a*)*?", "aaa", (("aaa", (0, 3)),)),
        ("(a*)*+", "aaa", (("", (3, 3)),)),
    ]

    def test_oracle_reproduces_known_semantics(self):
        for pattern, text, expected_groups in self.KNOWN:
            with self.subTest(pattern=ascii(pattern), text=ascii(text)):
                caps = _oracle_caps(pattern, text)
                if expected_groups is None:
                    self.assertIsNone(caps)
                    continue
                self.assertIsNotNone(caps)
                actual = tuple(
                    None if span is None else (text[span[0] : span[1]], span)
                    for span in caps[1:]
                )
                self.assertEqual(actual, expected_groups)


class GenerationDeterminismTest(unittest.TestCase):
    """The case set is bounded, duplicate-free and stable across runs."""

    def test_counts_are_fixed(self):
        self.assertEqual(len(PATTERNS), 1459)
        self.assertEqual(len(TEXTS), 35)
        self.assertEqual(len(CASES), 1459 * 35)

    def test_no_duplicates(self):
        self.assertEqual(len(set(PATTERNS)), len(PATTERNS))
        self.assertEqual(len(set(TEXTS)), len(TEXTS))
        self.assertEqual(len(set(CASES)), len(CASES))

    def test_required_edge_cases_are_present(self):
        # Empty text and texts that no pattern can fully consume.
        self.assertIn("", TEXTS)
        self.assertIn("aab", TEXTS)
        # Zero-width group repeats and competing greedy splits.
        for pattern in ("()*", "(){2}", "(a*)+", "(a?)*", "(a*)(a*)"):
            self.assertIn(pattern, PATTERNS)
        # Lazy and possessive modes, on atoms and on empty-able groups.
        for pattern in ("a*?", "a*+", "(a*)*?", "(a*)*+", "(a*?)(a*)"):
            self.assertIn(pattern, PATTERNS)
        # Newline and non-ASCII coverage in both patterns and texts.
        self.assertIn("\n", TEXTS)
        self.assertIn("a\nb", TEXTS)
        self.assertIn("é", ATOMS)
        self.assertIn("🙂", ATOMS)

    def test_every_generated_pattern_compiles(self):
        for pattern in PATTERNS:
            with self.subTest(pattern=ascii(pattern)):
                self.assertEqual(compile(pattern).pattern, pattern)


class ExhaustiveConsistencyTest(unittest.TestCase):
    """Oracle vs. all three public entry points, for every case."""

    def test_all_entry_points_agree_with_the_oracle(self):
        for pattern, text in CASES:
            expected = _expected(pattern, text)
            compiled = compile(pattern)
            attempts = (
                ("compile(pattern).fullmatch(text)", compiled.fullmatch(text)),
                ("fullmatch(pattern, text)", fullmatch(pattern, text)),
                ("repeated compiled.fullmatch(text)", compiled.fullmatch(text)),
            )
            for entry, match in attempts:
                actual = _observe(match)
                if actual != expected:
                    self.fail(
                        "matching semantics mismatch\n"
                        f"  pattern:  {ascii(pattern)}\n"
                        f"  text:     {ascii(text)}\n"
                        f"  entry:    {entry}\n"
                        f"  expected: {_format_obs(expected)}\n"
                        f"  actual:   {_format_obs(actual)}"
                    )


class ReuseAfterFailureTest(unittest.TestCase):
    """A failed match leaves a compiled Pattern's later results intact."""

    def test_failure_then_full_sweep_on_one_pattern_object(self):
        for pattern in GROUP_PATTERNS:
            failing = next(
                (t for t in TEXTS if _expected(pattern, t) is None), None
            )
            if failing is None:
                continue  # matches every generated text; nothing to probe
            compiled = compile(pattern)
            probe = compiled.fullmatch(failing)
            if probe is not None:
                self.fail(
                    "reuse probe unexpectedly matched\n"
                    f"  pattern:  {ascii(pattern)}\n"
                    f"  text:     {ascii(failing)}\n"
                    f"  expected: None\n"
                    f"  actual:   {_format_obs(_observe(probe))}"
                )
            for text in TEXTS:
                expected = _expected(pattern, text)
                actual = _observe(compiled.fullmatch(text))
                if actual != expected:
                    self.fail(
                        "result changed after a failed match on the same "
                        "Pattern\n"
                        f"  pattern:  {ascii(pattern)}\n"
                        f"  text:     {ascii(text)}\n"
                        f"  entry:    compiled.fullmatch(text) after "
                        f"compiled.fullmatch({ascii(failing)}) -> None\n"
                        f"  expected: {_format_obs(expected)}\n"
                        f"  actual:   {_format_obs(actual)}"
                    )


if __name__ == "__main__":
    unittest.main()
