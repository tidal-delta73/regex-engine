"""Exhaustive greedy-semantics consistency tests for ``regex_engine``.

This module pins down the *published* behaviour of the current greedy
executor on short patterns and short texts, so that a second execution
path can later be added against a stable regression baseline.

How it works
------------
* Patterns are generated from the currently supported syntax only
  (literals incl. Unicode, ``.`` matching newlines, positive/negated
  character classes, escapes, concatenation, empty and nested groups,
  and ``?`` ``*`` ``+`` ``{m}`` ``{m,}`` ``{m,n}`` quantifiers on atoms
  and on whole groups).  Alternation, anchors, named groups and
  backreferences are never generated -- they stay rejected.
* Expected results come from an *independent* semantic oracle below:
  it has its own parser and its own path enumerator and never touches
  ``regex_engine`` internals.  It enumerates feasible paths in the
  documented greedy priority order and takes the first complete match,
  modelling non-participating groups, groups that participate but match
  the empty string, last-participation-wins inside repeats, and capture
  rollback of abandoned branches (captures are immutable values here,
  so abandoning a branch restores them by construction).
* Every (pattern, text) case is observed through all three public
  entry points -- ``fullmatch(pattern, text)``,
  ``compile(pattern).fullmatch(text)`` and repeated calls on one shared
  ``Pattern`` (including a second full pass after failures) -- and each
  observation must equal the oracle's: ``None`` on failure, else
  ``group(0)``, ``groups()`` and every group's ``start``/``end``/``span``.

Everything is bounded (see the ``MAX_*`` constants) and generated in a
fixed order, so the case set is identical on every run.  Failure
messages quote the pattern, text and entry point with ``ascii()``, so
newlines and non-ASCII characters stay recognisable.
"""

from __future__ import annotations

import unittest
from itertools import product

from regex_engine import compile, fullmatch

_INF = float("inf")

# ---------------------------------------------------------------------------
# Bounded corpus dimensions.  These caps keep the generated case set small,
# deterministic and identical on every run.
# ---------------------------------------------------------------------------

MAX_SEQUENCE_PIECES = 3   # top-level concatenation length
MAX_GROUP_PIECES = 2      # pieces inside a generated group body
MAX_NESTING_DEPTH = 2     # groups inside groups
MAX_QUANTIFIER_BOUND = 2  # largest explicit {m,n} bound
MAX_TEXT_LENGTH = 3       # longest generated text

# ---------------------------------------------------------------------------
# Independent semantic oracle
#
# This parser and enumerator are written from the documented semantics only.
# They deliberately share no code with regex_engine's private machinery.
# ---------------------------------------------------------------------------


class _OracleParser:
    """Parses the supported subset into (atom, lo, hi) piece lists.

    Atom forms:
      ("lit", char)                    literal (incl. escaped chars)
      ("any",)                         "."
      ("cls", negated, chars, ranges)  character class
      ("grp", gid, sub_pieces)         numbered capturing group
    """

    def __init__(self, pattern):
        self._pat = pattern
        self._i = 0
        self.ngroups = 0

    def parse(self):
        pieces = self._parse_seq(in_group=False)
        if self._i != len(self._pat):
            raise ValueError("unbalanced parenthesis")
        return pieces, self.ngroups

    def _parse_seq(self, in_group):
        pieces = []
        while self._i < len(self._pat):
            c = self._pat[self._i]
            if c == ")":
                if not in_group:
                    raise ValueError("unbalanced parenthesis")
                break
            if c == "(":
                self._i += 1
                self.ngroups += 1
                gid = self.ngroups
                sub = self._parse_seq(in_group=True)
                if self._i >= len(self._pat):
                    raise ValueError("unbalanced parenthesis")
                self._i += 1  # consume ")"
                atom = ("grp", gid, sub)
            else:
                atom = self._parse_atom()
            lo, hi = self._parse_quantifier()
            pieces.append((atom, lo, hi))
        return pieces

    def _parse_atom(self):
        c = self._pat[self._i]
        if c == "\\":
            self._i += 2
            return ("lit", self._pat[self._i - 1])
        if c == ".":
            self._i += 1
            return ("any",)
        if c == "[":
            return self._parse_class()
        self._i += 1
        return ("lit", c)

    def _class_char(self):
        c = self._pat[self._i]
        if c == "\\":
            self._i += 2
            return self._pat[self._i - 1]
        self._i += 1
        return c

    def _parse_class(self):
        self._i += 1  # consume "["
        negated = False
        if self._pat[self._i] == "^":
            negated = True
            self._i += 1
        chars = set()
        ranges = []
        while True:
            if self._pat[self._i] == "]":
                self._i += 1
                break
            lo = self._class_char()
            if (
                self._i + 1 < len(self._pat)
                and self._pat[self._i] == "-"
                and self._pat[self._i + 1] != "]"
            ):
                self._i += 1
                ranges.append((lo, self._class_char()))
            else:
                chars.add(lo)
        return ("cls", negated, frozenset(chars), tuple(ranges))

    def _parse_quantifier(self):
        if self._i >= len(self._pat):
            return 1, 1
        c = self._pat[self._i]
        if c == "?":
            self._i += 1
            return 0, 1
        if c == "*":
            self._i += 1
            return 0, _INF
        if c == "+":
            self._i += 1
            return 1, _INF
        if c == "{":
            j = self._pat.index("}", self._i)
            body = self._pat[self._i + 1 : j]
            self._i = j + 1
            if "," in body:
                lo_s, hi_s = body.split(",")
                return int(lo_s), (_INF if hi_s == "" else int(hi_s))
            return int(body), int(body)
        return 1, 1


def _atom_matches(atom, ch):
    kind = atom[0]
    if kind == "lit":
        return ch == atom[1]
    if kind == "any":
        return True
    _, negated, chars, ranges = atom
    found = ch in chars or any(lo <= ch <= hi for lo, hi in ranges)
    return (not found) if negated else found


def _enum_seq(pieces, idx, pos, caps, text):
    """Yield (end, caps) for pieces[idx:] at pos, greediest first.

    ``caps`` is an immutable tuple; index 0 is unused, caps[g] is the
    (start, end) span of group g or None if it never participated.
    """
    if idx == len(pieces):
        yield pos, caps
        return
    atom, lo, hi = pieces[idx]
    if atom[0] == "grp":
        for end, caps2 in _enum_group(atom, lo, hi, pos, 0, caps, text):
            yield from _enum_seq(pieces, idx + 1, end, caps2, text)
        return
    count = 0
    while (
        count < hi
        and pos + count < len(text)
        and _atom_matches(atom, text[pos + count])
    ):
        count += 1
    for take in range(count, lo - 1, -1):
        yield from _enum_seq(pieces, idx + 1, pos + take, caps, text)


def _enum_group(atom, lo, hi, base, count, entry, text):
    """Yield (end, caps) for one quantified group piece, greediest first.

    ``entry`` is the capture state on entry to this repetition enumerator;
    ``count`` repetitions are already recorded in ``entry``.
    """
    _, gid, sub = atom
    if count >= hi:
        yield base, entry
        return
    for end, subcaps in _enum_seq(sub, 0, base, entry, text):
        repcaps = subcaps[:gid] + ((base, end),) + subcaps[gid + 1 :]
        if end == base:
            # Empty repetition: offer it once with the minimum waived,
            # then stop expanding instead of looping forever.
            yield base, repcaps
            if count >= lo:
                yield base, entry
            return
        if count + 1 < hi:
            # Greedy: exhaust one-more-repetition before settling here.
            yield from _enum_group(atom, lo, hi, end, count + 1, repcaps, text)
            if count + 1 >= lo:
                yield end, repcaps
        else:
            yield end, repcaps
        # Rejected: backtrack to a shorter sub-match, then stop here.
    if count >= lo:
        yield base, entry


def _oracle_spans(pieces, ngroups, text):
    """First complete-match capture state, or None (fullmatch semantics)."""
    init = (None,) * (ngroups + 1)
    for end, caps in _enum_seq(pieces, 0, 0, init, text):
        if end == len(text):
            return caps
    return None


# ---------------------------------------------------------------------------
# Deterministic corpus generation
# ---------------------------------------------------------------------------

# Single-character atoms: plain and Unicode literals, dot, positive and
# negated classes, a range class and an escaped metacharacter.
_CONCAT_ATOMS = ("a", "b", ".", "[ab]", "[^a]", "é")
_CONCAT3_ATOMS = ("a", "b", ".", "[ab]")
_QUANT_ATOMS = ("a", "b", ".", "é", "🙂", "[ab]", "[^a]", "[a-c]", r"\*")

# Every bound stays <= MAX_QUANTIFIER_BOUND; "" means "no quantifier".
_QUANTIFIERS = ("", "?", "*", "+", "{0}", "{2}", "{0,2}", "{1,2}")

# Adjacent quantified atoms, forcing the greedy splits to be shared.
_PAIR_LEFT = tuple(
    atom + q for atom in ("a", ".", "[ab]") for q in ("?", "*", "+", "{1,2}")
)
_PAIR_RIGHT = tuple(atom + q for atom in ("a", "[ab]") for q in ("", "?", "*", "+"))

# Group bodies: empty, literals, dot, classes, concatenations, quantified
# atoms and nested (possibly empty) groups, within the depth cap.
_GROUP_BODIES = (
    "", "a", "b", ".", "é",
    "ab", "a.", "[ab]", "[^a]",
    "a*", "a+", "b?", "a{1,2}", "[ab]*",
    "()", "(a)", "(a*)", "(a)(b)", "a(b)",
)
_GROUP_QUANTIFIERS = _QUANTIFIERS

# Bodies and quantifiers reused inside concatenation wrappers.
_WRAPPED_BODIES = ("", "a", "ab", "a*", "(a)")
_WRAPPED_QUANTIFIERS = ("", "?", "*", "+")

# Hand-picked structural edges: escapes, class corners, nesting, empty and
# zero-width repeats, last-participation and rollback shapes.
_EXTRA_PATTERNS = (
    "",
    r"\.", r"\*", r"\\", r"\]", r"\}", r"a\.b", r"\é", r"\🙂",
    r"[\]]", r"[\\]", r"[a\-z]", "[a-]", "[a^]", "[^a-c]", "[α-ω]",
    "...", ".*", ".+", "a.*", ".*a", ".*.*", "[^a]*", "[a-c]+", "[^a][^a]",
    "a{2,}", "b{0,}", "(a){1,}", "(ab){2,}",
    "()", "(())", "((()))", "()a", "a()", "()()",
    "(a)", "(a)(b)", "(a*)(a*)", "(a?)(b?)", "(.)(.*)", "(..)(..)",
    "((a)(b))", "((a+)b)+", "(a(b)?)*", "(a(b)c)?a*", "(a*)a",
    "(ab){2}", "(ab){1,2}", "(a*)+", "(a?)*", "(a*){2}", "((a)*)",
)


def _dedup(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def _generate_patterns():
    patterns = []

    # 1. Plain concatenations of atoms (no quantifiers, no groups).
    for length in (1, 2):
        for combo in product(_CONCAT_ATOMS, repeat=length):
            patterns.append("".join(combo))
    for combo in product(_CONCAT3_ATOMS, repeat=MAX_SEQUENCE_PIECES):
        patterns.append("".join(combo))

    # 2. Every quantifier on every atom.
    for atom in _QUANT_ATOMS:
        for quant in _QUANTIFIERS[1:]:
            patterns.append(atom + quant)

    # 3. Adjacent quantified atoms competing for the same characters.
    for left in _PAIR_LEFT:
        for right in _PAIR_RIGHT:
            patterns.append(left + right)

    # 4. Every group body with every quantifier (incl. zero-width repeats).
    for body in _GROUP_BODIES:
        for quant in _GROUP_QUANTIFIERS:
            patterns.append("(" + body + ")" + quant)

    # 5. Groups embedded in concatenations, forcing backtracking across
    #    the group boundary.
    for body in _WRAPPED_BODIES:
        for quant in _WRAPPED_QUANTIFIERS:
            for template in ("a({}){}", "({}){}b", "a({}){}b"):
                patterns.append(template.format(body, quant))

    # 6. Hand-picked structural edges.
    patterns.extend(_EXTRA_PATTERNS)

    return _dedup(patterns)


def _generate_texts():
    texts = [
        "",                       # empty text
        "\n", "\n\n", "a\n", "\nb", "a\nb",   # newlines (dot must match)
        "*", "**", "a*", ".", "..", "]",
        "é", "éé", "ééé", "🙂", "🙂🙂", "é🙂", "あ",
    ]
    for length in (1, 2, MAX_TEXT_LENGTH):
        for combo in product("ab", repeat=length):
            texts.append("".join(combo))
    for combo in product("aé", repeat=2):
        texts.append("".join(combo))
    return _dedup(texts)


# ---------------------------------------------------------------------------
# Observation and reporting helpers
# ---------------------------------------------------------------------------


def _expected_observation(caps, text, ngroups):
    """Normalise the oracle result exactly like the public Match API."""
    if caps is None:
        return None
    raw = ((0, len(text)),) + tuple(caps[1 : ngroups + 1])
    return (
        text,
        tuple(None if s is None else text[s[0] : s[1]] for s in raw[1:]),
        tuple((-1, -1) if s is None else s for s in raw),
        tuple(-1 if s is None else s[0] for s in raw),
        tuple(-1 if s is None else s[1] for s in raw),
    )


def _observed(match, ngroups):
    """Read everything the public API promises about one match result."""
    if match is None:
        return None
    return (
        match.group(0),
        match.groups(),
        tuple(match.span(i) for i in range(ngroups + 1)),
        tuple(match.start(i) for i in range(ngroups + 1)),
        tuple(match.end(i) for i in range(ngroups + 1)),
    )


def _report(pattern, text, entry, expected, actual):
    # ascii() keeps newlines and non-ASCII characters recognisable.
    return (
        f"pattern={ascii(pattern)} text={ascii(text)} entry={entry}\n"
        f"  expected: {ascii(expected)}\n"
        f"  actual:   {ascii(actual)}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class CorpusShapeTest(unittest.TestCase):
    """The generated corpus itself is frozen: same cases, same order."""

    def test_generation_is_deterministic(self):
        self.assertEqual(_generate_patterns(), _generate_patterns())
        self.assertEqual(_generate_texts(), _generate_texts())

    def test_corpus_size_is_frozen(self):
        self.assertEqual(len(_generate_patterns()), 513)
        self.assertEqual(len(_generate_texts()), 35)

    def test_corpus_covers_required_edges(self):
        patterns = _generate_patterns()
        texts = _generate_texts()
        # Empty text and empty pattern.
        self.assertIn("", texts)
        self.assertIn("", patterns)
        # Newline and non-ASCII texts.
        self.assertIn("\n", texts)
        self.assertIn("a\nb", texts)
        self.assertIn("é", texts)
        self.assertIn("🙂", texts)
        # Zero-width group repeats.
        for pattern in ("()*", "()+", "(){2}", "()?", "(a*)+", "(a?)*"):
            self.assertIn(pattern, patterns)
        # Multiple viable greedy splits.
        for pattern in ("(a*)(a*)", "a*a", ".*.*", "(a?)(b?)"):
            self.assertIn(pattern, patterns)
        # Quantifiers on whole groups, empty and nested groups.
        self.assertIn("(ab){1,2}", patterns)
        self.assertIn("((a)(b))", patterns)
        self.assertIn("()", patterns)


class GreedyOracleConsistencyTest(unittest.TestCase):
    """Every generated case agrees with the independent oracle via all
    three public entry points, including reuse of one Pattern object."""

    def test_all_cases_match_the_oracle(self):
        patterns = _generate_patterns()
        texts = _generate_texts()
        failures = []
        checked = 0
        for pattern in patterns:
            pieces, ngroups = _OracleParser(pattern).parse()
            expected_by_text = {
                text: _expected_observation(
                    _oracle_spans(pieces, ngroups, text), text, ngroups
                )
                for text in texts
            }
            compiled = compile(pattern)
            for entry, make in (
                ("fullmatch(pattern, text)",
                 lambda p, t: fullmatch(p, t)),
                ("compile(pattern).fullmatch(text)",
                 lambda p, t: compile(p).fullmatch(t)),
                ("shared_pattern.fullmatch(text) [pass 1]",
                 lambda p, t: compiled.fullmatch(t)),
            ):
                for text in texts:
                    expected = expected_by_text[text]
                    actual = _observed(make(pattern, text), ngroups)
                    checked += 1
                    if actual != expected:
                        failures.append(
                            _report(pattern, text, entry, expected, actual)
                        )
            # Second pass over every text with the same Pattern object:
            # results must be reproducible after intervening failures.
            for text in texts:
                expected = expected_by_text[text]
                actual = _observed(compiled.fullmatch(text), ngroups)
                checked += 1
                if actual != expected:
                    failures.append(
                        _report(
                            pattern, text,
                            "shared_pattern.fullmatch(text) [pass 2]",
                            expected, actual,
                        )
                    )
        self.assertGreater(checked, 0)
        if failures:
            shown = "\n".join(failures[:25])
            self.fail(
                f"{len(failures)} of {checked} observations diverged from "
                f"the oracle (showing {min(len(failures), 25)}):\n{shown}"
            )

    def test_oracle_exercises_key_semantic_corners(self):
        """The corpus really reaches the tricky published behaviours."""
        patterns = _generate_patterns()
        texts = _generate_texts()
        saw = {
            "match": False,
            "no_match": False,
            "absent_group": False,       # group never participated
            "empty_group": False,        # participated, matched ""
            "newline_via_dot": False,    # "." matched "\n"
            "failure_then_success": False,  # reuse after a failed text
        }
        for pattern in patterns:
            pieces, ngroups = _OracleParser(pattern).parse()
            outcomes = []
            for text in texts:
                caps = _oracle_spans(pieces, ngroups, text)
                outcomes.append(caps is not None)
                if caps is None:
                    saw["no_match"] = True
                    continue
                saw["match"] = True
                if any(caps[g] is None for g in range(1, ngroups + 1)):
                    saw["absent_group"] = True
                if any(
                    caps[g] is not None and caps[g][0] == caps[g][1]
                    for g in range(1, ngroups + 1)
                ):
                    saw["empty_group"] = True
                if "." in pattern and "\n" in text:
                    saw["newline_via_dot"] = True
            if not outcomes[0] and any(outcomes):
                saw["failure_then_success"] = True
        for corner, reached in sorted(saw.items()):
            with self.subTest(corner=corner):
                self.assertTrue(reached, f"corpus never exercises: {corner}")


if __name__ == "__main__":
    unittest.main()
