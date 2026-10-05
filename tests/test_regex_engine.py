"""System tests for the frozen public API of regex_engine.

Everything here goes through the documented surface only: ``compile``,
``fullmatch``, ``Pattern``, ``Match`` and ``RegexSyntaxError`` as exported
from the ``regex_engine`` package.  No private helpers (``_parse``, ``_run``,
``_pieces`` ...) are touched, and the standard-library ``re`` module is
deliberately not used as an oracle: this engine differs from it (dot matches
newlines, only full matches succeed, the accepted grammar is smaller).
"""

import subprocess
import sys
from pathlib import Path

import pytest

import regex_engine
from regex_engine import Match, Pattern, RegexSyntaxError, compile, fullmatch

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def assert_match(m, text):
    """Check every observable of a successful Match over ``text``."""
    assert isinstance(m, Match)
    assert m.group() == text
    assert m.group(0) == text
    assert m.start() == 0
    assert m.end() == len(text)
    assert m.span() == (0, len(text))
    assert m.string == text
    assert repr(m) == f"<Match {text!r}>"


def assert_matches(pattern, text):
    """Both public entry points must agree, and agree on a full Match."""
    m1 = fullmatch(pattern, text)
    m2 = compile(pattern).fullmatch(text)
    assert m1 is not None, f"{pattern!r} should match {text!r}"
    assert m2 is not None
    assert_match(m1, text)
    assert_match(m2, text)


def assert_no_match(pattern, text):
    assert fullmatch(pattern, text) is None
    assert compile(pattern).fullmatch(text) is None


# ---------------------------------------------------------------------------
# literals, empty pattern/text, Unicode
# ---------------------------------------------------------------------------


class TestLiterals:
    def test_empty_pattern_matches_empty_text(self):
        assert_matches("", "")

    def test_empty_pattern_rejects_nonempty_text(self):
        assert_no_match("", "a")
        assert_no_match("", "\n")

    def test_nonempty_pattern_rejects_empty_text(self):
        assert_no_match("a", "")
        assert_no_match("abc", "")

    def test_plain_literal(self):
        assert_matches("abc", "abc")

    def test_literal_is_fullmatch_only(self):
        assert_no_match("abc", "abcd")
        assert_no_match("abc", "xabc")
        assert_no_match("abc", "ab")

    def test_unicode_literals(self):
        assert_matches("héllo wörld", "héllo wörld")
        assert_matches("日本語", "日本語")
        assert_matches("🎉🎉", "🎉🎉")
        assert_no_match("日本語", "日本")
        assert_no_match("héllo", "hello")

    def test_bare_brackets_are_literals(self):
        # A bare "]" or "}" outside a class/quantifier is an ordinary char.
        assert_matches("]", "]")
        assert_matches("}", "}")
        assert_matches("a]b}c", "a]b}c")


class TestDot:
    def test_dot_matches_any_single_char(self):
        assert_matches("a.c", "abc")
        assert_matches("a.c", "a c")
        assert_matches(".", "é")

    def test_dot_matches_newline(self):
        # Deliberately different from re without DOTALL.
        assert_matches(".", "\n")
        assert_matches("a.b", "a\nb")
        assert_matches("...", "a\nb")

    def test_dot_consumes_exactly_one_char(self):
        assert_no_match(".", "")
        assert_no_match(".", "ab")
        assert_no_match("a.c", "ac")


class TestEscapes:
    @pytest.mark.parametrize("char", list(".*+?{}[]^$()|\\"))
    def test_escaped_metacharacters_are_literal(self, char):
        assert_matches("\\" + char, char)

    def test_escaped_metacharacter_does_not_act_specially(self):
        assert_no_match(r"\.", "a")
        assert_no_match(r"\*", "")
        assert_no_match(r"a\+", "aa")

    def test_escaped_ordinary_char_is_itself(self):
        assert_matches(r"\a", "a")
        assert_matches(r"\n", "n")  # literal "n", NOT a newline
        assert_no_match(r"\n", "\n")


class TestCharClasses:
    def test_plain_members(self):
        assert_matches("[abc]", "a")
        assert_matches("[abc]", "b")
        assert_matches("[abc]", "c")
        assert_no_match("[abc]", "d")
        assert_no_match("[abc]", "ab")

    def test_ranges_are_inclusive(self):
        for ch in "abcxyz":
            assert_matches("[a-z]", ch)
        assert_no_match("[a-z]", "A")
        assert_no_match("[a-z]", "0")
        assert_matches("[0-9]", "5")
        assert_matches("[α-ω]", "λ")  # Unicode range

    def test_negation(self):
        assert_matches("[^abc]", "d")
        assert_matches("[^abc]", "\n")  # negated class matches newline too
        assert_no_match("[^abc]", "a")
        assert_matches("[^a-z]", "A")
        assert_no_match("[^a-z]", "m")

    def test_escaped_members(self):
        assert_matches(r"[\]]", "]")
        assert_matches(r"[\-]", "-")
        assert_matches(r"[\^]", "^")
        assert_matches(r"[\\]", "\\")
        assert_no_match(r"[\]]", "[")

    def test_escaped_range_endpoint(self):
        # ']' (0x5D) through '_' (0x5F): ] ^ _
        assert_matches(r"[\]-_]", "]")
        assert_matches(r"[\]-_]", "^")
        assert_matches(r"[\]-_]", "_")
        assert_no_match(r"[\]-_]", "a")

    def test_dash_at_edge_is_literal(self):
        assert_matches("[-a]", "-")
        assert_matches("[-a]", "a")
        assert_matches("[a-]", "-")
        assert_matches("[a-]", "a")
        assert_no_match("[a-]", "b")

    def test_caret_not_first_is_literal(self):
        assert_matches("[a^]", "^")
        assert_matches("[a^]", "a")
        assert_no_match("[a^]", "b")

    def test_class_with_quantifier(self):
        assert_matches("[ab]{2}", "ab")
        assert_matches("[ab]{2}", "ba")
        assert_no_match("[ab]{2}", "a")
        assert_no_match("[ab]{2}", "aba")
        assert_matches("[0-9a-f]+", "a0f9")


# ---------------------------------------------------------------------------
# quantifiers: zero times, lower bound, upper bound, out of bounds
# ---------------------------------------------------------------------------


class TestQuantifiers:
    @pytest.mark.parametrize(
        "pattern,text,ok",
        [
            # ?
            ("ab?c", "ac", True),       # zero times
            ("ab?c", "abc", True),      # upper bound (1)
            ("ab?c", "abbc", False),    # beyond upper bound
            ("ab?c", "ab", False),
            # *
            ("ab*c", "ac", True),       # zero times
            ("ab*c", "abc", True),
            ("ab*c", "abbbbc", True),
            ("ab*c", "a", False),
            ("ab*c", "abd", False),
            # +
            ("ab+c", "ac", False),      # below lower bound
            ("ab+c", "abc", True),      # lower bound
            ("ab+c", "abbc", True),
            # {m}
            ("a{3}", "aaa", True),
            ("a{3}", "aa", False),
            ("a{3}", "aaaa", False),
            ("a{3}", "", False),
            ("a{1}", "a", True),
            ("a{1}", "aa", False),
            # {m,}
            ("a{2,}", "a", False),
            ("a{2,}", "aa", True),      # lower bound
            ("a{2,}", "aaaaa", True),
            ("a{0,}", "", True),
            ("a{0,}", "aaa", True),
            # {m,n}
            ("a{2,4}", "a", False),     # below lower bound
            ("a{2,4}", "aa", True),     # lower bound
            ("a{2,4}", "aaa", True),
            ("a{2,4}", "aaaa", True),   # upper bound
            ("a{2,4}", "aaaaa", False),  # beyond upper bound
            # zero-width bounds
            ("a{0}", "", True),
            ("a{0}", "a", False),
            ("a{0,2}", "", True),
            ("a{0,2}", "a", True),
            ("a{0,2}", "aa", True),
            ("a{0,2}", "aaa", False),
            # quantified dot and class
            (".{2}", "ab", True),
            (".{2}", "a", False),
            ("[ab]*", "", True),
            ("[ab]*", "abba", True),
            ("[ab]*", "abc", False),
            # stacked optional atoms
            ("x?y?z?", "", True),
            ("x?y?z?", "xyz", True),
            ("x?y?z?", "xz", True),
            ("x?y?z?", "yx", False),
        ],
    )
    def test_quantifier(self, pattern, text, ok):
        if ok:
            assert_matches(pattern, text)
        else:
            assert_no_match(pattern, text)


class TestBacktracking:
    """Adjacent quantifiers must give characters back (or fail as a whole)."""

    @pytest.mark.parametrize(
        "pattern,text,ok",
        [
            ("a*a", "aa", True),     # a* must hand one char back
            ("a*a", "a", True),
            ("a*ab", "aab", True),   # succeeds only after backtracking
            ("a*ab", "aa", False),   # every split tried, none works
            ("a+a", "aa", True),
            ("a+a", "a", False),
            ("a{2,3}ab", "aaab", True),
            ("a{2,3}ab", "aaaab", True),
            ("a{2,3}ab", "aab", False),  # would need fewer than 2 leading a's
            ("a{2,3}ab", "ab", False),
            ("[0-9]*[0-9]", "123", True),
            ("[0-9]*[0-9]", "a", False),
            (".*.", "ab", True),
            (".*a", "ba", True),
            (".*a", "ab", False),
            ("a*b*c*", "aabbcc", True),
            ("a*b*c*", "acb", False),
        ],
    )
    def test_combined_quantifiers(self, pattern, text, ok):
        if ok:
            assert_matches(pattern, text)
        else:
            assert_no_match(pattern, text)


# ---------------------------------------------------------------------------
# compiled Pattern reuse
# ---------------------------------------------------------------------------


class TestPatternReuse:
    def test_success_failure_success_leaves_no_state(self):
        p = compile("a+b")
        assert_match(p.fullmatch("aab"), "aab")   # success
        assert p.fullmatch("b") is None           # failure
        assert p.fullmatch("") is None            # failure
        assert_match(p.fullmatch("ab"), "ab")     # success again
        assert_match(p.fullmatch("aaab"), "aaab")

    def test_interleaved_patterns_do_not_interfere(self):
        p1 = compile("[0-9]+")
        p2 = compile("[a-z]+")
        assert_match(p1.fullmatch("42"), "42")
        assert_match(p2.fullmatch("abc"), "abc")
        assert p1.fullmatch("abc") is None
        assert p2.fullmatch("42") is None
        assert_match(p1.fullmatch("7"), "7")

    def test_pattern_attributes_and_repr(self):
        source = r"[a-z]+[0-9]{2,4}\.txt"
        p = compile(source)
        assert isinstance(p, Pattern)
        assert p.pattern == source
        assert repr(p) == f"<Pattern {source!r}>"

    def test_pattern_and_text_are_not_mutated(self):
        source = "a+b"
        text = "aab"
        p = compile(source)
        for _ in range(3):
            p.fullmatch(text)
            fullmatch(source, text)
        assert p.pattern == source
        assert text == "aab"


# ---------------------------------------------------------------------------
# Match object semantics
# ---------------------------------------------------------------------------


class TestMatchObject:
    def test_match_observables(self):
        m = fullmatch("a.c", "a\nc")
        assert_match(m, "a\nc")

    def test_match_of_empty_text(self):
        m = fullmatch("", "")
        assert_match(m, "")
        assert m.span() == (0, 0)

    def test_group_zero_is_the_whole_text(self):
        m = fullmatch("x+", "xxx")
        assert m.group(0) == "xxx"
        assert m.group() == "xxx"

    @pytest.mark.parametrize("index", [1, 2, 100, -1])
    def test_nonzero_group_index_raises_indexerror(self, index):
        m = fullmatch("a", "a")
        with pytest.raises(IndexError):
            m.group(index)

    def test_failure_is_exactly_none(self):
        assert fullmatch("a", "b") is None
        assert compile("a").fullmatch("b") is None


# ---------------------------------------------------------------------------
# exhaustive equivalence of the two public entry points
# ---------------------------------------------------------------------------


EXHAUSTIVE_PATTERNS = [
    "",
    "a",
    ".",
    "a.",
    ".b",
    "ab",
    "[ab]",
    "[^ab]",
    "[a-c]",
    "a?",
    "a*",
    "a+",
    "a{2}",
    "a{1,2}",
    "a{2,}",
    "a*b",
    "ba?",
    "[a-c]*",
    "ab?c",
    r"\.",
    r"a\+",
    "[^a]+",
    ".*",
]

EXHAUSTIVE_TEXTS = [
    "",
    "a",
    "b",
    "c",
    "ab",
    "ba",
    "aa",
    "abc",
    "aab",
    "\n",
    "a\nb",
    "é",
    "+",
]


class TestEntryPointEquivalence:
    @pytest.mark.parametrize("pattern", EXHAUSTIVE_PATTERNS)
    @pytest.mark.parametrize("text", EXHAUSTIVE_TEXTS)
    def test_fullmatch_equals_compile_then_fullmatch(self, pattern, text):
        direct = fullmatch(pattern, text)
        reused = compile(pattern).fullmatch(text)
        assert (direct is None) == (reused is None)
        if direct is not None:
            assert direct.group() == reused.group()
            assert direct.span() == reused.span()
            assert direct.start() == reused.start()
            assert direct.end() == reused.end()

    @pytest.mark.parametrize("pattern", EXHAUSTIVE_PATTERNS)
    @pytest.mark.parametrize("text", EXHAUSTIVE_TEXTS)
    def test_repeated_execution_is_deterministic(self, pattern, text):
        p = compile(pattern)
        first = p.fullmatch(text)
        for _ in range(3):
            again = p.fullmatch(text)
            assert (first is None) == (again is None)
            if first is not None:
                assert again.group() == first.group()
                assert again.span() == first.span()
        direct_again = fullmatch(pattern, text)
        assert (first is None) == (direct_again is None)


# ---------------------------------------------------------------------------
# syntax errors: type, pos and message
# ---------------------------------------------------------------------------

# (pattern, expected pos, expected message fragment)
SYNTAX_ERROR_CASES = [
    # unclosed / empty character classes
    ("[abc", 0, "unterminated character class"),
    ("[", 0, "unterminated character class"),
    ("a[bc", 1, "unterminated character class"),
    ("[]", 0, "empty character class"),
    ("[^]", 0, "empty character class"),
    # reversed range (pos = offset of the range's lower endpoint)
    ("[z-a]", 1, "reversed character range"),
    ("ab[9-0]", 3, "reversed character range"),
    # dangling backslash
    ("ab\\", 2, "dangling backslash"),
    ("\\", 0, "dangling backslash"),
    ("[a\\", 2, "dangling backslash in character class"),
    # quantifier with no preceding atom
    ("*a", 0, "quantifier with no preceding atom"),
    ("+x", 0, "quantifier with no preceding atom"),
    ("?y", 0, "quantifier with no preceding atom"),
    ("{2}a", 0, "quantifier with no preceding atom"),
    ("{2,3}x", 0, "quantifier with no preceding atom"),
    # repeated quantifier on one atom
    ("a**", 2, "multiple quantifiers on one atom"),
    ("a*?", 2, "multiple quantifiers on one atom"),
    ("a+*", 2, "multiple quantifiers on one atom"),
    ("a{2}*", 4, "multiple quantifiers on one atom"),
    ("a{2}{3}", 4, "multiple quantifiers on one atom"),
    # invalid or unclosed brace bounds (pos = offset of "{")
    ("a{,2}", 1, "invalid quantifier bounds"),
    ("a{2x}", 1, "invalid quantifier bounds"),
    ("a{2,x}", 1, "invalid quantifier bounds"),
    ("a{2,3", 1, "invalid quantifier bounds"),
    ("a{2", 1, "unclosed quantifier bounds"),
    ("{abc}", 0, "invalid quantifier bounds"),
    # upper bound smaller than lower bound
    ("a{3,2}", 1, "quantifier upper bound is smaller than lower bound"),
    ("x{10,1}", 1, "quantifier upper bound is smaller than lower bound"),
    # groups, alternation, anchors: explicitly unsupported
    ("(a", 0, "groups and alternation are not supported"),
    ("a)b", 1, "groups and alternation are not supported"),
    ("a|b", 1, "groups and alternation are not supported"),
    ("^a", 0, "anchors are not supported"),
    ("a$", 1, "anchors are not supported"),
]


class TestSyntaxErrors:
    @pytest.mark.parametrize("pattern,pos,fragment", SYNTAX_ERROR_CASES)
    def test_compile_raises(self, pattern, pos, fragment):
        with pytest.raises(RegexSyntaxError) as excinfo:
            compile(pattern)
        exc = excinfo.value
        assert exc.pos == pos
        assert fragment in str(exc)
        assert f"at position {pos}" in str(exc)

    @pytest.mark.parametrize("pattern,pos,fragment", SYNTAX_ERROR_CASES)
    def test_fullmatch_raises_the_same_error(self, pattern, pos, fragment):
        with pytest.raises(RegexSyntaxError) as excinfo:
            fullmatch(pattern, "anything")
        assert excinfo.value.pos == pos
        assert f"at position {pos}" in str(excinfo.value)

    def test_regex_syntax_error_is_a_value_error(self):
        assert issubclass(RegexSyntaxError, ValueError)
        with pytest.raises(ValueError):
            compile("[")

    def test_error_pos_is_zero_based_offset_of_first_bad_char(self):
        try:
            compile("abc[")
        except RegexSyntaxError as exc:
            assert exc.pos == 3
            assert "abc["[exc.pos] == "["
        else:  # pragma: no cover - must not happen
            pytest.fail("expected RegexSyntaxError")


# ---------------------------------------------------------------------------
# type errors
# ---------------------------------------------------------------------------


class TestTypeErrors:
    @pytest.mark.parametrize("bad", [None, 1, 1.5, b"a", ["a"], ("a",), object()])
    def test_non_str_pattern_raises_type_error(self, bad):
        with pytest.raises(TypeError):
            compile(bad)
        with pytest.raises(TypeError):
            fullmatch(bad, "text")

    @pytest.mark.parametrize("bad", [None, 1, 1.5, b"a", ["a"], object()])
    def test_non_str_text_raises_type_error(self, bad):
        with pytest.raises(TypeError):
            compile("a").fullmatch(bad)
        with pytest.raises(TypeError):
            fullmatch("a", bad)

    def test_type_error_is_not_a_syntax_error(self):
        with pytest.raises(TypeError) as excinfo:
            compile(None)
        assert not isinstance(excinfo.value, RegexSyntaxError)


# ---------------------------------------------------------------------------
# package surface and CLI (must stay unchanged)
# ---------------------------------------------------------------------------


class TestPackageSurface:
    def test_exports(self):
        assert set(regex_engine.__all__) == {
            "__version__",
            "RegexSyntaxError",
            "Match",
            "Pattern",
            "compile",
            "fullmatch",
        }
        for name in regex_engine.__all__:
            assert hasattr(regex_engine, name)

    def test_version_string(self):
        assert regex_engine.__version__ == "0.1.0"


class TestCLI:
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "regex_engine", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    def test_version(self):
        result = self.run_cli("version")
        assert result.returncode == 0
        assert result.stdout.strip() == regex_engine.__version__

    @pytest.mark.parametrize("command", ["help", "-h", "--help"])
    def test_help(self, command):
        result = self.run_cli(command)
        assert result.returncode == 0
        assert "usage:" in result.stdout

    def test_no_args_prints_help(self):
        result = self.run_cli()
        assert result.returncode == 0
        assert "usage:" in result.stdout

    def test_unknown_command(self):
        result = self.run_cli("bogus")
        assert result.returncode == 2
        assert "unknown command" in result.stderr
