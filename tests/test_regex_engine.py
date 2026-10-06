"""System tests for the frozen public API of ``regex_engine``.

Everything here goes through the documented entry points only:
``compile``, ``fullmatch``, ``Pattern``, ``Match`` and
``RegexSyntaxError`` as re-exported by the package.  Private helpers
(``_parse``, ``_run``, ``_pieces`` ...) are never touched.

The standard-library ``re`` module is deliberately NOT used as an
oracle: this engine differs from it on purpose (``.`` matches newlines,
only full matches succeed, and the set of rejected patterns is
different), so expected results are stated explicitly.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from itertools import product

import regex_engine
from regex_engine import Match, Pattern, RegexSyntaxError, compile, fullmatch


def matched_text(result):
    """Normalise a match result: None stays None, a Match becomes its text."""
    return None if result is None else result.group(0)


class PublicApiTest(unittest.TestCase):
    """The package surface stays exactly as documented."""

    def test_exports(self):
        self.assertEqual(
            set(regex_engine.__all__),
            {"__version__", "RegexSyntaxError", "Match", "Pattern",
             "compile", "fullmatch"},
        )

    def test_version_string(self):
        self.assertIsInstance(regex_engine.__version__, str)
        self.assertEqual(regex_engine.__version__, "0.1.0")

    def test_compile_returns_pattern(self):
        self.assertIsInstance(compile("a+"), Pattern)

    def test_pattern_repr_and_pattern_attribute(self):
        p = compile("a+b")
        self.assertEqual(p.pattern, "a+b")
        self.assertEqual(repr(p), "<Pattern 'a+b'>")

    def test_syntax_error_is_a_value_error(self):
        self.assertTrue(issubclass(RegexSyntaxError, ValueError))


class EmptyAndLiteralTest(unittest.TestCase):
    def test_empty_pattern_matches_empty_text(self):
        m = fullmatch("", "")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(), "")

    def test_empty_pattern_rejects_nonempty_text(self):
        self.assertIsNone(fullmatch("", "a"))
        self.assertIsNone(fullmatch("", " "))

    def test_nonempty_pattern_rejects_empty_text(self):
        self.assertIsNone(fullmatch("a", ""))
        self.assertIsNone(fullmatch("abc", ""))

    def test_literal_concatenation(self):
        self.assertIsNotNone(fullmatch("abc", "abc"))
        self.assertIsNone(fullmatch("abc", "ab"))     # too short
        self.assertIsNone(fullmatch("abc", "abcd"))   # too long (fullmatch)
        self.assertIsNone(fullmatch("abc", "abd"))

    def test_unicode_literals(self):
        self.assertIsNotNone(fullmatch("héllo", "héllo"))
        self.assertIsNotNone(fullmatch("日本語", "日本語"))
        self.assertIsNotNone(fullmatch("🙂🙂", "🙂🙂"))
        self.assertIsNone(fullmatch("日本語", "日本"))
        self.assertIsNone(fullmatch("héllo", "hello"))

    def test_bare_closing_brackets_are_literals(self):
        # A "]" or "}" outside a class/quantifier is an ordinary character.
        self.assertIsNotNone(fullmatch("]", "]"))
        self.assertIsNotNone(fullmatch("}", "}"))
        self.assertIsNone(fullmatch("]", "}"))


class DotTest(unittest.TestCase):
    def test_dot_matches_any_single_character(self):
        for ch in ("a", "Z", "0", " ", "\t", "🙂"):
            with self.subTest(ch=ch):
                self.assertIsNotNone(fullmatch(".", ch))

    def test_dot_matches_newline(self):
        # Unlike the default of many regex flavours, "." matches "\n" here.
        self.assertIsNotNone(fullmatch(".", "\n"))
        self.assertIsNotNone(fullmatch("a.c", "a\nc"))

    def test_dot_consumes_exactly_one_character(self):
        self.assertIsNone(fullmatch(".", ""))
        self.assertIsNone(fullmatch(".", "ab"))
        self.assertIsNotNone(fullmatch("...", "a\nb"))


class EscapeTest(unittest.TestCase):
    def test_escaped_metacharacters_are_literal(self):
        for escaped, char in [
            (r"\.", "."), (r"\*", "*"), (r"\+", "+"), (r"\?", "?"),
            (r"\[", "["), (r"\]", "]"), (r"\{", "{"), (r"\}", "}"),
            (r"\(", "("), (r"\|", "|"), (r"\\", "\\"),
        ]:
            with self.subTest(escaped=escaped):
                self.assertIsNotNone(fullmatch(escaped, char))

    def test_escaped_metacharacter_loses_special_meaning(self):
        self.assertIsNone(fullmatch(r"\.", "a"))      # not "any char"
        self.assertIsNone(fullmatch(r"a\*", "aa"))    # not a quantifier
        self.assertIsNotNone(fullmatch(r"a\*", "a*"))

    def test_escaped_ordinary_character_is_literal(self):
        self.assertIsNotNone(fullmatch(r"\a", "a"))
        self.assertIsNone(fullmatch(r"\a", "b"))

    def test_escape_inside_longer_pattern(self):
        self.assertIsNotNone(fullmatch(r"a\.txt", "a.txt"))
        self.assertIsNone(fullmatch(r"a\.txt", "a_txt"))


class CharacterClassTest(unittest.TestCase):
    def test_plain_members(self):
        for ch in "abc":
            self.assertIsNotNone(fullmatch("[abc]", ch))
        self.assertIsNone(fullmatch("[abc]", "d"))
        self.assertIsNone(fullmatch("[abc]", ""))
        self.assertIsNone(fullmatch("[abc]", "ab"))

    def test_range(self):
        self.assertIsNotNone(fullmatch("[a-z]", "m"))
        self.assertIsNotNone(fullmatch("[a-z]", "a"))
        self.assertIsNotNone(fullmatch("[a-z]", "z"))
        self.assertIsNone(fullmatch("[a-z]", "A"))
        self.assertIsNone(fullmatch("[a-z]", "0"))

    def test_multiple_ranges_and_members(self):
        self.assertIsNotNone(fullmatch("[a-c0-2_]", "b"))
        self.assertIsNotNone(fullmatch("[a-c0-2_]", "1"))
        self.assertIsNotNone(fullmatch("[a-c0-2_]", "_"))
        self.assertIsNone(fullmatch("[a-c0-2_]", "d"))
        self.assertIsNone(fullmatch("[a-c0-2_]", "3"))

    def test_negation(self):
        self.assertIsNone(fullmatch("[^a-z]", "m"))
        self.assertIsNotNone(fullmatch("[^a-z]", "A"))
        self.assertIsNotNone(fullmatch("[^abc]", "x"))
        self.assertIsNone(fullmatch("[^abc]", "b"))

    def test_negated_class_matches_newline(self):
        self.assertIsNotNone(fullmatch("[^a]", "\n"))

    def test_escapable_members(self):
        self.assertIsNotNone(fullmatch(r"[\]]", "]"))
        self.assertIsNotNone(fullmatch(r"[\\]", "\\"))
        self.assertIsNotNone(fullmatch(r"[\^]", "^"))
        self.assertIsNotNone(fullmatch(r"[a\-z]", "-"))
        self.assertIsNone(fullmatch(r"[\]]", "a"))

    def test_dash_at_class_edge_is_literal(self):
        self.assertIsNotNone(fullmatch("[a-]", "-"))
        self.assertIsNotNone(fullmatch("[a-]", "a"))
        self.assertIsNone(fullmatch("[a-]", "b"))

    def test_caret_not_at_start_is_literal(self):
        self.assertIsNotNone(fullmatch("[a^]", "^"))
        self.assertIsNotNone(fullmatch("[a^]", "a"))

    def test_unicode_range(self):
        self.assertIsNotNone(fullmatch("[α-ω]", "β"))
        self.assertIsNone(fullmatch("[α-ω]", "a"))
        self.assertIsNotNone(fullmatch("[^α-ω]", "a"))
        self.assertIsNone(fullmatch("[^α-ω]", "β"))

    def test_class_with_quantifier(self):
        self.assertIsNotNone(fullmatch("[0-9]+", "40812"))
        self.assertIsNone(fullmatch("[0-9]+", "408a"))
        self.assertIsNone(fullmatch("[0-9]+", ""))


class QuantifierTest(unittest.TestCase):
    # (pattern, texts that match, texts that do not)
    CASES = [
        ("a?", ["", "a"], ["aa", "b", "ab"]),
        ("a*", ["", "a", "aaa"], ["b", "ab", "ba"]),
        ("a+", ["a", "aa", "aaaa"], ["", "b", "aab"]),
        ("a{0}", [""], ["a", "aa"]),
        ("a{2}", ["aa"], ["", "a", "aaa"]),
        ("a{2,}", ["aa", "aaa", "aaaaaa"], ["", "a"]),
        ("a{0,2}", ["", "a", "aa"], ["aaa"]),
        ("a{2,4}", ["aa", "aaa", "aaaa"], ["", "a", "aaaaa"]),
        ("a{3,3}", ["aaa"], ["aa", "aaaa"]),
        ("ab{2,3}c", ["abbc", "abbbc"], ["abc", "abbbbc", "abbdc"]),
        (".*", ["", "anything at all", "a\nb\nc"], []),
        (".+", ["a", "a\nb"], [""]),
        ("[ab]{2}", ["aa", "ab", "ba", "bb"], ["", "a", "aaa", "ac"]),
        ("x?y", ["y", "xy"], ["xxy", "x", ""]),
    ]

    def test_quantifier_bounds(self):
        for pattern, hits, misses in self.CASES:
            for text in hits:
                with self.subTest(pattern=pattern, text=text):
                    self.assertIsNotNone(fullmatch(pattern, text))
            for text in misses:
                with self.subTest(pattern=pattern, text=text):
                    self.assertIsNone(fullmatch(pattern, text))

    def test_zero_repetitions_of_empty_text(self):
        self.assertIsNotNone(fullmatch("a*", ""))
        self.assertIsNotNone(fullmatch("[a-z]*", ""))
        self.assertIsNotNone(fullmatch("a{0,5}", ""))
        self.assertIsNone(fullmatch("a+", ""))
        self.assertIsNone(fullmatch("a{1,}", ""))

    def test_quantified_unicode_atom(self):
        self.assertIsNotNone(fullmatch("🙂{2}", "🙂🙂"))
        self.assertIsNone(fullmatch("🙂{2}", "🙂"))
        self.assertIsNotNone(fullmatch("é+", "ééé"))


class BacktrackingTest(unittest.TestCase):
    """Adjacent quantifiers must share characters via backtracking."""

    def test_backtrack_to_success(self):
        # a* must give one character back so the trailing atom can match.
        self.assertIsNotNone(fullmatch("a*a", "a"))
        self.assertIsNotNone(fullmatch("a*a", "aa"))
        self.assertIsNotNone(fullmatch("a*ab", "aab"))
        self.assertIsNotNone(fullmatch("a+ab", "aab"))
        self.assertIsNotNone(fullmatch(".*a", "bba"))
        self.assertIsNotNone(fullmatch("[a-z]*[0-9]", "abc1"))
        self.assertIsNotNone(fullmatch("a{1,3}ab", "aaab"))

    def test_backtrack_to_final_failure(self):
        # Greedy expansion and every backtracking alternative all fail.
        self.assertIsNone(fullmatch("a*ab", "aa"))
        self.assertIsNone(fullmatch("a+ab", "aa"))
        self.assertIsNone(fullmatch(".*a", "bb"))
        self.assertIsNone(fullmatch("[a-z]+[0-9]", "abc"))
        self.assertIsNone(fullmatch("a*a", "b"))

    def test_greedy_still_allows_full_match_only(self):
        self.assertIsNone(fullmatch("a*", "aab"))
        self.assertIsNotNone(fullmatch("a.*", "a\n\n"))


class PatternReuseTest(unittest.TestCase):
    def test_success_failure_success_sequence(self):
        p = compile("a+b")
        first = p.fullmatch("aab")
        self.assertIsNotNone(first)
        self.assertIsNone(p.fullmatch("aa"))       # failure leaves no state
        second = p.fullmatch("ab")
        self.assertIsNotNone(second)
        self.assertEqual(second.group(), "ab")
        # And the first success is still reproducible afterwards.
        self.assertIsNotNone(p.fullmatch("aab"))

    def test_interleaved_patterns_do_not_interfere(self):
        pa = compile("a+")
        pb = compile("b+")
        self.assertIsNotNone(pa.fullmatch("aa"))
        self.assertIsNone(pa.fullmatch("bb"))
        self.assertIsNotNone(pb.fullmatch("bb"))
        self.assertIsNone(pb.fullmatch("aa"))
        self.assertIsNotNone(pa.fullmatch("aaa"))

    def test_pattern_and_text_are_not_mutated(self):
        source = "a+b"
        p = compile(source)
        text = "aab"
        snapshot = (p.pattern, text)
        p.fullmatch(text)
        p.fullmatch("zz")
        self.assertEqual((p.pattern, text), snapshot)


class MatchObjectTest(unittest.TestCase):
    TEXT = "a\nc"

    def setUp(self):
        self.match = fullmatch("a.c", self.TEXT)
        self.assertIsInstance(self.match, Match)

    def test_group_zero_is_the_whole_text(self):
        self.assertEqual(self.match.group(), self.TEXT)
        self.assertEqual(self.match.group(0), self.TEXT)

    def test_start_end_span(self):
        self.assertEqual(self.match.start(), 0)
        self.assertEqual(self.match.end(), len(self.TEXT))
        self.assertEqual(self.match.span(), (0, len(self.TEXT)))

    def test_string_attribute(self):
        self.assertEqual(self.match.string, self.TEXT)

    def test_repr(self):
        self.assertEqual(repr(self.match), f"<Match {self.TEXT!r}>")
        self.assertEqual(repr(fullmatch("", "")), "<Match ''>")

    def test_nonzero_group_index_raises_index_error(self):
        for index in (1, 2, -1, 100):
            with self.subTest(index=index):
                with self.assertRaises(IndexError):
                    self.match.group(index)


class EntryPointEquivalenceTest(unittest.TestCase):
    """fullmatch(p, t) must always equal compile(p).fullmatch(t)."""

    PATTERNS = [
        "", "a", "b", "ab", "ba", ".", "..", "a.", ".b",
        "a?", "a*", "a+", "b?", "b*",
        "a{2}", "b{1,2}", "a{0,2}b",
        "[ab]", "[^a]", "[a-b]", "[ab]*", "[^b]+",
        "a*b", "a+b?", ".*b", "[ab]{2}",
        r"\.", r"a\*", "]",
    ]

    TEXTS = [""] + [
        "".join(chars)
        for length in (1, 2, 3)
        for chars in product("ab", repeat=length)
    ] + ["\n", "a\nb", ".", "*", "]", "é"]

    def assert_same_result(self, left, right):
        if left is None or right is None:
            self.assertIsNone(left)
            self.assertIsNone(right)
        else:
            self.assertEqual(left.group(0), right.group(0))
            self.assertEqual(left.span(), right.span())
            self.assertEqual(left.string, right.string)

    def test_direct_call_matches_compiled_call(self):
        for pattern in self.PATTERNS:
            compiled = compile(pattern)
            for text in self.TEXTS:
                with self.subTest(pattern=pattern, text=text):
                    self.assert_same_result(
                        fullmatch(pattern, text), compiled.fullmatch(text)
                    )

    def test_results_are_deterministic_across_repeats(self):
        for pattern in self.PATTERNS:
            compiled = compile(pattern)
            for text in self.TEXTS:
                first = matched_text(compiled.fullmatch(text))
                for _ in range(2):
                    with self.subTest(pattern=pattern, text=text):
                        self.assertEqual(
                            matched_text(compiled.fullmatch(text)), first
                        )
                        self.assertEqual(
                            matched_text(fullmatch(pattern, text)), first
                        )


class SyntaxErrorTest(unittest.TestCase):
    # (pattern, expected pos, a fragment of the error message)
    CASES = [
        # Unclosed or empty character classes.
        ("[", 0, "unterminated character class"),
        ("[ab", 0, "unterminated character class"),
        ("[a-", 0, "unterminated character class"),
        ("[]", 0, "empty character class"),
        ("[^]", 0, "empty character class"),
        # Reversed range: pos points at the range's lower endpoint.
        ("[z-a]", 1, "reversed character range"),
        ("ab[9-0]", 3, "reversed character range"),
        # Dangling backslash, inside and outside a class.
        ("\\", 0, "dangling backslash"),
        ("ab\\", 2, "dangling backslash"),
        ("[\\", 1, "dangling backslash in character class"),
        ("[a\\", 2, "dangling backslash in character class"),
        # Quantifier with no preceding atom.
        ("*a", 0, "no preceding atom"),
        ("?a", 0, "no preceding atom"),
        ("+a", 0, "no preceding atom"),
        ("{2}a", 0, "no preceding atom"),
        ("{2,}a", 0, "no preceding atom"),
        ("{2,3}a", 0, "no preceding atom"),
        # Repeated quantifiers on one atom.
        ("a**", 2, "multiple quantifiers"),
        ("a*?", 2, "multiple quantifiers"),
        ("a+*", 2, "multiple quantifiers"),
        ("a??", 2, "multiple quantifiers"),
        ("a*{2}", 2, "multiple quantifiers"),
        ("a{2}*", 4, "multiple quantifiers"),
        ("a{2}{3}", 4, "multiple quantifiers"),
        # Illegal or unclosed brace bounds.
        ("a{,2}", 1, "invalid quantifier bounds"),
        ("a{x}", 1, "invalid quantifier bounds"),
        ("a{2x}", 1, "invalid quantifier bounds"),
        ("a{2", 1, "unclosed quantifier bounds"),
        ("a{2,", 1, "invalid quantifier bounds"),
        ("a{2,3", 1, "invalid quantifier bounds"),
        ("a{2,x}", 1, "invalid quantifier bounds"),
        # Upper bound smaller than lower bound.
        ("a{3,2}", 1, "upper bound is smaller than lower bound"),
        ("a{10,2}", 1, "upper bound is smaller than lower bound"),
        # Alternation and anchors are still not supported.
        ("a|b", 1, "alternation is not supported"),
        ("(a|b)", 2, "alternation is not supported"),
        ("^a", 0, "anchors are not supported"),
        ("a$", 1, "anchors are not supported"),
        # Unbalanced groups: pos points at the offending parenthesis.
        ("(", 0, "unbalanced parenthesis"),
        ("(a", 0, "unbalanced parenthesis"),
        ("a(b", 1, "unbalanced parenthesis"),
        ("((a)", 0, "unbalanced parenthesis"),
        ("(a(b)", 0, "unbalanced parenthesis"),
        (")", 0, "unbalanced parenthesis"),
        ("a)b", 1, "unbalanced parenthesis"),
        ("(a))", 3, "unbalanced parenthesis"),
        # Quantifiers still need an atom, including right after "(" or ")".
        ("(?)", 1, "no preceding atom"),
        ("(a)*?", 4, "multiple quantifiers"),
        ("(a){", 3, "invalid quantifier bounds"),
    ]

    def test_each_bad_pattern_raises_with_position(self):
        for pattern, pos, fragment in self.CASES:
            with self.subTest(pattern=pattern):
                with self.assertRaises(RegexSyntaxError) as ctx:
                    compile(pattern)
                exc = ctx.exception
                self.assertEqual(exc.pos, pos)
                self.assertIn(fragment, str(exc))
                self.assertIn(f"position {pos}", str(exc))

    def test_fullmatch_raises_the_same_errors(self):
        for pattern, pos, _ in self.CASES:
            with self.subTest(pattern=pattern):
                with self.assertRaises(RegexSyntaxError) as ctx:
                    fullmatch(pattern, "anything")
                self.assertEqual(ctx.exception.pos, pos)

    def test_error_positions_point_into_the_pattern(self):
        for pattern, pos, _ in self.CASES:
            self.assertGreaterEqual(pos, 0)
            self.assertLess(pos, len(pattern))


class TypeErrorTest(unittest.TestCase):
    BAD_VALUES = [None, 0, 1, 3.14, b"a", bytearray(b"a"), ["a"], ("a",), object()]

    def test_non_str_pattern_raises_type_error(self):
        for value in self.BAD_VALUES:
            with self.subTest(value=repr(value)):
                with self.assertRaises(TypeError):
                    compile(value)
                with self.assertRaises(TypeError):
                    fullmatch(value, "a")

    def test_non_str_text_raises_type_error(self):
        p = compile("a*")
        for value in self.BAD_VALUES:
            with self.subTest(value=repr(value)):
                with self.assertRaises(TypeError):
                    p.fullmatch(value)
                with self.assertRaises(TypeError):
                    fullmatch("a*", value)

    def test_type_error_is_not_a_syntax_error(self):
        # The two failure channels must stay distinguishable.
        with self.assertRaises(TypeError) as ctx:
            compile(None)
        self.assertNotIsInstance(ctx.exception, RegexSyntaxError)


class GroupSyntaxTest(unittest.TestCase):
    """Groups parse, nest and take quantifiers; brackets are balanced."""

    def test_plain_group_matches(self):
        self.assertIsNotNone(fullmatch("(a)", "a"))
        self.assertIsNone(fullmatch("(a)", "b"))
        self.assertIsNone(fullmatch("(a)", "ab"))
        self.assertIsNone(fullmatch("(a)", ""))

    def test_empty_group(self):
        m = fullmatch("()", "")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "")
        self.assertEqual(m.start(1), 0)
        self.assertEqual(m.end(1), 0)
        self.assertEqual(m.span(1), (0, 0))

    def test_group_inside_concatenation(self):
        self.assertIsNotNone(fullmatch("a(bc)d", "abcd"))
        self.assertIsNone(fullmatch("a(bc)d", "abd"))

    def test_nested_groups_numbered_by_left_paren(self):
        m = fullmatch("((a)(b))", "ab")
        self.assertEqual(m.groups(), ("ab", "a", "b"))
        self.assertEqual(m.span(1), (0, 2))
        self.assertEqual(m.span(2), (0, 1))
        self.assertEqual(m.span(3), (1, 2))

    def test_deeply_nested_empty_groups(self):
        m = fullmatch("((()))", "")
        self.assertEqual(m.groups(), ("", "", ""))
        for i in (1, 2, 3):
            self.assertEqual(m.span(i), (0, 0))

    def test_group_quantifiers_match(self):
        for pattern, hits, misses in [
            ("(ab)?", ["", "ab"], ["a", "abab"]),
            ("(ab)*", ["", "ab", "abab"], ["a", "aba"]),
            ("(ab)+", ["ab", "abab"], ["", "a"]),
            ("(ab){2}", ["abab"], ["", "ab", "ababab"]),
            ("(ab){1,2}", ["ab", "abab"], ["", "ababab"]),
            ("(ab){0,1}", ["", "ab"], ["abab"]),
        ]:
            for text in hits:
                with self.subTest(pattern=pattern, text=text):
                    self.assertIsNotNone(fullmatch(pattern, text))
            for text in misses:
                with self.subTest(pattern=pattern, text=text):
                    self.assertIsNone(fullmatch(pattern, text))

    def test_quantifiers_inside_groups(self):
        self.assertIsNotNone(fullmatch("(a+bc)", "aaabc"))
        self.assertIsNone(fullmatch("(a+bc)", "bc"))
        self.assertIsNotNone(fullmatch("([0-9]+-[0-9]+)", "12-9"))

    def test_classes_and_escapes_around_groups(self):
        self.assertIsNotNone(fullmatch(r"(\()", "("))
        self.assertIsNotNone(fullmatch(r"(\))", ")"))
        self.assertIsNotNone(fullmatch("([ab]+)", "abba"))
        self.assertIsNone(fullmatch("([ab]+)", "abc"))

    def test_unbalanced_groups_raise_at_the_opening_paren(self):
        for pattern, pos in [
            ("(", 0), ("(a", 0), ("a(b", 1), ("((a)", 0), ("(a(b)", 0),
            ("()(", 2),
        ]:
            with self.subTest(pattern=pattern):
                with self.assertRaises(RegexSyntaxError) as ctx:
                    compile(pattern)
                self.assertEqual(ctx.exception.pos, pos)
                self.assertIn("unbalanced parenthesis", str(ctx.exception))

    def test_extra_closing_paren_raises_at_it(self):
        for pattern, pos in [(")", 0), ("a)b", 1), ("(a))", 3), (")(", 0)]:
            with self.subTest(pattern=pattern):
                with self.assertRaises(RegexSyntaxError) as ctx:
                    compile(pattern)
                self.assertEqual(ctx.exception.pos, pos)
                self.assertIn("unbalanced parenthesis", str(ctx.exception))

    def test_alternation_and_anchors_still_rejected(self):
        for pattern, pos in [("(a|b)", 2), ("a|b", 1),
                             ("(^a)", 1), ("(a$)", 2)]:
            with self.subTest(pattern=pattern):
                with self.assertRaises(RegexSyntaxError) as ctx:
                    compile(pattern)
                self.assertEqual(ctx.exception.pos, pos)


class GroupCaptureTest(unittest.TestCase):
    """Captured substrings, spans and the groups() tuple."""

    def test_group_returns_captured_text(self):
        m = fullmatch("a(bc)d", "abcd")
        self.assertEqual(m.group(1), "bc")
        self.assertEqual(m.group(0), "abcd")
        self.assertEqual(m.group(), "abcd")

    def test_groups_tuple_order(self):
        m = fullmatch("(a)(b)(c)", "abc")
        self.assertEqual(m.groups(), ("a", "b", "c"))

    def test_groups_empty_when_no_groups(self):
        self.assertEqual(fullmatch("abc", "abc").groups(), ())
        self.assertEqual(fullmatch("", "").groups(), ())

    def test_start_end_span_unicode_offsets(self):
        m = fullmatch("(..)x", "é🙂x")
        self.assertEqual(m.group(1), "é🙂")
        self.assertEqual(m.start(1), 0)
        self.assertEqual(m.end(1), 2)
        self.assertEqual(m.span(1), (0, 2))
        m2 = fullmatch("あ(い)う", "あいう")
        self.assertEqual(m2.span(1), (1, 2))

    def test_optional_group_absent(self):
        m = fullmatch("a(b)?c", "ac")
        self.assertIsNone(m.group(1))
        self.assertEqual(m.start(1), -1)
        self.assertEqual(m.end(1), -1)
        self.assertEqual(m.span(1), (-1, -1))
        self.assertEqual(m.groups(), (None,))

    def test_optional_group_present(self):
        m = fullmatch("a(b)?c", "abc")
        self.assertEqual(m.group(1), "b")
        self.assertEqual(m.span(1), (1, 2))

    def test_group_participating_but_matching_empty_string(self):
        m = fullmatch("a()b", "ab")
        self.assertEqual(m.group(1), "")
        self.assertEqual(m.start(1), 1)
        self.assertEqual(m.end(1), 1)
        self.assertEqual(m.span(1), (1, 1))

    def test_zero_repetition_group_is_absent_not_empty(self):
        m = fullmatch("(ab)*", "")
        self.assertIsNone(m.group(1))
        self.assertEqual(m.span(1), (-1, -1))

    def test_last_participation_is_kept_in_a_repeat(self):
        m = fullmatch("(ab)+", "abab")
        self.assertEqual(m.group(1), "ab")
        self.assertEqual(m.span(1), (2, 4))
        self.assertEqual(fullmatch("(a)+", "aaa").group(1), "a")
        self.assertEqual(fullmatch("(ab){3}", "ababab").group(1), "ab")

    def test_greedy_split_between_two_groups(self):
        m = fullmatch("(a*)(a*)", "aa")
        self.assertEqual(m.group(1), "aa")
        self.assertEqual(m.group(2), "")
        self.assertEqual(m.span(1), (0, 2))
        self.assertEqual(m.span(2), (2, 2))

    def test_nested_capture_in_repeat(self):
        m = fullmatch("((a+)b)+", "aabab")
        self.assertEqual(m.group(1), "ab")
        self.assertEqual(m.group(2), "a")
        self.assertEqual(m.span(1), (3, 5))

    def test_absent_nested_group_keeps_last_participation(self):
        m = fullmatch("(a(b)?)*", "aba")
        self.assertEqual(m.group(1), "a")
        self.assertEqual(m.group(2), "b")

    def test_absent_nested_group_is_none_when_never_participated(self):
        m = fullmatch("(a(b)?)*", "aaa")
        self.assertEqual(m.group(1), "a")
        self.assertIsNone(m.group(2))

    def test_backtracking_restores_captures(self):
        # The outer group gives characters back; its earlier captures revert.
        m = fullmatch("(a(b)?)*.b", "abab")
        self.assertEqual(m.group(1), "ab")
        self.assertEqual(m.group(2), "b")
        self.assertEqual(m.span(1), (0, 2))
        # A whole optional group abandoned by backtracking leaves no capture.
        m2 = fullmatch("(a(b)c)?a*", "aa")
        self.assertIsNone(m2.group(1))
        self.assertIsNone(m2.group(2))

    def test_group_backtracks_for_following_atom(self):
        m = fullmatch("(a*)a", "aaa")
        self.assertEqual(m.group(1), "aa")
        self.assertEqual(m.span(1), (0, 2))

    def test_empty_repeat_does_not_hang_and_stops(self):
        m = fullmatch("(a*)+", "aaa")
        self.assertEqual(m.group(1), "")
        self.assertEqual(m.span(1), (3, 3))
        self.assertEqual(fullmatch("(a?)*", "a").group(1), "")
        self.assertEqual(fullmatch("()*", "").group(1), "")
        self.assertEqual(fullmatch("(){2}", "").group(1), "")
        self.assertEqual(fullmatch("(a*){2}", "a").span(1), (1, 1))

    def test_empty_repeat_with_real_suffix(self):
        m = fullmatch("()*a", "a")
        self.assertEqual(m.group(1), "")
        self.assertEqual(m.span(1), (0, 0))

    def test_invalid_group_index_raises_index_error(self):
        m = fullmatch("(a)", "a")
        for index in (-1, -2, 2, 100):
            with self.subTest(index=index):
                with self.assertRaises(IndexError):
                    m.group(index)
                with self.assertRaises(IndexError):
                    m.start(index)
                with self.assertRaises(IndexError):
                    m.end(index)
                with self.assertRaises(IndexError):
                    m.span(index)
        m0 = fullmatch("abc", "abc")
        for index in (-1, 1, 2):
            with self.subTest(index=index):
                with self.assertRaises(IndexError):
                    m0.group(index)

    def test_string_and_repr_unchanged(self):
        m = fullmatch("(a)(b)", "ab")
        self.assertEqual(m.string, "ab")
        self.assertEqual(repr(m), "<Match 'ab'>")


class GroupReuseAndEquivalenceTest(unittest.TestCase):
    """Compiled patterns reuse cleanly and both entry points agree."""

    def test_failure_does_not_pollute_later_match(self):
        p = compile("(a+)(b)?")
        ok = p.fullmatch("aaab")
        self.assertEqual(ok.groups(), ("aaa", "b"))
        self.assertIsNone(p.fullmatch("zzz"))
        again = p.fullmatch("aaa")
        self.assertEqual(again.group(1), "aaa")
        self.assertIsNone(again.group(2))
        once_more = p.fullmatch("aaab")
        self.assertEqual(once_more.groups(), ("aaa", "b"))

    def test_entry_points_agree_with_groups(self):
        patterns = [
            "(a)", "(ab)+", "(a*)(a*)", "a(b)?c", "((a)(b))",
            "(a(b)?)*", "(x)?y", "(a*)+", "()*", "([0-9]+)(x)?",
            "(ab){2}", "(a?)(b?)", "(..)(..)",
        ]
        texts = ["", "a", "ab", "aa", "aaa", "abab", "ac", "xy",
                 "aba", "12x", "abcd", "éé"]
        for pattern in patterns:
            compiled = compile(pattern)
            for text in texts:
                with self.subTest(pattern=pattern, text=text):
                    direct = fullmatch(pattern, text)
                    reused = compiled.fullmatch(text)
                    if direct is None or reused is None:
                        self.assertIsNone(direct)
                        self.assertIsNone(reused)
                    else:
                        self.assertEqual(direct.groups(), reused.groups())
                        for i in range(len(direct.groups()) + 1):
                            self.assertEqual(direct.span(i), reused.span(i))
                        self.assertEqual(direct.group(0), reused.group(0))


class CommandLineTest(unittest.TestCase):
    """The existing ``version``/``help`` command line behaviour is frozen."""

    def run_cli(self, *args):
        from regex_engine.__main__ import main

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_version(self):
        code, out, err = self.run_cli("version")
        self.assertEqual(code, 0)
        self.assertEqual(out, regex_engine.__version__ + "\n")
        self.assertEqual(err, "")

    def test_help_and_aliases(self):
        for args in (["help"], ["-h"], ["--help"], []):
            with self.subTest(args=args):
                code, out, err = self.run_cli(*args)
                self.assertEqual(code, 0)
                self.assertIn("usage:", out)
                self.assertEqual(err, "")

    def test_unknown_command(self):
        code, out, err = self.run_cli("bogus")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("unknown command: bogus", err)
        self.assertIn("usage:", err)


if __name__ == "__main__":
    unittest.main()
