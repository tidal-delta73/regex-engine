import pytest

import regex_engine as re_engine
from regex_engine import Match, RegexSyntaxError, compile, fullmatch


# --- basic matching -------------------------------------------------------

def test_empty_pattern_matches_only_empty_text():
    assert fullmatch("", "") is not None
    assert fullmatch("", "a") is None


def test_literal_and_concatenation():
    assert fullmatch("abc", "abc") is not None
    assert fullmatch("abc", "ab") is None
    assert fullmatch("abc", "abcd") is None
    assert fullmatch("abc", "xbc") is None


def test_unicode_literals():
    assert fullmatch("héllo世", "héllo世") is not None
    assert fullmatch("héllo世", "héllo") is None


def test_dot_matches_any_single_char_including_newline():
    assert fullmatch("a.c", "abc") is not None
    assert fullmatch("a.c", "a\nc") is not None
    assert fullmatch("a.c", "ac") is None
    assert fullmatch(".", "ab") is None


def test_escaped_metacharacters_are_literals():
    assert fullmatch(r"\.\*\+\?\{\}\[\]\\", ".*+?{}[]\\") is not None
    assert fullmatch(r"\(\)\|\^\$", "()|^$") is not None
    assert fullmatch(r"\.", "x") is None


def test_match_object_protocol():
    m = fullmatch("a.c", "abc")
    assert isinstance(m, Match)
    assert m.group() == "abc"
    assert m.start() == 0
    assert m.end() == 3
    assert m.span() == (0, 3)


# --- character classes ----------------------------------------------------

def test_class_members_and_ranges():
    assert fullmatch("[abc]", "b") is not None
    assert fullmatch("[abc]", "d") is None
    assert fullmatch("[a-z]", "m") is not None
    assert fullmatch("[a-z]", "A") is None
    assert fullmatch("[a-cx-z]", "y") is not None


def test_negated_class_consumes_exactly_one_char():
    assert fullmatch("[^a]", "b") is not None
    assert fullmatch("[^a]", "a") is None
    assert fullmatch("[^a]", "bb") is None
    assert fullmatch("[^a-z]", "é") is not None


def test_class_escapes_and_literal_dash():
    assert fullmatch(r"[\]]", "]") is not None
    assert fullmatch(r"[a\-z]", "-") is not None
    assert fullmatch(r"[\^]", "^") is not None
    assert fullmatch(r"[\\]", "\\") is not None
    assert fullmatch("[-a]", "-") is not None
    assert fullmatch("[a-]", "-") is not None


# --- quantifiers ----------------------------------------------------------

@pytest.mark.parametrize("pattern,text,ok", [
    ("ab?c", "ac", True), ("ab?c", "abc", True), ("ab?c", "abbc", False),
    ("ab*c", "ac", True), ("ab*c", "abbbc", True), ("ab*c", "abcx", False),
    ("ab+c", "ac", False), ("ab+c", "abc", True), ("ab+c", "abbbc", True),
    ("a{3}", "aaa", True), ("a{3}", "aa", False), ("a{3}", "aaaa", False),
    ("a{2,}", "aaaaa", True), ("a{2,}", "a", False),
    ("a{1,3}", "a", True), ("a{1,3}", "aaa", True), ("a{1,3}", "aaaa", False),
    ("a{0}", "", True), ("a{0}", "a", False),
    ("a{0,0}", "", True),
    ("[ab]{2,3}c", "abac", True), ("[ab]{2,3}c", "abc", True),
    (".*", "anything\nat all", True),
])
def test_quantifiers(pattern, text, ok):
    assert (fullmatch(pattern, text) is not None) is ok


def test_quantifiers_count_unicode_characters():
    assert fullmatch(".{3}", "日本語") is not None
    assert fullmatch(".{3}", "日本") is None
    assert fullmatch("[一-鿿]{2}", "日本") is not None


def test_backtracking_finds_full_match():
    assert fullmatch("a*a", "aa") is not None
    assert fullmatch("a*ab", "aaab") is not None
    assert fullmatch(".*c", "abc") is not None


# --- compiled objects -----------------------------------------------------

def test_compiled_pattern_reusable_and_keeps_pattern():
    p = compile("a+")
    assert p.pattern == "a+"
    assert p.fullmatch("a") is not None
    assert p.fullmatch("b") is None
    assert p.fullmatch("aaa") is not None  # no residue from earlier calls
    assert p.fullmatch("a").group() == "a"


def test_package_fullmatch_matches_compile_then_call():
    p = compile("[a-c]{2}")
    for text in ("ab", "cc", "ad", "abc"):
        assert (fullmatch("[a-c]{2}", text) is None) == (p.fullmatch(text) is None)


# --- type errors ----------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 1, b"a", ["a"]])
def test_non_str_pattern_raises_type_error(bad):
    with pytest.raises(TypeError):
        compile(bad)
    with pytest.raises(TypeError):
        fullmatch(bad, "x")


@pytest.mark.parametrize("bad", [None, 1, b"x"])
def test_non_str_text_raises_type_error(bad):
    with pytest.raises(TypeError):
        fullmatch("a", bad)
    with pytest.raises(TypeError):
        compile("a").fullmatch(bad)


# --- syntax errors --------------------------------------------------------

@pytest.mark.parametrize("pattern", [
    "\\",            # dangling backslash
    "ab\\",          # dangling backslash at end
    "[", "[abc",     # unterminated class
    "[]", "[^]",     # empty class
    "[z-a]",         # reversed range
    "*", "+a", "?", "{2}",   # quantifier without atom
    "a**", "a*+", "a?{2}", "a{1}*", "a{1}{2}",  # repeated quantifier
    "a{", "a{1", "a{1,", "a{1,2",   # unterminated braces
    "a{}", "a{,3}", "a{x}", "a{1x}", "a{1,x}",  # missing/illegal bounds
    "a{3,2}",        # lower > upper
    "(a)", ")", "a|b", "^a", "a$",  # unsupported constructs
])
def test_syntax_errors(pattern):
    with pytest.raises(RegexSyntaxError):
        compile(pattern)


def test_syntax_error_is_value_error_with_offset():
    with pytest.raises(ValueError) as excinfo:
        compile("ab*+")
    assert isinstance(excinfo.value, RegexSyntaxError)
    assert "3" in str(excinfo.value)
    assert excinfo.value.offset == 3


def test_error_offset_points_at_first_offending_char():
    with pytest.raises(RegexSyntaxError) as excinfo:
        compile("ab[cd")
    assert excinfo.value.offset == 2
    with pytest.raises(RegexSyntaxError) as excinfo:
        compile("a\\")
    assert excinfo.value.offset == 1


def test_escaped_reserved_chars_are_not_errors():
    assert fullmatch(r"\(a\|b\)", "(a|b)") is not None
    assert fullmatch(r"\^\$", "^$") is not None


# --- robustness -----------------------------------------------------------

def test_zero_repetitions_and_empty_text_do_not_hang():
    assert fullmatch("a{0}b{0}", "") is not None
    assert fullmatch("x*y*", "") is not None
    assert fullmatch("[^a]*", "") is not None
    assert fullmatch("[^a]*", "bbb") is not None


def test_long_pattern_beyond_recursion_limit():
    # Plain literals: matching stays linear even past the recursion limit.
    pattern = "ab" * 2000
    assert fullmatch(pattern, "ab" * 2000) is not None
    assert fullmatch(pattern, "ab" * 1999 + "a") is None
    assert fullmatch("a?" * 2000, "a" * 2000) is not None
    assert fullmatch("a?" * 2000, "") is not None
