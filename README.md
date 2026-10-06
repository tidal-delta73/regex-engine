# regex-engine

Regular-expression engine and matcher.

Pure-Python, no runtime dependencies.

## Usage

```bash
python3 -m regex_engine version
python3 -m regex_engine help
```

## Python API

```python
from regex_engine import compile, fullmatch, Match, RegexSyntaxError

p = compile(r"[a-z]+[0-9]{2,4}\.txt")  # reusable compiled pattern
p.pattern          # original pattern string
m = p.fullmatch("hello123.txt")        # Match, or None
# fullmatch(pattern, text) is equivalent to compile(pattern).fullmatch(text)

m.group()          # full matched text
m.start()          # 0
m.end()            # len(text)
m.span()           # (0, len(text))
```

Numbered capturing groups `(...)` may be nested and can take the same
postfix quantifiers as an atom:

```python
m = fullmatch(r"([0-9]+)-([0-9]+)", "12-9")
m.group(1)         # '12'
m.groups()         # ('12', '9')
m.span(2)          # (3, 4) -- Unicode code-point offsets
```

Groups are numbered 1, 2, ... by the position of their opening
parenthesis; group 0 is the whole match. A group that does not take part
in the final match gives `group() is None`, `start()/end() == -1` and
`span() == (-1, -1)`; a group that participates but matches the empty
string gives `''` with equal start/end offsets. In a repetition the
capture of the last participation is kept, and captures abandoned while
backtracking are rolled back.

Supported pattern syntax: literals (Unicode) and concatenation, `.`
(any character including newline), backslash-escaped metacharacters,
character classes with `a-z` ranges and leading `^` negation, numbered
capturing groups `(...)` (nestable, empty groups allowed), and the
postfix quantifiers `?` `*` `+` `{m}` `{m,}` `{m,n}` on atoms or whole
groups. A quantifier is greedy by default and may take one mode marker
immediately after it: `?` makes it lazy (try the fewest repetitions and
extend only when the rest of the pattern cannot finish) and `+` makes
it possessive (take the longest repetition and commit it, never giving
characters back), as in `a*?`, `a++` and `(ab){1,}+`. Alternation,
anchors, named groups and backreferences are not supported. Only full
matches succeed. Invalid patterns raise `RegexSyntaxError` (a
`ValueError` subclass) whose `pos` is the zero-based offset of the
first bad character (the unmatched parenthesis for unbalanced groups;
an extra quantifier after a mode marker points at that extra
character); non-`str` pattern or text raises `TypeError`.

