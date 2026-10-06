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

m.group()          # full matched text (group 0)
m.group(1)         # text captured by group 1 (None if it did not participate)
m.groups()         # tuple of groups 1..n
m.start()          # 0
m.end()            # len(text)
m.span()           # (0, len(text))
m.span(2)          # (start, end) of group 2, or (-1, -1) if it did not participate
```

Supported pattern syntax: literals (Unicode) and concatenation, `.`
(any character including newline), backslash-escaped metacharacters,
character classes with `a-z` ranges and leading `^` negation, the
postfix quantifiers `?` `*` `+` `{m}` `{m,}` `{m,n}`, and numbered
capturing groups `(...)` (nestable, quantifiable as a whole, numbered
from 1 by their opening parenthesis; group 0 is the whole match).
Alternation and anchors are rejected. Only full matches
succeed. Invalid patterns raise `RegexSyntaxError` (a `ValueError`
subclass) whose `pos` is the zero-based offset of the first bad
character; non-`str` pattern or text raises `TypeError`.

