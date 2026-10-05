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

Supported pattern syntax: literals (Unicode) and concatenation, `.`
(any character including newline), backslash-escaped metacharacters,
character classes with `a-z` ranges and leading `^` negation, and the
postfix quantifiers `?` `*` `+` `{m}` `{m,}` `{m,n}`. Only full matches
succeed. Invalid patterns raise `RegexSyntaxError` (a `ValueError`
subclass) whose `pos` is the zero-based offset of the first bad
character; non-`str` pattern or text raises `TypeError`.

