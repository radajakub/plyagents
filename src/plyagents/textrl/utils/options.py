from __future__ import annotations

# options written name=value, separated by commas; a value may be a component config in brackets


def split_options(text: str, where: str) -> dict[str, str]:
    # "a=x,b=y[c=1,d=2]" -> {"a": "x", "b": "y[c=1,d=2]"}: a comma inside brackets belongs to the value
    options: dict[str, str] = {}
    depth, start = 0, 0
    for index, char in enumerate(text + ","):
        depth += {"[": 1, "]": -1}.get(char, 0)
        if depth < 0:
            raise ValueError(f"Unbalanced brackets in {where}: {text!r}")
        if char == "," and depth == 0:
            option = text[start:index]
            start = index + 1
            if not option:
                continue
            name, separator, value = option.partition("=")
            if not separator or name in options:
                raise ValueError(f"Invalid or repeated option {option!r} in {where}: {text!r}")
            options[name] = value
    if depth != 0:
        raise ValueError(f"Unbalanced brackets in {where}: {text!r}")
    return options


def read_options(text: str, allowed: set[str], where: str) -> dict[str, str]:
    options = split_options(text, where)
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f"Unknown options {unknown} in {where}; known: {sorted(allowed)}")
    return options
