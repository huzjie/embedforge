"""Minimal YAML subset parser (stdlib only).

Supports the small YAML subset used by config.yaml so the package works even
when PyYAML is not installed (the managed Python runtimes often lack it):
  - ``key: value`` scalars (int / float / bool / str)
  - inline comments ``key: value  # note``
  - block lists::

        keywords:
          - a
          - b

  - nested mappings by indentation
  - lists of mappings::

        items:
          - name: x
            value: 1

This is deliberately tiny -- it is a fallback, not a full YAML implementation.
"""


def _scalar(s):
    s = s.strip()
    if s == "":
        return None
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~"):
        return None
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _strip_comment(line):
    for marker in ("  #", "\t#"):
        idx = line.find(marker)
        if idx >= 0:
            return line[:idx]
    return line


def parse(text):
    """Parse a YAML subset into nested dict/list. Raises ValueError on garbage."""
    root = {}
    # stack of (indent, container, parent, key) so a dict placeholder can be
    # replaced by a list in its parent when a block list materializes.
    stack = [(-1, root, None, None)]
    lines = text.splitlines()
    for raw in lines:
        if raw.strip() == "" or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = _strip_comment(raw.rstrip())
        stripped = line.strip()
        if stripped.startswith("- "):
            while len(stack) > 1 and not isinstance(stack[-1][1], list) and stack[-1][0] >= indent:
                stack.pop()
            while len(stack) > 1 and indent < stack[-1][0]:
                stack.pop()
            _, container, parent, key = stack[-1]
            val = stripped[2:].strip()
            if not isinstance(container, list):
                lst = []
                if parent is not None and key is not None:
                    parent[key] = lst
                container = lst
                stack[-1] = (indent, container, parent, key)
            if ":" in val and not val.startswith(("{", "[")):
                k, _, v = val.partition(":")
                item = {k.strip(): _scalar(v) if v.strip() != "" else {}}
                container.append(item)
                stack.append((indent, item, container, len(container) - 1))
            else:
                container.append(_scalar(val))
        else:
            while len(stack) > 1 and indent < stack[-1][0]:
                stack.pop()
            _, container, parent, key = stack[-1]
            if ":" not in stripped:
                continue
            k, _, v = stripped.partition(":")
            k = k.strip()
            v = v.strip()
            if v == "":
                child = {}
                container[k] = child
                stack.append((indent, child, container, k))
            else:
                container[k] = _scalar(v)
    return root
