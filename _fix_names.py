"""Give the public one-shot helper a name that matches the package, not a mangled one.

`build_guidance_field` was renamed to `build_field_guided_search` by the blanket text
replacement. The module already exposes a lower-level ``field.build_field(grid, sources…)``,
so the trajectory-level helper is now called ``field_from_trajectory``.
"""

import pathlib

pkg = pathlib.Path("src/field_guided_search")

RENAMES = [
    ("build_field_guided_search", "field_from_trajectory"),
]

for name in ("__init__.py", "api.py"):
    path = pkg / name
    text = path.read_text(encoding="utf-8")
    for old, new in RENAMES:
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    print("fixed", name)

# README code blocks + API table must use the same names
readme = pathlib.Path("README.md")
s = readme.read_text(encoding="utf-8")
for old, new in RENAMES:
    s = s.replace(old, new)
readme.write_text(s, encoding="utf-8")
print("fixed README.md")

# make sure the public surface is coherent
init = (pkg / "__init__.py").read_text(encoding="utf-8")
print("\nbuild_field_guided_search leftovers:", init.count("build_field_guided_search"))
print("field_from_trajectory present:", "field_from_trajectory" in init)
