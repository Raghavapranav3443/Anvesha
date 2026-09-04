import ast
import sys

files = [
    "satquery/agent.py",
    "satquery/server/main.py",
    "satquery/tools_impl.py",
    "satquery/io_utils.py",
]

ok = True
for f in files:
    try:
        ast.parse(open(f, encoding="utf-8").read())
        print(f"{f}: OK")
    except SyntaxError as e:
        print(f"{f}: SYNTAX ERROR at line {e.lineno}: {e.msg}")
        ok = False

sys.exit(0 if ok else 1)

