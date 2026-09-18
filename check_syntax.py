import ast
import sys

files = [
    "anvesha/agent.py",
    "anvesha/server/main.py",
    "anvesha/tools_impl.py",
    "anvesha/io_utils.py",
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

