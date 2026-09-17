# -*- coding: utf-8 -*-
"""校验 docs/ 与 src/ 下所有 md 文件的相对链接是否指向存在的文件。"""
import os, re, posixpath

ROOT = os.path.dirname(os.path.abspath(__file__))
link_re = re.compile(r"\]\(([^)]+)\)")
broken = 0
for base in ["docs", "src"]:
    for root, _, files in os.walk(os.path.join(ROOT, base)):
        for fn in files:
            if not fn.endswith(".md"):
                continue
            path = os.path.join(root, fn)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            for m in link_re.finditer(text):
                url = m.group(1).strip()
                if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url) or url.startswith("#"):
                    continue
                url = url.split("#", 1)[0]
                if not url:
                    continue
                target = os.path.normpath(os.path.join(root, url))
                if not os.path.exists(target):
                    broken += 1
                    print(f"BROKEN: {os.path.relpath(path, ROOT)} -> {m.group(1)}")
print(f"broken links: {broken}")
