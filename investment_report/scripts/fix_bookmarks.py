"""docx 后处理：为书签分配唯一 w:id（docx 库生成的书签 id 全部为 1）。"""
import re
import shutil
import sys
import zipfile

src = sys.argv[1]
tmp = src + ".tmp"
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "word/document.xml":
            xml = data.decode("utf-8")
            counter = {"n": 0}

            def repl(m):
                tag = m.group(1)
                if tag == "bookmarkStart":
                    counter["n"] += 1
                return f'<w:{tag}{m.group(2)}w:id="{counter["n"] + 1000}"'

            xml = re.sub(r'<w:(bookmarkStart|bookmarkEnd)([^>]*?)w:id="\d+"', repl, xml)
            data = xml.encode("utf-8")
        zout.writestr(item, data)
shutil.move(tmp, src)
print("bookmarks renumbered")
