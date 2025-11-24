import os
from pathlib import Path

os.chdir(Path(__file__).parent.parent)

files = Path("assets/pdfs/").rglob("*")

for file in files:
    if file.suffix != ".pdf" and file.is_file():
        file.unlink()
