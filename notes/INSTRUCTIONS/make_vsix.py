import os
import requests

EXT_FILE = "/workspace/AAA_project/notes/INSTRUCTIONS/vscode-extensions-uniq.txt"
SAVE_DIR = "vsix_extensions"
os.makedirs(SAVE_DIR, exist_ok=True)


def build_vsix_url(publisher, name):
    return f"https://marketplace.visualstudio.com/_apis/public/gallery/publishers/{publisher}/vsextensions/{name}/latest/vspackage"


with open(EXT_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]


for ext in lines:
    try:
        publisher, name = ext.split(".")
        url = build_vsix_url(publisher, name)
        filename = f"{SAVE_DIR}/{publisher}.{name}.vsix"

        print(f"[+] Скачиваю {ext}...")
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"    ✅ Сохранено как {filename}")
    except Exception as e:
        print(f"    ⚠️ Пропущено {ext}: {e}")
