"""Descarga únicamente bank-full.csv desde la fuente pública de UCI."""

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip"
DESTINATION = Path(__file__).resolve().parent / "data" / "bank-full.csv"


def main():
    if DESTINATION.exists():
        print(f"Ya existe: {DESTINATION}")
        return
    with urlopen(URL, timeout=60) as response:
        payload = response.read()
    with ZipFile(BytesIO(payload)) as outer:
        bank_zip = next(name for name in outer.namelist() if name.endswith("bank.zip"))
        with ZipFile(BytesIO(outer.read(bank_zip))) as inner:
            name = next(name for name in inner.namelist() if name.endswith("bank-full.csv"))
            DESTINATION.parent.mkdir(parents=True, exist_ok=True)
            DESTINATION.write_bytes(inner.read(name))
    print(f"Dataset guardado: {DESTINATION}")


if __name__ == "__main__":
    main()
