"""Write a minimal multi-page PDF with a real text layer (Helvetica), for tests.

No reportlab needed: the file is hand-assembled PDF 1.4 and PyPDF2 reads its text back.
"""
import textwrap
from pathlib import Path


def _escape(line: str) -> str:
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_text_pdf(path: Path, pages: list[str], *, width_chars: int = 90) -> Path:
    objects: list[bytes] = []

    def add(obj: str | bytes) -> int:
        objects.append(obj.encode("latin-1") if isinstance(obj, str) else obj)
        return len(objects)  # 1-based object number

    catalog = add("<< /Type /Catalog /Pages 2 0 R >>")  # placeholder, rewritten below
    pages_obj = add("<< >>")  # placeholder
    font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    for text in pages:
        lines = []
        for para in text.split("\n"):
            lines.extend(textwrap.wrap(para, width_chars) or [""])
        body = "BT /F1 10 Tf 40 760 Td 13 TL " + " ".join(f"({_escape(l)}) Tj T*" for l in lines) + " ET"
        content = add(f"<< /Length {len(body)} >>\nstream\n{body}\nendstream")
        page_ids.append(add(f"<< /Type /Page /Parent {pages_obj} 0 R /MediaBox [0 0 612 792] "
                            f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>"))
    objects[pages_obj - 1] = (f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] "
                              f"/Count {len(page_ids)} >>").encode()
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode()

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    Path(path).write_bytes(bytes(out))
    return Path(path)
