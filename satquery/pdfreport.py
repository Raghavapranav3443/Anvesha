"""Branded PDF report generation for a run (no external service needed)."""
from __future__ import annotations

import json
from pathlib import Path

from .config import CONFIG


def generate_pdf(run_id: str) -> Path | None:
    """Compose a simple, print-ready PDF using only reportlab if available;
    falls back to a minimal hand-rolled PDF writer when it isn't."""
    run_dir = CONFIG.runs_dir / run_id
    report = run_dir / "report.json"
    if not report.exists():
        return None
    data = json.loads(report.read_text(encoding="utf-8"))

    out = run_dir / "report.pdf"
    try:
        return _reportlab_pdf(data, run_dir, out)
    except ImportError:
        return _minimal_pdf(data, out)


def _reportlab_pdf(data: dict, run_dir: Path, out: Path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(out), pagesize=A4)
    W, H = A4
    y = H - 20 * mm

    def line(txt, size=10, bold=False, dy=6 * mm, color="#0D1526"):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.setFillColorRGB(0x0D / 255, 0x15 / 255, 0x26 / 255)
        c.drawString(20 * mm, y, txt)
        y -= dy

    line("Anvesha — Earth Observation & Investigation Report", 16, bold=True, dy=10 * mm)
    line(f"Run {data.get('run_id', run_id)} · generated {data.get('generated_at', '')}", 9)
    line(f"Query: {data.get('query', '')}", 11, bold=True, dy=8 * mm)
    line(f"Selected task: {data.get('selected_task', '')}    "
         f"Confidence: {data.get('confidence', 0):.2f}", 10, dy=8 * mm)

    line("Answer", 12, bold=True)
    for chunk in _wrap(str(data.get("answer", "")), 95):
        line(chunk, 10, dy=5 * mm)
    y -= 4 * mm

    line("Inputs", 12, bold=True)
    for i in data.get("inputs", []):
        line(f"· {i.get('file')} — {i.get('modality')} · {i.get('bands')} bands"
             f" · {'georeferenced' if i.get('georeferenced') else 'not georeferenced'}",
             9, dy=4.5 * mm)
    y -= 3 * mm

    line("Execution summary", 12, bold=True)
    for s in data.get("execution_summary", []):
        if s.get("name") == "finish":
            continue
        line(f"· {s.get('name')}  [{s.get('status', 'ok')}"
             f"{', ' + str(s.get('duration_ms')) + ' ms' if s.get('duration_ms') else ''}]",
             9, dy=4.5 * mm)

    # evidence images
    for name, y_img in (("change_overlay.png", H - 120 * mm),
                        ("overlay.png", H - 120 * mm)):
        p = run_dir / "visuals" / name
        if p.exists() and y > 60 * mm:
            try:
                c.drawImage(str(p), 20 * mm, 40 * mm, width=80 * mm,
                            height=80 * mm, preserveAspectRatio=True)
                line(f"Evidence: {name}", 9, dy=0)
                break
            except Exception:
                pass

    c.showPage()
    c.save()
    return out


def _wrap(text: str, width: int):
    import textwrap
    return textwrap.wrap(text, width) or [""]


def _minimal_pdf(data: dict, out: Path):
    """Tiny dependency-free PDF writer (text-only)."""
    def esc(s):
        return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    lines = [
        f"Anvesha - EO Investigation Report ({data.get('run_id', '')})",
        f"Query: {data.get('query', '')}",
        f"Task: {data.get('selected_task', '')}   Confidence: {data.get('confidence', 0)}",
        "",
        "Answer:",
        *_wrap(str(data.get("answer", "")), 90),
        "",
        "Execution summary:",
        *[f"  {s.get('name')} [{s.get('status', 'ok')}]" for s in
          data.get("execution_summary", []) if s.get("name") != "finish"],
    ]
    content = ["BT /F1 10 Tf 50 770 Td 14 TL"]
    for ln in lines:
        content.append(f"({esc(ln)}) Tj T*")
    content.append("ET")
    stream = "\n".join(content).encode("latin-1", "replace")

    objs = []
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>")
    objs.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
                + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, o in enumerate(objs, 1):
        offsets.append(len(pdf))
        pdf += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(pdf)
    pdf += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
    for off in offsets[1:]:
        pdf += f"{off:010d} 00000 n \n".encode()
    pdf += (f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF").encode()
    out.write_bytes(bytes(pdf))
    return out
