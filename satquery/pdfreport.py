"""Branded PDF report generation for a run (no external service needed)."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

from .config import CONFIG


def generate_pdf(run_id: str) -> Path | None:
    """Compose a formal, print-ready PDF using reportlab if available;
    falls back to a minimal hand-rolled PDF writer when it isn't."""
    run_dir = CONFIG.runs_dir / run_id
    report = run_dir / "report.json"
    if not report.exists():
        return None
    data = json.loads(report.read_text(encoding="utf-8"))

    out = run_dir / "report.pdf"
    try:
        return _reportlab_pdf(data, run_dir, out, run_id=run_id)
    except ImportError:
        return _minimal_pdf(data, out)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("reportlab PDF failed, falling back to minimal")
        return _minimal_pdf(data, out)


# -----------------------------------------------------------------------
#  Colour palette
# -----------------------------------------------------------------------
_PRIMARY = "#1F5FD6"
_DARK = "#0D1526"
_MUTED = "#5B6B88"
_LIGHT_BG = "#F4F6FA"
_GOOD = "#12805C"
_WARN = "#A16207"


def _reportlab_pdf(data: dict, run_dir: Path, out: Path, run_id: str = ""):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(out), pagesize=A4)
    W, H = A4
    margin = 20 * mm
    y = H - margin
    page_num = [1]

    primary = HexColor(_PRIMARY)
    dark = HexColor(_DARK)
    muted = HexColor(_MUTED)
    light_bg = HexColor(_LIGHT_BG)
    good = HexColor(_GOOD)

    def new_page():
        _draw_footer(c, W, H, margin, page_num[0])
        c.showPage()
        page_num[0] += 1
        return H - margin

    def check_page(need_mm=40):
        nonlocal y
        if y < need_mm * mm:
            y = new_page()

    def draw_line(y_pos, color="#D8DFEA"):
        c.setStrokeColor(HexColor(color))
        c.setLineWidth(0.5)
        c.line(margin, y_pos, W - margin, y_pos)

    def text(txt, size=10, bold=False, dy=5.5 * mm, color=dark,
             x=None, max_width=None):
        nonlocal y
        check_page(30)
        font = "Helvetica-Bold" if bold else "Helvetica"
        c.setFont(font, size)
        c.setFillColor(color)
        cx = x or margin
        if max_width:
            chars_per_line = int(max_width / (size * 0.5))
            for line_text in textwrap.wrap(txt, chars_per_line) or [""]:
                c.drawString(cx, y, line_text)
                y -= size * 1.3
        else:
            c.drawString(cx, y, txt)
            y -= dy

    def section_header(title):
        nonlocal y
        check_page(35)
        y -= 3 * mm
        draw_line(y, _PRIMARY)
        y -= 2 * mm
        text(title, 13, bold=True, dy=7 * mm, color=primary)
        y -= 1 * mm

    def kv_row(key, value, indent=0):
        nonlocal y
        check_page(15)
        x = margin + indent * mm
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(muted)
        c.drawString(x, y, key)
        c.setFont("Helvetica", 9)
        c.setFillColor(dark)
        c.drawString(x + 45 * mm, y, str(value)[:100])
        y -= 4.5 * mm

    # ── Title block ──────────────────────────────────────────────────
    # Blue accent bar
    c.setFillColor(primary)
    c.rect(0, H - 12 * mm, W, 12 * mm, fill=True, stroke=False)
    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, H - 9 * mm, "ANVESHA")
    c.setFont("Helvetica", 9)
    c.drawString(margin + 32 * mm, H - 8 * mm,
                 "Earth Observation & Investigation Report")

    y = H - 18 * mm
    draw_line(y, _PRIMARY)
    y -= 6 * mm

    # ── Metadata block ───────────────────────────────────────────────
    text(f"Run ID:  {data.get('run_id', '')}", 9, dy=4.5 * mm, color=muted)
    text(f"Generated:  {data.get('generated_at', '')}", 9, dy=4.5 * mm, color=muted)
    text(f"System:  {data.get('system', 'Anvesha')}", 9, dy=4.5 * mm, color=muted)
    y -= 2 * mm

    # ── Query ─────────────────────────────────────────────────────────
    section_header("Query")
    text(str(data.get("query", "")), 11, bold=True, dy=6 * mm)
    y -= 2 * mm

    # ── Input Configuration ──────────────────────────────────────────
    section_header("Input Configuration")
    cfg = data.get("input_configuration", {})
    kv_row("Configuration:", cfg.get("configuration_label", cfg.get("configuration", "")))
    kv_row("Selected task:", str(data.get("selected_task", "")))
    kv_row("Confidence:", f"{data.get('confidence', 0):.1%}")
    y -= 2 * mm

    # ── Input Files Table ────────────────────────────────────────────
    inputs = data.get("inputs", [])
    if inputs:
        section_header("Input Files")
        # Table header
        check_page(30)
        c.setFillColor(light_bg)
        c.rect(margin, y - 1 * mm, W - 2 * margin, 6 * mm, fill=True, stroke=False)
        c.setFont("Helvetica-Bold", 8.5)
        c.setFillColor(muted)
        col_x = [margin + 2 * mm, margin + 55 * mm, margin + 95 * mm, margin + 125 * mm]
        for i, hdr in enumerate(["File", "Modality", "Bands", "Geo-ref"]):
            c.drawString(col_x[i], y, hdr)
        y -= 6 * mm
        for inp in inputs:
            check_page(12)
            c.setFont("Helvetica", 8.5)
            c.setFillColor(dark)
            c.drawString(col_x[0], y, str(inp.get("file", ""))[:50])
            c.drawString(col_x[1], y, str(inp.get("modality", "")))
            c.drawString(col_x[2], y, str(inp.get("bands", "")))
            c.drawString(col_x[3], y, "Yes" if inp.get("georeferenced") else "No")
            y -= 4.5 * mm
        y -= 2 * mm

    # ── Answer ────────────────────────────────────────────────────────
    section_header("Answer")
    text(str(data.get("final_answer", data.get("answer", ""))), 10,
         max_width=W - 2 * margin)
    y -= 3 * mm

    # ── Execution Trace ──────────────────────────────────────────────
    trace = data.get("execution_summary", [])
    if trace:
        section_header("Execution Trace")
        for s in trace:
            if s.get("name") == "finish":
                continue
            check_page(15)
            status = s.get("status", "ok")
            status_col = good if status == "ok" else HexColor(_WARN)
            c.setFont("Helvetica", 8.5)
            c.setFillColor(dark)
            name_str = str(s.get("name", ""))
            c.drawString(margin + 2 * mm, y, name_str[:50])
            c.setFillColor(status_col)
            c.drawString(margin + 65 * mm, y, status)
            dur = s.get("duration_ms")
            if dur is not None:
                c.setFillColor(muted)
                c.drawString(margin + 85 * mm, y, f"{dur} ms")
            y -= 4.5 * mm
        y -= 2 * mm

    # ── Structured Outputs (summary) ─────────────────────────────────
    outputs = data.get("outputs", {})
    if outputs:
        section_header("Outputs Summary")
        for key, val in outputs.items():
            if isinstance(val, (dict, list)):
                preview = json.dumps(val, default=str)[:120]
            else:
                preview = str(val)[:120]
            kv_row(f"{key}:", preview, indent=1)
        y -= 2 * mm

    # ── Visual Evidence Note ─────────────────────────────────────────
    extra = data.get("extra", {})
    visual_evidence = extra.get("visual_evidence", {})
    if visual_evidence:
        section_header("Visual Evidence")
        for name, path in visual_evidence.items():
            kv_row(f"{name}:", str(path))
            # Try to embed the image
            img_path = Path(path)
            if img_path.exists():
                check_page(90)
                try:
                    c.drawImage(str(img_path), margin + 5 * mm, y - 70 * mm,
                                width=70 * mm, height=70 * mm,
                                preserveAspectRatio=True)
                    y -= 75 * mm
                except Exception:
                    pass

    _draw_footer(c, W, H, margin, page_num[0])
    c.showPage()
    c.save()
    return out


def _draw_footer(c, W, H, margin, page_num):
    """Draw page number and branding footer."""
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    y_foot = margin - 8 * mm
    c.setStrokeColor(HexColor(_PRIMARY))
    c.setLineWidth(0.3)
    c.line(margin, y_foot + 5 * mm, W - margin, y_foot + 5 * mm)
    c.setFont("Helvetica", 7)
    c.setFillColor(HexColor(_MUTED))
    c.drawString(margin, y_foot, "Anvesha — Earth Observation & Investigation System · SIH26167 · ISRO / SAC")
    c.drawRightString(W - margin, y_foot, f"Page {page_num}")


def _minimal_pdf(data: dict, out: Path):
    """Tiny dependency-free PDF writer (text-only, improved layout)."""
    def esc(s):
        return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    task = data.get("selected_task", "")
    confidence = data.get("confidence", 0)
    answer = str(data.get("final_answer", data.get("answer", "")))

    lines = [
        "ANVESHA - Earth Observation & Investigation Report",
        "",
        f"Run: {data.get('run_id', '')}",
        f"Generated: {data.get('generated_at', '')}",
        "",
        "--- Query ---",
        esc(str(data.get("query", ""))),
        "",
        "--- Analysis ---",
        f"Task: {task}    Confidence: {confidence:.1%}",
        "",
        "--- Answer ---",
    ]
    for chunk in textwrap.wrap(answer, 85):
        lines.append(esc(chunk))
    lines += ["", "--- Execution Trace ---"]
    for s in data.get("execution_summary", []):
        if s.get("name") == "finish":
            continue
        dur = f" ({s.get('duration_ms')} ms)" if s.get("duration_ms") else ""
        lines.append(f"  {s.get('name')} [{s.get('status', 'ok')}]{dur}")

    content = ["BT /F1 10 Tf 50 760 Td 14 TL"]
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
