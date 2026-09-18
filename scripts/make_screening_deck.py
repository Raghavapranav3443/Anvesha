"""Build the SIH national-screening deck from the official idea-submission template.

Every content element is a NATIVE PowerPoint shape with real, extractable text --
no rasters carry information. That matters because an evaluator (or an ATS/text
extraction pass) sees nothing inside an image, and the original deck put its entire
argument inside five PNGs.

Hard constraints encoded here:
  * Slide 1 is the fixed template title page and is NOT modified.
  * No QR code. The link appears only on the final slide; slides 2-5 carry a small
    pointer phrase telling the reader where it is.
  * All six template section headings are preserved.

Geometry was checked against PowerPoint's own reported text bounds, never
estimated. A `<p:sp>.*?</p:sp>` regex mis-pairs blocks on this template, so shape
surgery uses a balanced-tag scan, and the build ends by pruning now-unreferenced
images so the package does not carry dead megabytes.

Source:  Anvesha-MCS-SIH2026-IDEA-Presentation.pptx   (never modified)
Output:  Anvesha-MCS-SIH2026-Screening.pptx

Run:  python scripts/make_screening_deck.py
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Anvesha-MCS-SIH2026-IDEA-Presentation.pptx"
DST = ROOT / "Anvesha-MCS-SIH2026-Screening.pptx"

EMU = 914400
HUB_URL = "https://anvesha-hub.vercel.app/"

INK, MUTED = "1A1A1A", "444444"
GREEN, GREEN_SOFT, GREEN_LINE = "14693C", "EAF3EC", "C9DED0"
BLUE = "0070C0"
RED, RED_SOFT, RED_LINE = "B03A2E", "FDF1EF", "E8B4AE"
AMBER, AMBER_SOFT, AMBER_LINE = "B9770E", "FDF6E8", "EED9AE"

_next_id = 900

# Shapes this script adds; the layout checker uses this list to test for overlap.
OUR_SHAPES: set[str] = set()


def _id() -> int:
    global _next_id
    _next_id += 1
    return _next_id


def emu(inches: float) -> int:
    return int(round(inches * EMU))


def run(text, sz=1000, b=False, i=False, color=INK, hlink=None, u=False):
    link = f'<a:hlinkClick r:id="{hlink}" tooltip="Open the live evidence brief"/>' if hlink else ""
    ul = ' u="sng"' if u else ""
    return (
        f'<a:r><a:rPr lang="en-IN" sz="{sz}" b="{1 if b else 0}" i="{1 if i else 0}"{ul} dirty="0">'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        f'<a:latin typeface="Calibri"/>{link}</a:rPr><a:t>{escape(text)}</a:t></a:r>'
    )


def para(runs, algn="l", before=0):
    pr = f'<a:pPr algn="{algn}">'
    if before:
        pr += f'<a:spcBef><a:spcPts val="{before}"/></a:spcBef>'
    pr += "</a:pPr>"
    return f"<a:p>{pr}{''.join(runs)}</a:p>"


def shp(name, prst, x, y, w, h, fill=None, line=None, paras=None, anchor="ctr",
        lIns=0.07, rIns=0.07, tIns=0.045, bIns=0.045, adj=None, sharp=False):
    """A native autoshape with optional real text. This is the deck's only building block."""
    global OUR_SHAPES
    OUR_SHAPES.add(name)
    av = f'<a:avLst><a:gd name="adj" fmla="val {adj}"/></a:avLst>' if adj else "<a:avLst/>"
    fill_xml = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else "<a:noFill/>"
    line_xml = (f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
                if line else '<a:ln><a:noFill/></a:ln>')
    body = (f'<p:txBody><a:bodyPr wrap="square" lIns="{emu(lIns)}" rIns="{emu(rIns)}"'
            f' tIns="{emu(tIns)}" bIns="{emu(bIns)}" anchor="{anchor}"><a:noAutofit/></a:bodyPr>'
            f'<a:lstStyle/>{paras or "<a:p/>"}</p:txBody>')
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{_id()}" name="{escape(name)}"/>'
        f'<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
        f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
        f'<a:prstGeom prst="{prst}">{av}</a:prstGeom>{fill_xml}{line_xml}</p:spPr>{body}</p:sp>'
    )


def card(name, x, y, w, h, title, body, line, fill="FFFFFF", title_color=None, tsz=1000, bsz=850):
    tcol = title_color or INK
    ps = [para([run(title, sz=tsz, b=True, color=tcol)])]
    if body:
        ps.append(para([run(body, sz=bsz, color=MUTED)], before=120))
    return shp(name, "roundRect", x, y, w, h, fill=fill, line=line, paras="".join(ps),
               anchor="ctr", adj=7000, tIns=0.05, bIns=0.05)


def chip(name, x, y, n, size=0.30, fill=GREEN):
    return shp(name, "roundRect", x, y, size, size, fill=fill, adj=16000,
               paras=para([run(str(n), sz=1000, b=True, color="FFFFFF")], algn="ctr"),
               lIns=0, rIns=0, tIns=0, bIns=0)


def ribbon(substance, pointer="Live prototype + demo \u2192 link on slide 6"):
    y, h = 6.50, 0.42
    band = shp("Evidence Ribbon", "roundRect", 0.48, y, 12.38, h, fill=GREEN_SOFT,
               line=GREEN_LINE, adj=26000, paras="<a:p/>")
    left = shp("Ribbon Substance", "rect", 0.60, y, 8.85, h,
               paras=para([run(substance, sz=950)]), lIns=0.03, tIns=0, bIns=0)
    right = shp("Ribbon Pointer", "rect", 9.50, y, 3.22, h,
                paras=para([run(pointer, sz=950, b=True, color=GREEN)], algn="r"),
                lIns=0, rIns=0.02, tIns=0, bIns=0)
    return band, left, right


# ---------------------------------------------------------------- XML surgery

_TAG = re.compile(r"<p:(sp|pic|grpSp)(?=[\s/>])")


def _elements(tree):
    pos = 0
    while True:
        m = _TAG.search(tree, pos)
        if not m:
            return
        tag = m.group(1)
        open_re = re.compile(rf"<p:{tag}(?=[\s/>])")
        close_re = re.compile(rf"</p:{tag}>")
        depth, i = 0, m.start()
        while True:
            o = open_re.search(tree, i)
            c = close_re.search(tree, i)
            if c is None:
                return
            if o is not None and o.start() < c.start():
                depth += 1
                i = o.end()
            else:
                depth -= 1
                i = c.end()
                if depth == 0:
                    yield m.start(), c.end(), tree[m.start():c.end()]
                    pos = c.end()
                    break


def _find(tree, name):
    for s, e, x in _elements(tree):
        if f'name="{name}"' in x[:400]:
            return s, e, x
    return None


def add(tree, *shapes):
    assert "</p:spTree>" in tree
    return tree.replace("</p:spTree>", "".join(shapes) + "</p:spTree>")


def drop(tree, name):
    f = _find(tree, name)
    assert f, f"shape not found: {name}"
    return tree[:f[0]] + tree[f[1]:]


# ---------------------------------------------------------------- content data

PROBLEM = [
    ("Terabytes of satellite data generated daily",
     "India produces more EO data than most can analyse"),
    ("Analysis requires expensive GIS expertise",
     "Specialised software + trained analysts = bottleneck"),
    ("Generic AI fails on satellite images",
     "Hallucinates, ignores coordinates, needs internet"),
    ("ISRO needs an air-gapped system",
     "Secure facilities cannot use cloud tools"),
]
SOLUTION = [
    ("Plain-language queries",
     "\u201cHas built-up area increased?\u201d \u2014 no GIS training needed"),
    ("7 specialist AI models",
     "11M-parameter shared encoder + lightweight specialist heads"),
    ("Ledger of every decision",
     "Auditable execution-trace receipt of each AI decision"),
    ("Air-gapped, no internet",
     "Docker + CPU-first, built for ISRO labs"),
]
STACK = [
    ("Python 3.10+", "Scientific computing ecosystem"),
    ("PyTorch + TorchScript", "Train models, export compact int8 versions for CPU"),
    ("FastAPI", "Serves the AI to the web console"),
    ("rasterio", "Reads satellite GeoTIFF imagery with map coordinates"),
    ("React 18 + TypeScript + Tailwind", "Console interface, type-safe and clean"),
    ("Leaflet", "Draws answer maps and region overlays"),
    ("SQLite + Docker", "Offline single-file storage; runs fully air-gapped on CPU"),
    ("185 automated tests", "Every feature verified before release"),
]
PIPELINE = [
    ("Upload", "Analyst gives satellite images + a question in plain English"),
    ("Validate", "System checks format, map coordinates, image pairing and band count"),
    ("Understand", "Agent identifies what the question is actually asking"),
    ("Route to 7 specialist AI models",
     "Shared band-adaptive ResNet-18 (11M params, EuroSAT warm-start) feeds dedicated "
     "heads for VQA, captioning, grounding, change detection and optical\u2013SAR fusion"),
    ("Answer with evidence", "Confidence score, highlighted map overlays, exportable GeoTIFF masks"),
    ("Auditable report", "Every decision recorded in a step-by-step execution trace (PDF/JSON)"),
]
PROVEN = [
    ("Working system, not a concept", "7 specialist models trained; live demo runs end-to-end"),
    ("Measured accuracy",
     "98.86% EuroSAT validation accuracy; change-detection F1 0.818 over n = 1,500 pairs"),
    ("Runs on commodity CPU", "int8 models lose only 0.16% accuracy"),
    ("Engineered for reliability", "185 automated tests guard every release"),
]
CHALLENGES = [
    ("Precise object localisation below target",
     "VRSBench grounding IoU@0.5 0.126, detection rate 0.090 over n = 455 \u2014 measured and published"),
    ("VRSBench captioning and VQA still weak",
     "captioning BLEU-4 0.000 (n = 300) against human references; VQA exact-match 0.175 (n = 200)"),
    ("Domain shift to real ISRO sensors",
     "Cartosat-2S and RISAT differ from every training sensor; counting stays the hardest question type"),
    ("CPU speed on very large scenes",
     "Sliding-window inference scales with scene area"),
]
STRATEGY = [
    ("Interpretable fallback", "physics-based indices ship today; detection-grade upgrade pre-gated"),
    ("Transparent confidence", "every answer labelled with its source; low confidence \u2192 ask, never guess"),
    ("Band-adaptive models", "any sensor auto-detected; the uncertainty safeguard switches to physics mode"),
    ("Tiled int8 inference", "large scenes split into tiles; every run timed and traced"),
]
IMPACT = [
    ("DISASTER RESPONSE", RED, RED_SOFT, RED_LINE,
     "When floods kill the internet, AI still works",
     ["Air-gapped \u2014 works when connectivity is down", "Answers on CPU, zero internet",
      "Flood-extent and change maps in minutes", "Change-detection F1 0.818 (n = 1,500)",
      "Every answer carries an audit trail", "Findings officials can defend"]),
    ("AGRICULTURE & WATER", "0F7B6C", "EBF6F4", "BEE0DA",
     "From raw satellite tiles to field answers",
     ["98.86% EuroSAT scene classification", "No GIS specialist needed",
      "Crop-stress and irrigation zones", "Returned as highlighted map overlays",
      "~\u20b90 per query", "Open data + a laptop, no cloud fees"]),
    ("GOVERNMENT & STRATEGIC", "1B3C7B", "EEF2F9", "C4D0E6",
     "Built for the secure lab, priced for the nation",
     ["Docker + CPU-only + offline", "Deployable in secure ISRO facilities",
      "Reproducible execution trace", "Per answer, with confidence scores",
      "Encroachment and border monitoring", "Multi-date imagery, confidence-scored"]),
]
PATH = [
    ("Adapt to ISRO sensors", "Fine-tune on Cartosat, RISAT, NISAR, MOSDAC"),
    ("Raise accuracy", "Improve VQA and change IoU toward SOTA"),
    ("Secure and control access", "Add auth, RBAC, logs, approvals"),
    ("MLOps and governance", "Versioning, monitoring, retraining, audit trails"),
    ("Scale to full scenes", "Tiling, batch queues, multi-host deployment"),
    ("Stronger NLU", "Better intent understanding for analysts"),
    ("Robust geo-processing", "CRS, nodata, reflectance, cloud-mask handling"),
    ("Data and licence compliance", "Legal review, compliant model distribution"),
    ("Ecosystem integration", "Bhuvan, MOSDAC APIs, SIH-GA systems"),
]


# ---------------------------------------------------------------- slide builders

def slide1(tree):
    """Slide 1 is the fixed template title page. It must not be modified."""
    return tree


def slide2(tree):
    for dead in ("Picture 4", "Picture 11", "TextBox 7"):
        tree = drop(tree, dead)

    # THE PROBLEM | SOLVE | OUR SOLUTION  -- native panels, no raster
    lx, lw, py, ph = 0.48, 5.62, 1.28, 3.06
    rx, rw = 7.32, 5.53
    ps = [para([run("THE PROBLEM", sz=1350, b=True, color=RED)], algn="ctr")]
    for i, (t, d) in enumerate(PROBLEM, 1):
        ps.append(para([run(f"({i})  ", sz=1000, b=True, color=RED), run(t, sz=1000, b=True)],
                       before=210))
        ps.append(para([run(d, sz=950, color=MUTED)]))
    left = shp("Problem Panel", "roundRect", lx, py, lw, ph, fill=RED_SOFT, line=RED_LINE,
               paras="".join(ps), anchor="ctr", adj=4000)
    arrow = shp("Solve Arrow", "rightArrow", 6.20, py + (ph - 0.56) / 2, 1.05, 0.56, fill=GREEN,
                paras=para([run("SOLVE", sz=950, b=True, color="FFFFFF")], algn="ctr"),
                lIns=0, rIns=0.10, tIns=0, bIns=0)
    ps = [para([run("OUR SOLUTION", sz=1350, b=True, color=GREEN)], algn="ctr")]
    for i, (t, d) in enumerate(SOLUTION, 1):
        ps.append(para([run(f"({i})  ", sz=1000, b=True, color=GREEN), run(t, sz=1000, b=True)],
                       before=210))
        ps.append(para([run(d, sz=950, color=MUTED)]))
    right = shp("Solution Panel", "roundRect", rx, py, rw, ph, fill=GREEN_SOFT, line=GREEN_LINE,
                paras="".join(ps), anchor="ctr", adj=4000)

    ps = [para([run("IDEA \u2014 WHAT THE AGENT DOES", sz=900, b=True, color=GREEN)]),
          para([run("Validates every input, routes the query to the specialists it needs, fuses "
                    "their outputs, estimates confidence, and records an auditable execution trace.",
                    sz=900)], before=150),
          para([run("MANDATORY SCOPE \u2014 ALL IMPLEMENTED", sz=900, b=True, color=GREEN)], before=280)]
    for t in ["Single-image visual question answering (mandatory baseline)",
              "Captioning and text-guided region grounding",
              "Bi-temporal change description and change-VQA",
              "Co-registered optical\u2013SAR pair analysis",
              "Agentic tool selection with an observable trace",
              "Remote-sensing adaptation via BigEarthNet.txt"]:
        ps.append(para([run("\u2713  ", sz=900, b=True, color=GREEN), run(t, sz=900)], before=110))
    scope = shp("Mandatory Scope", "rect", 0.48, 4.48, 7.30, 1.90, paras="".join(ps), anchor="ctr")

    proof = shp("Proof Strip", "roundRect", 8.00, 4.48, 4.85, 1.90, fill=GREEN_SOFT,
                line=GREEN_LINE, adj=7000,
                paras="".join([
                    para([run("MEASURED ON THE PRESCRIBED SPLITS", sz=900, b=True, color=GREEN)]),
                    para([run("RSVQA-LR exact-match ", sz=1000), run("0.700", sz=1000, b=True),
                          run("   n = 9,491", sz=900, color=MUTED)], before=200),
                    para([run("LEVIR-CD change F1 ", sz=1000), run("0.818", sz=1000, b=True),
                          run("   n = 1,500", sz=900, color=MUTED)], before=140),
                    para([run("EuroSAT val accuracy ", sz=1000), run("98.86%", sz=1000, b=True),
                          run("   SceneEncoder", sz=900, color=MUTED)], before=140),
                    para([run("WHAT SITS BEHIND THE LINK", sz=900, b=True, color=GREEN)], before=280),
                    para([run("A 3-minute demo video, the execution trace for every answer, full "
                              "benchmark provenance and downloadable GeoTIFF reports.", sz=880,
                              color=MUTED)], before=120),
                ]), anchor="ctr")
    return add(tree, left, arrow, right, scope, proof, *ribbon(
        "Seven specialist tasks on one 11M-parameter band-adaptive ResNet-18 encoder; every answer "
        "ships with the tools it used and the parameters it bound."))


def slide3(tree):
    tree = drop(tree, "Picture 3")
    lx, lw = 0.48, 5.85
    rx, rw = 6.72, 6.13
    head_l = shp("Stack Heading", "rect", lx, 1.30, lw, 0.30,
                 paras=para([run("TECHNOLOGY STACK", sz=1200, b=True, color=BLUE)], algn="ctr"))
    top, avail = 1.68, 4.42
    rh, gap = 0.50, 0.06
    rows = []
    for i, (t, d) in enumerate(STACK):
        y = top + i * (rh + gap)
        rows.append(card(f"Stack {i+1}", lx, y, lw, rh, t, d, GREEN_LINE,
                         title_color=BLUE, tsz=950, bsz=800))
    head_r = shp("Pipeline Heading", "rect", rx, 1.30, rw, 0.30,
                 paras=para([run("HOW IT WORKS \u2014 THE ANVESHA PIPELINE", sz=1200, b=True,
                                 color=GREEN)], algn="ctr"))
    steps = []
    bar_h = 0.40
    step_area = avail - bar_h - 0.10
    sh = (step_area - 5 * 0.10) / 6
    for i, (t, d) in enumerate(PIPELINE):
        y = top + i * (sh + 0.10)
        steps.append(chip(f"Step Chip {i+1}", rx + 0.10, y + 0.10, i + 1, 0.26))
        ps = [para([run(t, sz=950, b=True, color=GREEN)]),
              para([run(d, sz=800, color=MUTED)], before=80)]
        steps.append(shp(f"Step {i+1}", "roundRect", rx + 0.44, y, rw - 0.44, sh,
                         fill="FFFFFF", line=GREEN_LINE, paras="".join(ps), anchor="ctr",
                         adj=7000, tIns=0.04, bIns=0.04))
        if i < len(PIPELINE) - 1:
            steps.append(shp(f"Step Arrow {i+1}", "downArrow", rx + 0.22, y + sh + 0.005,
                             0.13, 0.09, fill=GREEN, paras="<a:p/>"))
    bar = shp("Airgap Bar", "roundRect", rx, top + avail - bar_h, rw, bar_h, fill=GREEN, adj=20000,
              paras=para([run("Air-gapped  \u00b7  No internet  \u00b7  Runs on CPU for ISRO labs",
                              sz=950, b=True, color="FFFFFF")], algn="ctr"))
    return add(tree, head_l, *rows, head_r, *steps, bar, *ribbon(
        "Adaptation: EuroSAT warm-start (98.86% val), then BigEarthNet.txt multisensor adaptation. "
        "Benchmarks use the prescribed splits."))


def slide4(tree):
    tree = drop(tree, "Picture 4")
    cols = [("PROVEN TODAY \u2713", GREEN, GREEN_SOFT, PROVEN),
            ("HONEST CHALLENGES", AMBER, AMBER_SOFT, CHALLENGES),
            ("OUR STRATEGY \u2713", GREEN, GREEN_SOFT, STRATEGY)]
    xs = [0.48, 4.40, 8.55]
    ws = [3.70, 3.70, 4.30]
    top, heads = 1.30, 0.34
    ch, cgap = 1.08, 0.06
    out = []
    for ci, ((title, tcol, panelfill, items), x, w) in enumerate(zip(cols, xs, ws)):
        out.append(shp(f"Col {ci+1} Head", "roundRect", x, top, w, heads, fill=panelfill,
                       line=tcol, adj=25000,
                       paras=para([run(title, sz=1050, b=True, color=tcol)], algn="ctr")))
        for ri, (t, d) in enumerate(items):
            y = top + heads + 0.08 + ri * (ch + cgap)
            out.append(card(f"Col {ci+1} Card {ri+1}", x, y, w, ch, t, d, tcol,
                            panelfill if ci == 1 else "FFFFFF", title_color=tcol))
    for ri in range(4):
        y = top + heads + 0.08 + ri * (ch + cgap) + ch / 2 - 0.075
        out.append(shp(f"Link Arrow {ri+1}", "rightArrow", 8.24, y, 0.24, 0.15, fill=AMBER,
                       paras="<a:p/>"))
    return add(tree, *out, *ribbon(
        "RSVQA-LR 0.700 (n=9,491) \u00b7 LEVIR-CD F1 0.818 / IoU 0.692 (n=1,500) \u00b7 CDVQA 0.683 "
        "full test \u00b7 int8 CPU inference costs 0.16% accuracy"))


def slide5(tree):
    tree = drop(tree, "Picture 8")
    pw, gap = 4.06, 0.15
    top = 1.28
    hdr_h, sub_h = 0.42, 0.26
    body_h = 2.62
    out = []
    for i, (title, col, soft, line, subtitle, bullets) in enumerate(IMPACT):
        x = 0.48 + i * (pw + gap)
        out.append(shp(f"Impact Head {i+1}", "roundRect", x, top, pw, hdr_h, fill=col, adj=22000,
                       paras=para([run(title, sz=1050, b=True, color="FFFFFF")], algn="ctr")))
        out.append(shp(f"Impact Sub {i+1}", "rect", x, top + hdr_h, pw, sub_h,
                       paras=para([run(subtitle, sz=850, b=True, color=col)], algn="ctr"),
                       tIns=0.02, bIns=0))
        ps = []
        for b in bullets:
            ps.append(para([run("\u2022  ", sz=850, b=True, color=col), run(b, sz=850)], before=110))
        out.append(shp(f"Impact Body {i+1}", "roundRect", x, top + hdr_h + sub_h, pw, body_h,
                       fill=soft, line=line, paras="".join(ps), anchor="ctr", adj=6000))
    # PATH TO SCALE
    by = top + hdr_h + sub_h + body_h + 0.10
    bh = 6.42 - by
    out.append(shp("Path Band", "roundRect", 0.48, by, 12.37, bh, fill="F4F7FB", line="C4D0E6",
                   adj=6000, paras="<a:p/>"))
    out.append(shp("Path Heading", "rect", 0.58, by + 0.02, 12.17, 0.24,
                   paras=para([run("PATH TO SCALE \u2014 from proof of concept today to "
                                   "operational impact", sz=900, b=True, color="1B3C7B")]),
                   tIns=0, bIns=0))
    sw = (12.17 - 8 * 0.06) / 9
    for i, (t, d) in enumerate(PATH):
        x = 0.58 + i * (sw + 0.06)
        out.append(shp(f"Path Step {i+1}", "roundRect", x, by + 0.28, sw, bh - 0.34,
                       fill="FFFFFF", line="C4D0E6", adj=9000,
                       paras="".join([
                           para([run(f"{i+1}  ", sz=800, b=True, color="1B3C7B"), run(t, sz=800, b=True)],
                                algn="ctr"),
                           para([run(d, sz=700, color=MUTED)], algn="ctr", before=80),
                       ]), anchor="ctr", tIns=0.03, bIns=0.03, lIns=0.04, rIns=0.04))
    return add(tree, *out, *ribbon(
        "Cartosat-2S optical and RISAT SAR pairs, co-registered and pre-georeferenced, analysed on "
        "CPU inside an air-gapped lab \u2014 no cloud, no internet, no GIS specialist"))


def slide6(tree):
    for dead in ("TextBox 1", "TextBox 3", "TextBox 4"):
        tree = drop(tree, dead)
    cta = shp("CTA", "roundRect", 7.44, 1.28, 5.41, 1.32, fill=GREEN_SOFT, line=GREEN, adj=7000,
              paras="".join([
                  para([run("LIVE EVIDENCE BRIEF \u2014 START HERE", sz=900, b=True, color=GREEN)]),
                  para([run("anvesha-hub.vercel.app", sz=1400, b=True, color=BLUE, u=True,
                            hlink="rId900")], before=140),
                  para([run("Demo video \u00b7 execution traces \u00b7 benchmark provenance \u00b7 "
                            "downloadable reports", sz=880, color=MUTED)], before=110),
              ]), anchor="ctr")
    ps = [para([run("What the screenshot shows:", sz=880, b=True, color=GREEN)])]
    for t in ["input-mode selection (single / bi-temporal / optical\u2013SAR)",
              "a natural-language query with the routed task and agent trace",
              "confidence-scored output with an exportable GeoTIFF change mask"]:
        ps.append(para([run("\u2022  ", sz=850, color=GREEN), run(t, sz=850, color=MUTED)], before=110))
    cap = shp("Screenshot Caption", "rect", 0.48, 4.98, 6.30, 1.42, paras="".join(ps))
    ps = [para([run("DATA AND BENCHMARKS", sz=900, b=True, color=GREEN)])]
    for t in ["BigEarthNet.txt \u2014 SAR + multispectral with text (adaptation)",
              "VRSBench \u2014 captioning BLEU-4 0.000 (n=300), grounding IoU@0.5 0.126 "
              "(n=455), VQA 0.175 (n=200) \u2014 our weakest rows, published",
              "RSVQA-LR \u2014 VQA (0.700, n = 9,491)",
              "CDVQA \u2014 bi-temporal change VQA (0.683 full test)",
              "LEVIR-CD \u2014 change detection (F1 0.818, n = 1,500)",
              "EuroSAT \u2014 SceneEncoder warm-start (98.86% val)"]:
        ps.append(para([run("\u2022  ", sz=880, color=GREEN), run(t, sz=880)], before=110))
    ps.append(para([run("FOUNDATIONAL WORK", sz=900, b=True, color=GREEN)], before=300))
    for t in ["ResNet-18 (He et al.) \u00b7 Feature Pyramid Networks (Lin et al.)",
              "RemoteCLIP \u2014 remote-sensing vision\u2013language pretraining",
              "RSVQA (Lobry et al., IEEE TGRS 2020)"]:
        ps.append(para([run("\u2022  ", sz=880, color=GREEN), run(t, sz=880)], before=110))
    ps.append(para([run("Repository, model cards and decision log \u2014 linked from the evidence "
                        "brief above.", sz=860, i=True, color=MUTED)], before=300))
    refs = shp("References", "rect", 7.44, 2.76, 5.41, 3.64, paras="".join(ps))
    return add(tree, cta, cap, refs)


BUILDERS = {1: slide1, 2: slide2, 3: slide3, 4: slide4, 5: slide5, 6: slide6}

HYPERLINK_NS = ("http://schemas.openxmlformats.org/officeDocument/2006/"
                "relationships/hyperlink")


def prune_unused_images(parts: dict) -> tuple[dict, list[str]]:
    """Drop image relationships that no shape references, plus now-orphaned media."""
    for key in [k for k in parts if k.endswith(".rels")]:
        rels = parts[key].decode("utf-8")
        owner = key.replace("_rels/", "").replace(".rels", "")
        body = parts.get(owner, b"").decode("utf-8", "ignore")
        used = set(re.findall(r'r:(?:embed|link)="(rId\d+)"', body))
        keep, dropped = [], []
        for m in re.finditer(r"<Relationship\b[^>]*/>", rels):
            r = m.group(0)
            rid = re.search(r'Id="([^"]+)"', r).group(1)
            is_img = "/image" in r
            if is_img and rid not in used:
                dropped.append(rid)
            else:
                keep.append(r)
        if dropped:
            # NB: the root element is <Relationships>, which also starts with
            # "<Relationship" -- match the opening tag explicitly instead.
            root = re.match(r"(.*?<Relationships\b[^>]*>)", rels, re.S)
            if root:
                parts[key] = (root.group(1) + "".join(keep)
                              + "</Relationships>").encode("utf-8")
    referenced = set()
    for key in [k for k in parts if k.endswith(".rels")]:
        referenced |= set(re.findall(r'Target="([^"]*media/[^"]+)"',
                                     parts[key].decode("utf-8")))
    removed = []
    for n in list(parts):
        if n.startswith("ppt/media/") and f"../{n.split('ppt/')[1]}" not in referenced \
                and n.split("ppt/")[1] not in referenced:
            removed.append(n)
            del parts[n]
    return parts, removed


def main() -> int:
    assert SRC.exists(), f"template missing: {SRC}"
    with zipfile.ZipFile(SRC) as zin:
        parts = {n: zin.read(n) for n in zin.namelist()}

    for i, fn in BUILDERS.items():
        key = f"ppt/slides/slide{i}.xml"
        parts[key] = fn(parts[key].decode("utf-8")).encode("utf-8")

    rkey = "ppt/slides/_rels/slide6.xml.rels"
    rels = parts[rkey].decode("utf-8")
    if 'Id="rId900"' not in rels:
        rels = rels.replace(
            "</Relationships>",
            f'<Relationship Id="rId900" Type="{HYPERLINK_NS}" Target="{HUB_URL}"'
            ' TargetMode="External"/></Relationships>')
        parts[rkey] = rels.encode("utf-8")

    parts, removed = prune_unused_images(parts)

    DST.unlink(missing_ok=True)
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zout:
        for n, b in parts.items():
            zout.writestr(n, b)

    print(f"wrote {DST.name}  ({DST.stat().st_size/1e6:.2f} MB, {len(parts)} parts)")
    print("pruned media: " + (", ".join(sorted(x.split('/')[-1] for x in removed)) or "none"))
    print(f"native shapes added: {len(OUR_SHAPES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
