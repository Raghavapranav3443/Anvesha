"""LEGACY: Streamlit UI — retained for reference; superseded by the React
console served by `satquery.server.main` (uvicorn, port 8000).

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from satquery.agent import get_controller, classify_task
from satquery.config import CONFIG, EUROSAT_CLASSES
from satquery.io_utils import load_image, rgb_composite, validate_inputs, describe_configuration
from satquery.registry import build_default_registry

st.set_page_config(page_title="SatQuery AI", page_icon="🛰️", layout="wide")

# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def controller():
    return get_controller()


def np_to_png(arr: np.ndarray) -> bytes:
    a = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if a.ndim == 2:
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format="PNG")
        return buf.getvalue()
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    buf = io.BytesIO()
    Image.fromarray(a[:, :, :3]).save(buf, format="PNG")
    return buf.getvalue()


def show_image_bytes(arr: np.ndarray, caption: str):
    st.image(np_to_png(arr), caption=caption, use_container_width=True)


# --------------------------------------------------------------------------- #
st.title("🛰️ SatQuery AI")
st.caption("Agentic vision-language assistant for multimodal remote-sensing "
           "image analysis through text queries.")

with st.sidebar:
    st.header("1 · Input images")
    input_mode = st.radio("Input configuration",
                          ["Single image", "Bi-temporal pair (change)",
                           "Optical + SAR pair"], index=0)

    uploads = []
    if input_mode == "Single image":
        f = st.file_uploader("Upload GeoTIFF/TIFF/PNG/JPEG",
                             type=["tif", "tiff", "png", "jpg", "jpeg"])
        uploads = [f] if f else []
    else:
        c1, c2 = st.columns(2)
        with c1:
            f1 = st.file_uploader("Image A / Optical",
                                  type=["tif", "tiff", "png", "jpg", "jpeg"], key="a")
        with c2:
            f2 = st.file_uploader("Image B / SAR",
                                  type=["tif", "tiff", "png", "jpg", "jpeg"], key="b")
        uploads = [f for f in (f1, f2) if f]

    # bundled demo samples
    demo_dir = CONFIG.samples_dir
    demos_available = sorted(p.name for p in demo_dir.glob("*")) if demo_dir.exists() else []
    if demos_available:
        st.divider()
        st.subheader("Demo samples")
        pick = st.multiselect("Load bundled samples", demos_available)
        demo_imgs = [load_image(demo_dir / name) for name in pick]
    else:
        demo_imgs = []

    st.divider()
    st.header("2 · Query")
    examples = [
        "Describe the land-cover and major objects visible in this image.",
        "Highlight the water body referred to in the query.",
        "What changed between these two dates, and where did the change occur?",
        "Has the built-up area increased, decreased, or remained unchanged?",
        "Use the optical and SAR images together to identify built-up and water-covered regions.",
    ]
    query = st.text_area("Natural-language query",
                         examples[0], height=90)
    for ex in examples:
        if st.button(ex[:44] + "…", key=ex[:20], use_container_width=True):
            query = ex
            st.session_state["query_box"] = ex

    task_override = st.selectbox(
        "Force task (optional — leave on Auto for agentic routing)",
        ["auto"] + sorted(build_default_registry().keys()))

    run_btn = st.button("Run analysis", type="primary", use_container_width=True)

# --------------------------------------------------------------------------- #
images = []
for up in uploads:
    up.seek(0)
    tmp = CONFIG.runs_dir / "_uploads" / up.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(up.read())
    images.append(load_image(tmp))
images += demo_imgs

if not images:
    st.info("⬅ Upload an image or load demo samples to begin. Supported: "
            "GeoTIFF/TIFF (geospatial), PNG/JPEG (benchmark datasets).")
    st.stop()

try:
    cfg = validate_inputs(images)
except Exception as e:
    st.error(f"Input validation failed: {e}")
    st.stop()

st.success(describe_configuration(images, cfg))
cols = st.columns(len(images))
for col, img in zip(cols, images):
    with col:
        s = img.summary()
        show_image_bytes(rgb_composite(img),
                         f"{s['file']} · {s['modality']} · {s['bands']} bands"
                         f"{'' if s['crs'] is None else ' · ' + s['crs']}")

if not run_btn:
    st.stop()

with st.spinner("Agent planning and executing specialist tools..."):
    try:
        result = controller().run(
            images, query,
            task_override=None if task_override == "auto" else task_override)
    except Exception as e:
        st.error(f"Execution failed: {e}")
        st.stop()

intent = next((s for s in result.trace if s.get("name") == "classify_task"), {})
exec_steps = [s for s in result.trace if str(s.get("name", "")).startswith("execute")]

st.header(f"Result — task: `{result.selected_task}`")
cA, cB = st.columns([3, 1])
with cA:
    st.markdown(f"### {result.answer}")
with cB:
    st.metric("Confidence", f"{result.confidence:.2f}")

if visuals := result.visuals:
    vkeys = list(visuals.keys())
    vcols = st.columns(min(3, len(vkeys)))
    for col, key in zip(vcols, vkeys):
        arr = visuals[key]
        with col:
            if arr.ndim == 3:
                show_image_bytes(arr, key)
            elif arr.ndim == 2 and key == "prob_map":
                show_image_bytes(arr, "change probability map")
            elif arr.ndim == 2:
                show_image_bytes(arr, key)

with st.expander("📋 Evidence details"):
    st.json({k: v for k, v in result.outputs.items()
             if not isinstance(v, np.ndarray)}, expanded=True)

with st.expander("🔍 Execution trace (auditable)"):
    st.json(result.trace)

dl1, dl2 = st.columns(2)
dl1.download_button("⬇ Download report (Markdown)",
                    Path(result.report_paths["markdown"]).read_text(encoding="utf-8"),
                    file_name=f"{result.run_id}.md")
dl2.download_button("⬇ Download report (JSON)",
                    Path(result.report_paths["json"]).read_text(encoding="utf-8"),
                    file_name=f"{result.run_id}.json")
