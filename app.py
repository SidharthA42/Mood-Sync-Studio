import time
from datetime import datetime
import numpy as np

import cv2
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from audio_transcribe import transcribe_audio
from fusion import load_learned_model, rule_based_fusion
from generator import generate_summary
from image_emotion import annotate_faces, detect_faces, predict_emotion, predict_emotion_frame
from text_sentiment import predict_sentiment
from utils import emotion_to_sentiment, label_order_image, label_order_text, load_image_model, load_text_model

st.set_page_config(page_title="MoodSync Studio", page_icon="MS", layout="wide")

EMOTION_C = {"happy": "#10b981", "sad": "#6366f1", "angry": "#ef4444", "fear": "#8b5cf6", "disgust": "#f97316", "surprise": "#eab308", "neutral": "#64748b"}
SENT_C = {"positive": "#10b981", "negative": "#ef4444", "neutral": "#64748b"}
RGB_C = {"happy": (16, 185, 129), "sad": (99, 102, 241), "angry": (239, 68, 68), "fear": (139, 92, 246), "disgust": (249, 115, 22), "surprise": (234, 179, 8), "neutral": (100, 116, 139)}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
*{box-sizing:border-box} html,body,[class*="css"]{font-family:Inter,sans-serif} #MainMenu,footer,header{visibility:hidden}
.stApp{background:#09111f;color:#f8fafc}.block-container{max-width:calc(100vw - 12px)!important;padding:.65rem .45rem 2rem!important}
.top{display:flex;justify-content:space-between;align-items:center;border:1px solid rgba(226,232,240,.14);background:#101a2b;border-radius:8px;padding:.95rem 1.1rem;margin-bottom:.85rem}
.brand{display:flex;gap:.9rem;align-items:center}.logo{width:52px;height:52px;border-radius:8px;background:linear-gradient(135deg,#14b8a6,#7c3aed);display:grid;place-items:center;font-size:1.35rem;font-weight:800}
.name{font-size:1.85rem;font-weight:800;letter-spacing:0}.sub{font:500 .78rem 'JetBrains Mono';color:#8ea0b8}.chips{display:flex;gap:.45rem;flex-wrap:wrap}.chip{border:1px solid rgba(226,232,240,.18);border-radius:999px;padding:.3rem .65rem;color:#cbd5e1;font-size:.74rem}
.hero,.panel,.card{border:1px solid rgba(226,232,240,.14);background:linear-gradient(180deg,#141f33,#101a2b);border-radius:8px}.hero{padding:1.15rem;margin:.75rem 0}
.hero h1{font-size:1.95rem;line-height:1.12;margin:.25rem 0 .45rem;letter-spacing:0}.hero p,.muted{color:#8ea0b8;line-height:1.6}.kicker{font:600 .72rem 'JetBrains Mono';color:#2dd4bf;text-transform:uppercase;letter-spacing:1px}
.panel{padding:1rem;margin-bottom:.9rem;min-height:100%}.pt{display:flex;justify-content:space-between;align-items:center;gap:.75rem;font-weight:800;margin-bottom:.75rem}.pill{font:600 .68rem 'JetBrains Mono';color:#8ea0b8;border:1px solid rgba(226,232,240,.18);border-radius:999px;padding:.22rem .55rem}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:1.15rem}.card{padding:1rem}.label{font:600 .68rem 'JetBrains Mono';color:#8ea0b8;text-transform:uppercase}.value{font-weight:800;font-size:1.5rem;margin:.3rem 0;color:#f8fafc}.note{color:#8ea0b8;font-size:.8rem;line-height:1.42}
.history-row{display:grid;grid-template-columns:repeat(4,1fr);gap:.85rem}.history-item{border:1px solid rgba(226,232,240,.14);border-radius:8px;background:rgba(255,255,255,.035);padding:.85rem}
.action-list{display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;margin-bottom:1rem}.action-card{border:1px solid rgba(226,232,240,.14);border-radius:8px;background:rgba(255,255,255,.035);padding:1rem}.action-card h3{font-size:1rem;margin:.1rem 0 .45rem}
.visual-grid{display:grid;grid-template-columns:1.05fr .95fr;gap:1rem;margin-top:1rem}.visual-stage{min-height:270px;border:1px solid rgba(226,232,240,.14);border-radius:8px;background:radial-gradient(circle at 24% 28%,rgba(45,212,191,.18),transparent 26%),radial-gradient(circle at 78% 72%,rgba(245,158,11,.12),transparent 30%),linear-gradient(135deg,#101a2b,#0b1220);position:relative;overflow:hidden;padding:1rem}.face-demo{position:absolute;left:9%;top:18%;width:42%;height:56%;border:2px solid #2dd4bf;border-radius:8px;background:linear-gradient(145deg,rgba(20,184,166,.12),rgba(15,23,42,.2));box-shadow:0 0 0 1px rgba(45,212,191,.12),0 18px 34px rgba(0,0,0,.18)}.face-demo:before{content:"LIVE MOOD  HAPPY";position:absolute;left:-2px;top:-30px;background:#14b8a6;color:white;font:800 .72rem 'JetBrains Mono';padding:.35rem .55rem;border-radius:6px}.face-demo:after{content:"";position:absolute;inset:12px;border:1px dashed rgba(153,246,228,.22);border-radius:6px}.sticker{position:absolute;border-radius:50%;background:#facc15;border:2px solid rgba(255,255,255,.9);box-shadow:0 10px 24px rgba(250,204,21,.22),0 0 0 8px rgba(250,204,21,.08);z-index:2}.sticker:before,.sticker:after{content:"";position:absolute;top:31%;width:8px;height:8px;border-radius:50%;background:#0f172a}.sticker:before{left:29%}.sticker:after{right:29%}.sticker .mouth{position:absolute;left:27%;top:45%;width:46%;height:25%;border-bottom:4px solid #0f172a;border-radius:0 0 999px 999px}.sticker .blush{position:absolute;left:18%;top:54%;width:10px;height:6px;border-radius:999px;background:#fb7185;box-shadow:38px 0 #fb7185;opacity:.78}.sticker.main{width:74px;height:74px;left:17%;top:29%}.sticker.small{width:48px;height:48px;right:19%;top:18%;transform:rotate(8deg)}.sticker.small:before,.sticker.small:after{width:6px;height:6px}.sticker.small .mouth{border-bottom-width:3px}.sticker.small .blush{width:7px;box-shadow:25px 0 #fb7185}.sticker.tiny:before,.sticker.tiny:after{width:5px;height:5px}.sticker.tiny .mouth{border-bottom-width:3px}.sticker.tiny .blush{width:6px;box-shadow:21px 0 #fb7185}.flow-line{position:absolute;left:51%;width:7%;height:2px;background:linear-gradient(90deg,#2dd4bf,#8b5cf6);opacity:.85}.flow-line.one{top:31%}.flow-line.two{top:54%}.flow-line.three{top:76%}.flow-dot{position:absolute;left:57.2%;width:9px;height:9px;border-radius:50%;background:#2dd4bf;box-shadow:0 0 0 7px rgba(45,212,191,.09)}.flow-dot.one{top:29.5%}.flow-dot.two{top:52.5%}.flow-dot.three{top:74.5%;background:#f59e0b}.signal-demo{position:absolute;right:5%;top:15%;width:37%;display:grid;gap:.65rem}.signal-card{border:1px solid rgba(226,232,240,.14);background:rgba(255,255,255,.045);border-radius:8px;padding:.8rem}.mini-bar{height:8px;border-radius:999px;background:#1e293b;overflow:hidden;margin-top:.45rem}.mini-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#14b8a6,#7c3aed)}.roadmap-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:1rem}.roadmap-card{border:1px solid rgba(226,232,240,.14);background:linear-gradient(180deg,rgba(20,31,51,.98),rgba(16,26,43,.96));border-radius:8px;padding:1rem;min-height:150px}
.button-row{max-width:360px;margin:.9rem auto 1.15rem}.download-space{height:.85rem}.chart-space{height:1.15rem}.input-button-space{height:.55rem}.stTabs [data-baseweb="tab-list"]{gap:1.1rem;border-bottom:1px solid rgba(226,232,240,.14);padding-bottom:.25rem}
.stTabs [data-baseweb="tab"]{border-radius:8px!important;color:#8ea0b8!important;font-weight:800!important;background:rgba(255,255,255,.025)!important;border:1px solid rgba(226,232,240,.1)!important;padding:.58rem 1.15rem!important}
.stTabs [aria-selected="true"]{background:rgba(45,212,191,.1)!important;color:#fff!important;border-color:rgba(45,212,191,.35)!important}.stTabs [data-baseweb="tab-panel"]{padding-top:.9rem!important}
.stButton>button{border:0!important;border-radius:8px!important;background:linear-gradient(135deg,#14b8a6,#7c3aed)!important;color:#fff!important;font-weight:800!important;min-height:2.65rem}
.stButton>button:disabled{opacity:.42!important}.stDownloadButton{margin-top:.8rem!important}.stFileUploader>div,.stTextArea textarea{background:rgba(255,255,255,.035)!important;border:1px solid rgba(226,232,240,.2)!important;border-radius:8px!important;color:#f8fafc!important}
.stCheckbox label,.stFileUploader label{color:#8ea0b8!important}.camera-note{border:1px solid rgba(45,212,191,.25);background:rgba(45,212,191,.06);border-radius:8px;padding:.75rem;color:#99f6e4;font-size:.84rem;line-height:1.5}
.sync-flow{min-height:235px;height:100%;display:grid;grid-template-columns:minmax(150px,1fr) 34px minmax(135px,.85fr) 34px minmax(130px,.8fr);gap:.5rem;align-items:center}.demo-face,.demo-text,.demo-result{border:1px solid rgba(226,232,240,.16);border-radius:8px;background:rgba(255,255,255,.04);padding:.78rem;min-height:158px}.demo-face{position:relative;border:2px solid #2dd4bf;background:linear-gradient(145deg,rgba(20,184,166,.13),rgba(15,23,42,.24));display:grid;place-items:center}.demo-face:before{content:"LIVE MOOD  HAPPY";position:absolute;left:-2px;top:-32px;background:#14b8a6;color:white;font:800 .68rem 'JetBrains Mono';padding:.36rem .58rem;border-radius:6px}.mood-frame{position:relative;width:122px;height:108px;border:1px dashed rgba(153,246,228,.4);border-radius:8px;display:grid;place-items:center}.mood-frame:before{content:"HAPPY";position:absolute;right:8px;top:8px;background:#0f766e;color:#ccfbf1;font:800 .58rem 'JetBrains Mono';padding:.22rem .38rem;border-radius:5px}.happy-face{position:relative;width:64px;height:64px;border-radius:50%;background:#facc15;border:3px solid rgba(255,255,255,.92);box-shadow:0 0 0 8px rgba(250,204,21,.11),0 14px 22px rgba(0,0,0,.25)}.happy-face:before,.happy-face:after{content:"";position:absolute;top:30%;width:7px;height:8px;border-radius:50%;background:#0f172a}.happy-face:before{left:28%}.happy-face:after{right:28%}.happy-mouth{position:absolute;left:27%;top:46%;width:46%;height:24%;border-bottom:4px solid #0f172a;border-radius:0 0 999px 999px}.happy-blush{position:absolute;left:17%;top:55%;width:9px;height:6px;border-radius:999px;background:#fb7185;box-shadow:34px 0 #fb7185}.demo-symbol{height:34px;width:34px;border-radius:50%;display:grid;place-items:center;background:#111c2e;border:1px solid rgba(45,212,191,.34);color:#99f6e4;font-weight:800;font-size:1.25rem}.sample-text{margin:.65rem 0 .75rem;color:#f8fafc;font-size:.92rem;line-height:1.42}.source-pills{display:flex;gap:.35rem;flex-wrap:wrap}.source-pills span,.result-choice span{font:700 .58rem 'JetBrains Mono';border:1px solid rgba(226,232,240,.14);border-radius:999px;padding:.25rem .4rem;color:#8ea0b8}.source-pills span:first-child{color:#bfdbfe;border-color:rgba(96,165,250,.35);background:rgba(96,165,250,.08)}.demo-result{display:grid;align-content:center}.result-choice{display:flex;gap:.35rem;flex-wrap:wrap;margin:.55rem 0}.result-choice span.active{color:#fed7aa;border-color:rgba(245,158,11,.55);background:rgba(245,158,11,.12)}
@media(max-width:900px){.grid3,.history-row,.action-list,.visual-grid,.roadmap-grid{grid-template-columns:1fr}.chips{display:none}.hero h1{font-size:1.45rem}.name{font-size:1.1rem}.visual-stage{min-height:470px}.face-demo{left:8%;top:14%;width:84%;height:170px}.signal-demo{left:8%;right:auto;top:57%;width:84%}.flow-line,.flow-dot{display:none}}
@media(max-width:900px){.sync-flow{grid-template-columns:1fr;min-height:auto}.demo-symbol{justify-self:center}.visual-stage{min-height:auto}}

/* ── webcam live styles ── */
.live-badge{display:inline-flex;align-items:center;gap:6px;background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.35);border-radius:999px;padding:.28rem .75rem;font:600 .7rem 'JetBrains Mono';color:#10b981;margin-bottom:.7rem}
.live-dot{width:7px;height:7px;border-radius:50%;background:#10b981;animation:blink 1.2s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
.emotion-box{border:1px solid rgba(226,232,240,.14);border-radius:8px;background:rgba(255,255,255,.03);padding:1rem;text-align:center;margin-top:.75rem}
.emotion-name{font-size:2rem;font-weight:800;letter-spacing:-.5px;margin:.2rem 0}
.emotion-conf{font:500 .75rem 'JetBrains Mono';color:#8ea0b8}
.cam-hint{border:1px solid rgba(45,212,191,.2);background:rgba(45,212,191,.05);border-radius:8px;padding:.7rem .9rem;color:#99f6e4;font-size:.8rem;line-height:1.55;margin-bottom:.8rem}
.snapshot-box{border:2px solid rgba(45,212,191,.4);border-radius:8px;padding:.75rem;margin-top:.75rem;background:rgba(45,212,191,.04)}
.snapshot-label{font:600 .68rem 'JetBrains Mono';color:#2dd4bf;text-transform:uppercase;letter-spacing:1px;margin-bottom:.5rem}
</style>
""", unsafe_allow_html=True)


def pct(v):
    return f"{v * 100:.0f}%"


def init_state():
    defaults = {
        "upload_result": None,
        "webcam_result": None,
        "captured_frame": None,
        "camera_on": False,
        "camera": None,
        "latest_clean_rgb": None,
        "cam_label": "Scanning",
        "cam_conf": 0.0,
        "cam_probs": {},
        "cam_faces": 0,
        "cam_last_infer": 0.0,
        "cam_history": [],
        "smile_cascade": None,
        "analysis_history": [],
        "latest_source": None,
        "webcam_running": False,
        "webcam_cam_version": 0,
        "webcam_snapshot": None,
        "webcam_snapshot_annot": None,
        "webcam_snapshot_label": None,
        "webcam_snapshot_conf": None,
        "last_processed_frame": None,
        "live_emotion": "Scanning",
        "live_confidence": 0.0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def chart_probs(probs, colors, title, horizontal=False, height=210, key=None, percent=True):
    labels = list(probs.keys())
    vals = [probs[k] * 100 for k in labels] if percent else [probs[k] for k in labels]
    text_vals = [f"{v:.0f}%" for v in vals] if percent else [f"{v:.0f}" for v in vals]
    marker = dict(color=[colors.get(k.lower(), "#2dd4bf") for k in labels], line_width=0)
    trace = go.Bar(x=vals, y=labels, orientation="h", marker=marker, text=text_vals, textposition="outside") if horizontal else go.Bar(x=labels, y=vals, marker=marker, text=text_vals, textposition="outside")
    fig = go.Figure(trace)
    fig.update_layout(title=dict(text=title, font=dict(size=12, color="#8ea0b8"), x=0), height=height, margin=dict(l=8, r=8, t=34, b=12), showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#8ea0b8", family="Inter"))
    st.plotly_chart(fig, use_container_width=True, key=key)


def sentiment_donut(probs, key=None):
    labels = ["positive", "negative", "neutral"]
    values = [probs.get(label, 0) * 100 for label in labels]
    fig = go.Figure(
        go.Pie(
            labels=[label.title() for label in labels],
            values=values,
            hole=0.62,
            sort=False,
            marker=dict(colors=[SENT_C[label] for label in labels], line=dict(color="#09111f", width=3)),
            textinfo="percent",
            textfont=dict(color="#f8fafc", size=13),
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        )
    )
    top_label = max(probs, key=probs.get)
    fig.update_layout(
        title=dict(text="Text sentiment probabilities", font=dict(size=12, color="#8ea0b8"), x=0),
        height=260,
        margin=dict(l=8, r=8, t=34, b=10),
        showlegend=True,
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center", font=dict(color="#8ea0b8")),
        annotations=[dict(text=f"{top_label.title()}<br>{pct(probs[top_label])}", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#f8fafc"))],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8ea0b8", family="Inter"),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def token_chart(attention, key=None):
    clean = [(a.replace("\u0120", "").replace("Ġ", "").replace("Ä ", "").strip(), w) for a, w in attention or [] if a.strip()]
    if not clean:
        return
    toks, weights = zip(*clean)
    m = max(weights) or 1
    vals = [w / m for w in weights]
    fig = go.Figure(go.Bar(x=list(toks), y=vals, marker=dict(color=vals, colorscale=[[0, "#193049"], [.55, "#2dd4bf"], [1, "#7c3aed"]]), text=[f"{v:.2f}" for v in vals], textposition="outside"))
    fig.update_layout(title=dict(text="Token attention distribution", font=dict(size=12, color="#8ea0b8"), x=0), height=260, margin=dict(l=8, r=8, t=34, b=12), showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#8ea0b8", family="Inter"))
    st.plotly_chart(fig, use_container_width=True, key=key)


def cards(img=None, txt=None, fus=None):
    blocks = []
    if img:
        c = EMOTION_C.get(img["label"].lower(), "#2dd4bf")
        blocks.append(f'<div class="card"><div class="label">Facial mood</div><div class="value" style="color:{c}">{img["label"].title()}</div><div class="note">{pct(img["confidence"])} confidence | {len(img["faces"])} face(s)</div></div>')
    if txt:
        c = SENT_C.get(txt["label"].lower(), "#2dd4bf")
        blocks.append(f'<div class="card"><div class="label">Text sentiment</div><div class="value" style="color:{c}">{txt["label"].title()}</div><div class="note">{pct(txt["confidence"])} confidence</div></div>')
    if fus:
        c = "#f59e0b" if fus["result"] == "MISMATCH" else "#10b981"
        v = "Mismatch" if fus["result"] == "MISMATCH" else "Aligned"
        blocks.append(f'<div class="card"><div class="label">Fusion verdict</div><div class="value" style="color:{c}">{v}</div><div class="note">{fus["method"]}</div></div>')
    if blocks:
        st.markdown(f'<div class="grid3">{"".join(blocks)}</div>', unsafe_allow_html=True)


def transcribe_audio_widget(audio_file, source):
    try:
        audio_bytes = audio_file.getvalue() if hasattr(audio_file, "getvalue") else audio_file.read()
        return transcribe_audio(audio_bytes).strip()
    except Exception:
        st.error(f"{source} could not be transcribed. Try clearer audio or type text instead.")
        return ""


def text_audio_box(key):
    reset_key = f"{key}_audio_reset"
    st.session_state.setdefault(reset_key, 0)
    audio_version = st.session_state[reset_key]
    rec_key = f"{key}_rec_{audio_version}"
    aud_key = f"{key}_aud_{audio_version}"

    st.markdown('<div class="panel"><div class="pt">Text and audio <span class="pill">text or audio required</span></div>', unsafe_allow_html=True)
    text = st.text_area("Text to analyse", key=f"{key}_text", height=128, placeholder="Type what the person said, or record/upload audio below.")
    st.markdown('<div class="input-button-space"></div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    transcript = ""
    audio_supplied = False
    with a1:
        rec = st.audio_input("Record audio", key=rec_key)
        if rec:
            audio_supplied = True
            with st.spinner("Transcribing recording..."):
                transcript = transcribe_audio_widget(rec, "Recording")
    with a2:
        aud = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"], key=aud_key)
        if aud:
            audio_supplied = True
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio_widget(aud, "Audio file")
    if audio_supplied:
        if st.button("Clear audio", key=f"{key}_clear_audio", use_container_width=True):
            st.session_state[reset_key] = audio_version + 1
            st.rerun()
    if transcript:
        st.text_area("Audio transcript", value=transcript, height=70, key=f"{key}_transcript_{audio_version}")
    elif audio_supplied and not text.strip():
        st.warning("Audio was added, but no transcript is available yet. Use clearer audio or add text before analysing.")
    elif not text.strip():
        st.caption("Text is required only when no audio recording or upload is available.")
    st.markdown("</div>", unsafe_allow_html=True)
    return text.strip(), transcript.strip(), audio_supplied


def analyse_all(image, text, transcript, heatmap, token_attention, learned):
    full_text = (text + ("\n" + transcript if transcript else "")).strip()
    img_label, img_conf, img_probs, hm = predict_emotion(image, heatmap)
    annotated, faces = annotate_faces(image, img_label, img_conf)
    txt_label, txt_conf, txt_probs, attention = predict_sentiment(full_text, token_attention)
    if learned:
        try:
            import torch
            model = load_learned_model()
            v = torch.tensor([float(img_probs.get(l, 0)) for l in label_order_image], dtype=torch.float32)
            t = torch.tensor([float(txt_probs.get(l, 0)) for l in label_order_text], dtype=torch.float32)
            with torch.no_grad():
                pred = model.fc(torch.cat([v, t]).unsqueeze(0)).argmax(dim=1).item()
            verdict, method = ("MISMATCH" if pred == 1 else "ALIGNED"), "Learned fusion"
        except Exception:
            verdict, _ = rule_based_fusion(img_label, txt_label)
            method = "Rule-based fallback"
    else:
        verdict, _ = rule_based_fusion(img_label, txt_label)
        method = "Rule-based fusion"
    img = {"label": img_label, "confidence": img_conf, "probs": img_probs, "heatmap": hm, "annotated": annotated, "faces": faces}
    txt = {"label": txt_label, "confidence": txt_conf, "probs": txt_probs, "attention": attention}
    fus = {"result": verdict, "method": method, "summary": generate_summary(img_label, img_conf, txt_label, txt_conf, verdict)}
    return img, txt, fus


def record_history(source, result):
    img, txt, fus = result
    item = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "date": datetime.now().strftime("%d %b %Y"),
        "source": source,
        "emotion": img["label"],
        "emotion_conf": img["confidence"],
        "sentiment": txt["label"],
        "sentiment_conf": txt["confidence"],
        "fusion": fus["result"],
        "method": fus["method"],
        "summary": fus["summary"],
    }
    history = st.session_state.analysis_history
    history.insert(0, item)
    st.session_state.analysis_history = history[:12]
    st.session_state.latest_source = source


def latest_result():
    if st.session_state.latest_source == "Webcam" and st.session_state.webcam_result:
        return "Webcam", st.session_state.webcam_result
    if st.session_state.latest_source == "Upload" and st.session_state.upload_result:
        return "Upload", st.session_state.upload_result
    if st.session_state.webcam_result:
        return "Webcam", st.session_state.webcam_result
    if st.session_state.upload_result:
        return "Upload", st.session_state.upload_result
    return None, None


def mismatch_score(img, txt, fus):
    visual_sentiment = emotion_to_sentiment.get(img["label"], "neutral")
    gap = abs(img["confidence"] - txt["confidence"])
    base = 35 if fus["result"] == "MISMATCH" else 8
    if visual_sentiment != txt["label"]:
        base += 30
    base += int(max(img["confidence"], txt["confidence"]) * 25)
    base += int(gap * 10)
    return min(100, base)


def action_plan(img, txt, fus):
    score = mismatch_score(img, txt, fus)
    if fus["result"] == "MISMATCH":
        headline = "Signals are not aligned"
        steps = [
            "Ask one calm follow-up question before making a conclusion.",
            "Treat the facial signal as a cue, not proof.",
            "Re-check with a second image or webcam capture if confidence is low.",
            "Look for repeated mismatch across multiple samples before escalating.",
        ]
    else:
        headline = "Signals are broadly aligned"
        steps = [
            "Use the result as a supportive signal, not a final diagnosis.",
            "Capture another sample if the face is poorly lit or partially visible.",
            "Compare confidence values before trusting subtle emotional differences.",
            "Save this run in history for trend comparison.",
        ]
    prompts = [
        "Can you tell me a little more about what you mean?",
        "Is that how you feel, or are you trying to keep it neutral?",
        "Would you like to pause and come back to this?",
    ]
    return score, headline, steps, prompts


def report_text(source, img, txt, fus):
    score, headline, steps, prompts = action_plan(img, txt, fus)
    return f"""# MoodSync Analysis Report

Source: {source}
Generated: {datetime.now().strftime("%d %b %Y, %H:%M:%S")}

## Results
- Facial mood: {img["label"].title()} ({pct(img["confidence"])} confidence)
- Text sentiment: {txt["label"].title()} ({pct(txt["confidence"])} confidence)
- Fusion verdict: {fus["result"].title()}
- Mismatch score: {score}/100

## Interpretation
{headline}

## Recommended next steps
{chr(10).join(f"- {step}" for step in steps)}

## Follow-up prompts
{chr(10).join(f"- {prompt}" for prompt in prompts)}
"""


def show_results(img, txt, fus, show_heatmap, show_tokens, prefix):
    cards(img, txt, fus)
    st.markdown('<div class="chart-space"></div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        chart_probs(img["probs"], EMOTION_C, "Facial emotion probabilities", height=240, key=f"{prefix}_facial_probs")
    with right:
        sentiment_donut(txt["probs"], key=f"{prefix}_sentiment_donut")

    st.markdown('<div class="chart-space"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        if show_heatmap and img["heatmap"] is not None:
            st.image(img["heatmap"], caption="Full-image attention heatmap", use_container_width=True)
        else:
            st.markdown('<div class="panel"><div class="pt">Attention heatmap <span class="pill">off</span></div><div class="muted">Enable Attention heatmap before analysing to show the full-image contribution map here.</div></div>', unsafe_allow_html=True)
    with c2:
        if show_tokens:
            token_chart(txt["attention"], key=f"{prefix}_token_attention")
        else:
            st.markdown('<div class="panel"><div class="pt">Token attention <span class="pill">off</span></div><div class="muted">Enable Token attention before analysing to show the distribution here.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-space"></div>', unsafe_allow_html=True)
    cmp_col, sum_col = st.columns(2, gap="large")
    with cmp_col:
        chart_probs({"visual negative": sum(v for k, v in img["probs"].items() if emotion_to_sentiment.get(k) == "negative"), "visual positive": sum(v for k, v in img["probs"].items() if emotion_to_sentiment.get(k) == "positive"), "text negative": txt["probs"].get("negative", 0), "text positive": txt["probs"].get("positive", 0)}, {"visual negative": "#ef4444", "visual positive": "#10b981", "text negative": "#ef4444", "text positive": "#10b981"}, "Cross-modal comparison", height=240, key=f"{prefix}_cross_modal")
    with sum_col:
        st.markdown(f'<div class="panel"><div class="pt">Summary <span class="pill">{fus["method"]}</span></div><div class="muted">{fus["summary"]}</div></div>', unsafe_allow_html=True)


# ── Boot ──────────────────────────────────────────────────────────────────────
init_state()

st.markdown('<div class="top"><div class="brand"><div class="logo">M</div><div><div class="name">MoodSync Studio</div><div class="sub">facial mood + language fusion</div></div></div><div class="chips"><span class="chip">Camera Input</span><span class="chip">Face Box</span><span class="chip">Heatmap</span><span class="chip">Token Attention</span><span class="chip">Fusion</span></div></div>', unsafe_allow_html=True)

with st.spinner("Loading models..."):
    load_image_model()
    load_text_model()

overview, upload_tab, webcam_tab, history_tab, plan_tab, future_tab = st.tabs(
    ["Overview", "Upload Image Analysis", "Camera Capture Analysis", "Analysis History", "Action Plan", "Future Roadmap"]
)

# ── Overview (unchanged) ─────────────────────────────────────────────────────
with overview:
    st.markdown("""
    <div class="hero">
      <div class="kicker">App description</div>
      <h1>Detect facial mood, compare it with typed or spoken language, and identify emotional mismatch.</h1>
      <p>MoodSync uses a facial emotion model for images or webcam captures, a sentiment model for typed text or audio transcript, and a fusion layer to decide whether the visual and verbal signals are aligned or mismatched.</p>
    </div>
    <div class="grid3">
      <div class="card"><div class="label">Input</div><div class="value">Face</div><div class="note">Upload an image or take a photo with your camera. Emotion is displayed immediately on the photo.</div></div>
      <div class="card"><div class="label">Language</div><div class="value">Text / Audio</div><div class="note">Type text, record audio, or upload audio. Text is required only when no audio is provided.</div></div>
      <div class="card"><div class="label">Output</div><div class="value">Fusion</div><div class="note">Mood, sentiment, heatmap, token attention, action plan, and history tracking.</div></div>
    </div>
    <div class="visual-grid">
      <div class="visual-stage">
        <div class="sync-flow">
          <div class="demo-face">
            <div class="mood-frame">
              <div class="happy-face"><span class="happy-mouth"></span><span class="happy-blush"></span></div>
            </div>
          </div>
          <div class="demo-symbol">+</div>
          <div class="demo-text">
            <div class="label">Language input</div>
            <div class="sample-text">"I am fine, but today feels heavy."</div>
            <div class="source-pills"><span>Text</span><span>Audio transcript</span></div>
          </div>
          <div class="demo-symbol">=</div>
          <div class="demo-result">
            <div class="label">Fusion result</div>
            <div class="value" style="color:#f59e0b;font-size:1.25rem">Mismatch</div>
            <div class="result-choice"><span>Match</span><span class="active">Mismatch</span></div>
            <div class="note">Happy face + negative language</div>
          </div>
        </div>
      </div>
      <div class="panel">
        <div class="pt">How to read it <span class="pill">example</span></div>
        <div class="muted">
          MoodSync highlights the detected face, estimates emotion, reads the text or transcript, and then compares both modalities. The heatmap explains visual focus while token attention shows which words influenced the sentiment model.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Upload tab (unchanged) ───────────────────────────────────────────────────
with upload_tab:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="panel"><div class="pt">Image upload <span class="pill">left section</span></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"], key="upload_image")
        image = Image.open(uploaded).convert("RGB") if uploaded else None
        if image:
            st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        text, transcript, _audio_supplied = text_audio_box("upload")
    c1, c2, c3 = st.columns(3)
    with c1:
        heatmap = st.checkbox("Attention heatmap", value=True, key="upload_heatmap")
    with c2:
        tokens = st.checkbox("Token attention distribution", value=True, key="upload_tokens")
    with c3:
        learned = st.checkbox("Learned fusion", value=False, key="upload_learned")
    st.markdown('<div class="button-row">', unsafe_allow_html=True)
    language_ready = bool(text or transcript)
    if st.button("Analyse", disabled=image is None or not language_ready, use_container_width=True, key="upload_analyse"):
        with st.spinner("Analysing image, text, and fusion..."):
            st.session_state.upload_result = analyse_all(image, text, transcript, heatmap, tokens, learned)
            record_history("Upload", st.session_state.upload_result)
    st.markdown("</div>", unsafe_allow_html=True)
    if st.session_state.upload_result:
        show_results(*st.session_state.upload_result, heatmap, tokens, "upload")
    else:
        st.info("Upload an image and provide either text or a transcribed audio recording/upload.")

# ── Camera Capture tab (with Start/Stop/Take Photo buttons) ───────────────────
with webcam_tab:
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="panel"><div class="pt">Live Webcam <span class="pill">start/stop/take photo</span></div>', unsafe_allow_html=True)

        btn_c1, btn_c2, btn_c3 = st.columns(3, gap="small")
        with btn_c1:
            start_clicked = st.button("▶  Start Camera", use_container_width=True,
                                      key="webcam_start",
                                      disabled=st.session_state.webcam_running)
        with btn_c2:
            stop_clicked  = st.button("■  Stop Camera",  use_container_width=True,
                                      key="webcam_stop",
                                      disabled=not st.session_state.webcam_running)
        with btn_c3:
            photo_clicked = st.button("📸  Take Photo",  use_container_width=True,
                                      key="webcam_take_photo",
                                      disabled=not st.session_state.webcam_running)

        if start_clicked:
            st.session_state.webcam_running = True
            st.session_state.webcam_snapshot = None
            st.session_state.webcam_snapshot_annot = None
            st.session_state.captured_frame = None
            st.rerun()

        if stop_clicked:
            st.session_state.webcam_running = False
            st.session_state.webcam_snapshot = None
            st.session_state.webcam_snapshot_annot = None
            st.session_state.captured_frame = None
            st.rerun()

        st.markdown('<div class="input-button-space"></div>', unsafe_allow_html=True)

        # Show camera input only if running
        if st.session_state.webcam_running:
            st.markdown("""
            <div class="cam-hint">
              Click the camera shutter button below to take a photo. The face will be detected and emotion displayed immediately.
              Then click <strong>Take Photo</strong> to save this frame for analysis (or it auto-saves when you take the picture).
            </div>
            """, unsafe_allow_html=True)

            # Camera widget
            camera_photo = st.camera_input("Take a picture", key="camera_input_widget")

            # When a new photo is taken, process it immediately
            if camera_photo is not None:
                # Convert to PIL Image
                img = Image.open(camera_photo).convert("RGB")
                # Run emotion prediction
                label, conf, probs, _ = predict_emotion(img, heatmap=False)
                # Draw face box and label
                annotated, faces = annotate_faces(img, label, conf)
                # Store in session state (overwrite previous)
                st.session_state.captured_frame = img
                st.session_state.webcam_snapshot = img
                st.session_state.webcam_snapshot_annot = annotated
                st.session_state.webcam_snapshot_label = label
                st.session_state.webcam_snapshot_conf = conf
                st.session_state.cam_label = label
                st.session_state.cam_conf = conf
                st.session_state.cam_probs = probs
                # Display annotated image
                st.image(annotated, caption=f"Mood: {label.upper()} ({pct(conf)} confidence)", use_container_width=True)
            else:
                st.info("Press the shutter button above to take a photo.")

            # Take Photo button: if a photo has already been taken, it just saves it (already done)
            # We'll also add a manual save button in case the user wants to re-save the same photo.
            if photo_clicked and st.session_state.captured_frame is not None:
                st.success("Photo saved! You can now add text/audio and click Analyse.")
        else:
            st.markdown("""
            <div style="text-align:center;padding:2.8rem 1rem;color:#8ea0b8;">
              <div style="font-size:2.4rem;opacity:.2;margin-bottom:.8rem">◈</div>
              <div style="font-size:.88rem;font-weight:600">Click <strong style="color:#f8fafc">▶ Start Camera</strong> to open the camera</div>
              <div style="font-size:.76rem;opacity:.6;margin-top:.4rem">Then take a photo and the emotion will appear instantly</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        text, transcript, _audio_supplied = text_audio_box("webcam")

    c1, c2, c3 = st.columns(3)
    with c1:
        heatmap = st.checkbox("Attention heatmap", value=True, key="webcam_heatmap")
    with c2:
        tokens = st.checkbox("Token attention distribution", value=True, key="webcam_tokens")
    with c3:
        learned = st.checkbox("Learned fusion", value=False, key="webcam_learned")

    st.markdown('<div class="button-row">', unsafe_allow_html=True)
    language_ready = bool(text or transcript)
    ready = st.session_state.captured_frame is not None and language_ready
    if st.button("Analyse", disabled=not ready, use_container_width=True, key="webcam_analyse"):
        with st.spinner("Analysing captured image, text, and fusion..."):
            st.session_state.webcam_result = analyse_all(
                st.session_state.captured_frame, text, transcript, heatmap, tokens, learned
            )
            record_history("Webcam", st.session_state.webcam_result)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.webcam_result:
        show_results(*st.session_state.webcam_result, heatmap, tokens, "webcam")
    else:
        if st.session_state.captured_frame is None:
            st.info("Start camera, take a photo, then provide text or audio and click Analyse.")
        elif not language_ready:
            st.info("Provide text or audio before clicking Analyse.")

# ── History tab (unchanged) ──────────────────────────────────────────────────
with history_tab:
    history = st.session_state.analysis_history
    st.markdown('<div class="hero"><div class="kicker">Session memory</div><h1>Track recent analyses and compare emotional patterns.</h1><p>This tab automatically stores your latest upload and webcam runs during the current session, making it easier to compare repeated mismatch or alignment patterns.</p></div>', unsafe_allow_html=True)
    if not history:
        st.info("No analyses saved yet. Run an upload or webcam analysis and it will appear here automatically.")
    else:
        mood_counts = {}
        fusion_counts = {"ALIGNED": 0, "MISMATCH": 0}
        for item in history:
            mood_counts[item["emotion"]] = mood_counts.get(item["emotion"], 0) + 1
            fusion_counts[item["fusion"]] = fusion_counts.get(item["fusion"], 0) + 1
        c1, c2 = st.columns(2, gap="large")
        with c1:
            chart_probs(mood_counts, EMOTION_C, "Mood frequency in this session", height=260, key="history_mood_frequency", percent=False)
        with c2:
            chart_probs(fusion_counts, {"aligned": "#10b981", "mismatch": "#f59e0b"}, "Fusion verdict frequency", height=260, key="history_fusion_frequency", percent=False)

        rows = []
        for item in history[:8]:
            fusion_c = "#f59e0b" if item["fusion"] == "MISMATCH" else "#10b981"
            emotion_c = EMOTION_C.get(item["emotion"].lower(), "#2dd4bf")
            rows.append(
                f'<div class="history-item"><div class="label">{item["source"]} | {item["time"]}</div>'
                f'<div class="value" style="color:{emotion_c};font-size:1.15rem">{item["emotion"].title()}</div>'
                f'<div class="note">Text: {item["sentiment"].title()} | <span style="color:{fusion_c}">{item["fusion"].title()}</span></div></div>'
            )
        st.markdown(f'<div class="history-row">{"".join(rows)}</div>', unsafe_allow_html=True)
        if st.button("Clear History", key="clear_history"):
            st.session_state.analysis_history = []
            st.rerun()

# ── Action Plan tab (unchanged) ──────────────────────────────────────────────
with plan_tab:
    source, result = latest_result()
    st.markdown('<div class="hero"><div class="kicker">Decision support</div><h1>Turn the latest analysis into a practical response plan.</h1><p>This tab adds a mismatch score, suggested next steps, follow-up prompts, and a downloadable report for the most recent run.</p></div>', unsafe_allow_html=True)
    if not result:
        st.info("Run an upload or webcam analysis first. The latest result will be converted into an action plan here.")
    else:
        img, txt, fus = result
        score, headline, steps, prompts = action_plan(img, txt, fus)
        score_color = "#ef4444" if score >= 70 else "#f59e0b" if score >= 45 else "#10b981"
        st.markdown(
            f'<div class="grid3">'
            f'<div class="card"><div class="label">Latest source</div><div class="value">{source}</div><div class="note">Most recent completed analysis</div></div>'
            f'<div class="card"><div class="label">Mismatch score</div><div class="value" style="color:{score_color}">{score}/100</div><div class="note">{headline}</div></div>'
            f'<div class="card"><div class="label">Confidence pair</div><div class="value">{pct(img["confidence"])} / {pct(txt["confidence"])}</div><div class="note">Facial mood / text sentiment</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        items = "".join(f"<li>{step}</li>" for step in steps)
        qs = "".join(f"<li>{prompt}</li>" for prompt in prompts)
        st.markdown(
            f'<div class="action-list" style="margin-top:1rem">'
            f'<div class="action-card"><h3>Recommended next steps</h3><div class="muted"><ul>{items}</ul></div></div>'
            f'<div class="action-card"><h3>Follow-up prompts</h3><div class="muted"><ul>{qs}</ul></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="download-space"></div>', unsafe_allow_html=True)
        st.download_button(
            "Download Report",
            report_text(source, img, txt, fus),
            file_name=f"moodsync_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            key="download_action_report",
            use_container_width=True,
        )

# ── Future tab (unchanged) ───────────────────────────────────────────────────
with future_tab:
    st.markdown(
        """
        <div class="hero">
          <div class="kicker">Future implementations</div>
          <h1>Roadmap for making MoodSync more capable, explainable, and production-ready.</h1>
          <p>These are planned directions that would make the app more useful for real workflows while keeping the interface clean and decision-focused.</p>
        </div>
        <div class="roadmap-grid">
          <div class="roadmap-card"><div class="label">Near term</div><div class="value" style="font-size:1.2rem">Timeline Export</div><div class="note">Export full session history as CSV/PDF with thumbnails, scores, and summaries.</div></div>
          <div class="roadmap-card"><div class="label">Near term</div><div class="value" style="font-size:1.2rem">Multi-face Analysis</div><div class="note">Track multiple faces in one image or webcam frame with separate mood labels.</div></div>
          <div class="roadmap-card"><div class="label">Modeling</div><div class="value" style="font-size:1.2rem">Temporal Smoothing</div><div class="note">Use rolling emotion sequences to detect mood shifts rather than single-frame predictions.</div></div>
          <div class="roadmap-card"><div class="label">Explainability</div><div class="value" style="font-size:1.2rem">Confidence Warnings</div><div class="note">Flag low light, poor face angle, blur, and weak text evidence before showing verdicts.</div></div>
          <div class="roadmap-card"><div class="label">Audio</div><div class="value" style="font-size:1.2rem">Voice Emotion</div><div class="note">Add pitch, pause, and speaking-rate emotion cues alongside transcript sentiment.</div></div>
          <div class="roadmap-card"><div class="label">Deployment</div><div class="value" style="font-size:1.2rem">Local Privacy Mode</div><div class="note">Keep all analysis on-device with no external API calls and clear data retention controls.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="muted" style="border-top:1px solid rgba(226,232,240,.14);padding-top:1rem;text-align:center;font:600 .72rem JetBrains Mono;margin-top:1rem">MoodSync Studio | ViT emotion | RoBERTa sentiment | Whisper audio | Camera capture + instant mood preview</div>', unsafe_allow_html=True)
