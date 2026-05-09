# ============================================================
#  app.py  —  Driver Fatigue Detection · Streamlit Dashboard
#  Webcam via streamlit-webrtc (works on Streamlit Cloud)
#  Run:  streamlit run app.py
# ============================================================

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
import plotly.graph_objects as go
import av
from datetime import datetime
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(
    page_title="FatigueGuard — Driver Monitor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }

.stApp {
    background: #080c10;
    background-image:
        linear-gradient(rgba(0,229,255,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,229,255,0.02) 1px, transparent 1px);
    background-size: 40px 40px;
}
[data-testid="stSidebar"] { background: #0d1520 !important; border-right: 1px solid #1a2e45; }
[data-testid="stSidebar"] * { color: #c8dce8 !important; }

.main-title {
    font-family: 'Orbitron', sans-serif; font-size: 2rem; font-weight: 900;
    color: #fff; text-shadow: 0 0 30px rgba(0,229,255,0.5);
    letter-spacing: 3px; margin: 0;
}
.main-title span { color: #00e5ff; }
.sub-title {
    font-family: 'Share Tech Mono', monospace; font-size: 0.75rem;
    color: #4a6070; letter-spacing: 4px; text-transform: uppercase; margin-top: 4px;
}
.metric-card {
    background: #0d1520; border: 1px solid #1a2e45; border-radius: 10px;
    padding: 18px 20px; position: relative;
}
.metric-card::before {
    content: ''; position: absolute; top: -1px; left: -1px;
    width: 18px; height: 18px;
    border-top: 2px solid #00e5ff; border-left: 2px solid #00e5ff;
    border-radius: 10px 0 0 0;
}
.metric-card.warn::before  { border-color: #ffd700; }
.metric-card.danger::before { border-color: #ff6b35; }
.metric-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.65rem;
    letter-spacing: 2px; color: #4a6070; text-transform: uppercase; margin-bottom: 6px;
}
.metric-value { font-family: 'Orbitron', sans-serif; font-size: 1.8rem; font-weight: 900; color: #00e5ff; line-height: 1; }
.metric-value.warn   { color: #ffd700; text-shadow: 0 0 15px rgba(255,215,0,0.4); }
.metric-value.danger { color: #ff6b35; text-shadow: 0 0 15px rgba(255,107,53,0.5); }
.metric-sub { font-size: 0.75rem; color: #4a6070; margin-top: 4px; }

.status-badge {
    display: inline-block; font-family: 'Orbitron', sans-serif; font-size: 1.1rem;
    font-weight: 700; padding: 10px 28px; border-radius: 6px;
    letter-spacing: 3px; text-align: center; width: 100%;
}
.badge-alert   { background: rgba(0,255,136,0.1);  border: 2px solid #00ff88; color: #00ff88; }
.badge-warning { background: rgba(255,215,0,0.1);  border: 2px solid #ffd700; color: #ffd700; }
.badge-danger  { background: rgba(255,107,53,0.15); border: 2px solid #ff6b35; color: #ff6b35;
                 animation: pulse-danger 1s ease-in-out infinite; }
@keyframes pulse-danger {
    0%,100% { box-shadow: 0 0 10px rgba(255,107,53,0.3); }
    50%      { box-shadow: 0 0 25px rgba(255,107,53,0.7); }
}
.gauge-wrap { margin: 10px 0; }
.gauge-label { font-family: 'Share Tech Mono', monospace; font-size: 0.65rem;
               letter-spacing: 2px; color: #4a6070; text-transform: uppercase; margin-bottom: 4px; }
.gauge-track { background: #1a2e45; border-radius: 4px; height: 8px; overflow: hidden; }
.gauge-fill  { height: 100%; border-radius: 4px; transition: width 0.4s ease; }

.sec-header {
    font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; letter-spacing: 3px;
    color: #00e5ff; text-transform: uppercase;
    border-bottom: 1px solid #1a2e45; padding-bottom: 6px; margin: 20px 0 14px;
}
hr { border-color: #1a2e45 !important; margin: 20px 0 !important; }
.stButton > button {
    background: #0d1520; border: 1px solid #00e5ff; color: #00e5ff;
    font-family: 'Share Tech Mono', monospace; letter-spacing: 2px;
    border-radius: 6px; transition: 0.2s;
}
.stButton > button:hover { background: rgba(0,229,255,0.1); box-shadow: 0 0 15px rgba(0,229,255,0.3); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "score_history":  deque(maxlen=200),
        "event_log":      [],
        "frame_count":    0,
        "session_start":  None,
        "last_result":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════
#  LOAD ENGINE
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model...")
def load_engine(model_path):
    from inference import FatigueEngine
    return FatigueEngine(model_path)


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def score_color(score):
    if score < 30: return "#00ff88"
    if score < 60: return "#ffd700"
    return "#ff6b35"

def level_class(level):
    return {"ALERT": "alert", "WARNING": "warning", "DANGER": "danger"}[level]

def gauge_html(label, value, max_val=1.0, color="#00e5ff"):
    pct = min(value / max_val * 100, 100)
    return f"""
    <div class="gauge-wrap">
        <div class="gauge-label">{label}</div>
        <div class="gauge-track">
            <div class="gauge-fill" style="width:{pct:.1f}%;background:{color};"></div>
        </div>
    </div>"""

def metric_html(label, value, sub="", cls=""):
    return f"""
    <div class="metric-card {cls}">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cls}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>"""


# ══════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════
def make_timeline_chart(scores):
    if not scores: scores = [0]
    y, x = list(scores), list(range(len(list(scores))))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", fill="tozeroy",
        fillcolor="rgba(0,229,255,0.06)",
        line=dict(color="#00e5ff", width=2),
        hovertemplate="Frame %{x}<br>Score: %{y:.1f}<extra></extra>"
    ))
    fig.add_hline(y=30, line_dash="dot", line_color="#ffd700",
                  annotation_text="WARNING", annotation_font_color="#ffd700", annotation_font_size=10)
    fig.add_hline(y=60, line_dash="dot", line_color="#ff6b35",
                  annotation_text="DANGER",  annotation_font_color="#ff6b35", annotation_font_size=10)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080c10",
        font=dict(family="Share Tech Mono", color="#4a6070", size=10),
        margin=dict(l=10,r=10,t=10,b=10),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#1a2e45", range=[0,105]),
        height=200, showlegend=False,
    )
    return fig

def make_gauge_chart(score):
    color = score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(font=dict(family="Orbitron", size=36, color=color)),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=1, tickcolor="#1a2e45",
                      tickfont=dict(color="#4a6070", size=9)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#0d1520", borderwidth=1, bordercolor="#1a2e45",
            steps=[
                dict(range=[0,30],  color="rgba(0,255,136,0.06)"),
                dict(range=[30,60], color="rgba(255,215,0,0.06)"),
                dict(range=[60,100],color="rgba(255,107,53,0.08)"),
            ],
            threshold=dict(line=dict(color=color, width=3), value=score)
        )
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Rajdhani"),
        margin=dict(l=20,r=20,t=10,b=10), height=200,
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  WEBRTC VIDEO PROCESSOR
# ══════════════════════════════════════════════════════════════
class FatigueProcessor(VideoProcessorBase):
    """
    Runs inside the WebRTC thread — processes every frame from the browser camera.
    Results are stored in shared state so the Streamlit UI can read them.
    """
    def __init__(self, model_path):
        from inference import FatigueEngine
        self.engine  = FatigueEngine(model_path)
        self.result  = {}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.result = self.engine.process_frame(img)

        # Push to session state for dashboard
        r = self.result
        if r.get("face_detected"):
            st.session_state.score_history.append(r["fatigue_score"])
            st.session_state.frame_count += 1
            st.session_state.event_log.append({
                "time":  datetime.now().strftime("%H:%M:%S"),
                "level": r["fatigue_level"],
                "score": round(r["fatigue_score"], 1),
            })
            if len(st.session_state.event_log) > 500:
                st.session_state.event_log = st.session_state.event_log[-500:]
            st.session_state.last_result = r

        annotated = r.get("frame", img)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def on_ended(self):
        self.engine.release()


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='font-family:"Orbitron",sans-serif;font-size:1rem;
                color:#00e5ff;letter-spacing:2px;margin-bottom:4px;'>
        FATIGUE<span style='color:#fff'>GUARD</span>
    </div>
    <div style='font-family:"Share Tech Mono",monospace;font-size:0.6rem;
                color:#4a6070;letter-spacing:3px;margin-bottom:20px;'>
        DRIVER MONITORING SYSTEM
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### ⚙️ Model Path")
    model_path = st.text_input("Fatigue Model (.pth)", value="saved_models/fatigue_phase2.pth")

    # Auto-fallback to phase1
    if not os.path.exists(model_path):
        fallback = "saved_models/fatigue_phase1.pth"
        if os.path.exists(fallback):
            model_path = fallback

    models_ready = os.path.exists(model_path)

    if models_ready:
        st.success(f"✓ Model found\n\n`{os.path.basename(model_path)}`")
    else:
        st.warning("⚠ Model not found.\n\nTrain first:\n`python train.py`")

    st.markdown("---")
    st.markdown("#### 🎛️ Thresholds")
    st.slider("EAR Threshold",          0.10, 0.40, 0.25, 0.01)
    st.slider("MAR Threshold",          0.30, 0.90, 0.60, 0.05)
    st.slider("Head Pitch (°)",         10,   40,   20,   1)
    st.slider("Temporal Window",        20,   120,  60,   10)

    st.markdown("---")
    if st.button("↺  Reset Session", use_container_width=True):
        st.session_state.score_history.clear()
        st.session_state.event_log.clear()
        st.session_state.frame_count = 0
        st.session_state.last_result = None
        st.rerun()


# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style='margin-bottom:28px;'>
    <div class='main-title'>FATIGUE<span>GUARD</span></div>
    <div class='sub-title'>// Real-Time Driver Fatigue Detection System</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab_webcam, tab_video, tab_analytics = st.tabs([
    "📷  Webcam — Real Time",
    "🎬  Video File",
    "📊  Session Analytics",
])


# ──────────────────────────────────────────────────────────────
# TAB 1: WEBCAM  (via streamlit-webrtc)
# ──────────────────────────────────────────────────────────────
with tab_webcam:
    col_feed, col_dash = st.columns([3, 2], gap="medium")

    with col_feed:
        st.markdown('<div class="sec-header">Live Feed</div>', unsafe_allow_html=True)

        if not models_ready:
            st.error("Model not found — check the path in the sidebar.")
        else:
            # STUN server so WebRTC works behind firewalls/NAT
            RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

            ctx = webrtc_streamer(
                key            = "fatigue-detection",
                video_processor_factory = lambda: FatigueProcessor(model_path),
                rtc_configuration       = RTC_CONFIG,
                media_stream_constraints= {"video": True, "audio": False},
                async_processing        = True,
            )

            if ctx.state.playing:
                st.session_state.session_start = st.session_state.session_start or datetime.now()
                st.info("🎥 Camera active — analysis running in real time.")
            else:
                st.markdown("""
                <div style='background:#0d1520;border:1px dashed #1a2e45;border-radius:10px;
                            height:200px;display:flex;flex-direction:column;align-items:center;
                            justify-content:center;color:#4a6070;font-family:"Share Tech Mono";
                            letter-spacing:2px;font-size:0.75rem;'>
                    <div style='font-size:2rem;margin-bottom:10px;'>📷</div>
                    CLICK START TO BEGIN
                </div>
                """, unsafe_allow_html=True)

    with col_dash:
        st.markdown('<div class="sec-header">Live Metrics</div>', unsafe_allow_html=True)

        r = st.session_state.last_result
        if r:
            score = r["fatigue_score"]
            level = r["fatigue_level"]
            lc    = level_class(level)

            st.markdown(
                f'<div class="status-badge badge-{lc}">{level}</div>',
                unsafe_allow_html=True
            )
            st.plotly_chart(make_gauge_chart(score), use_container_width=True, key="live_gauge")
            st.markdown(
                gauge_html("EAR — Eye Aspect Ratio", r["ear"], 0.4) +
                gauge_html("MAR — Mouth Aspect Ratio", r["mar"], 1.0, "#ffd700") +
                gauge_html("Head Pitch", abs(r["pitch"]), 40, "#ff6b35"),
                unsafe_allow_html=True
            )
            st.plotly_chart(
                make_timeline_chart(st.session_state.score_history),
                use_container_width=True, key="live_timeline"
            )
        else:
            st.markdown("""
            <div style='color:#4a6070;font-family:"Share Tech Mono";font-size:0.7rem;
                        letter-spacing:2px;text-align:center;padding:40px 0;'>
                WAITING FOR CAMERA FEED...
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# TAB 2: VIDEO FILE
# ──────────────────────────────────────────────────────────────
with tab_video:
    st.markdown('<div class="sec-header">Upload Video File</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload MP4 / AVI / MOV",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed"
    )

    if uploaded:
        col_v1, col_v2 = st.columns([3, 2], gap="medium")

        with col_v1:
            vframe_ph   = st.empty()
            vprogress   = st.progress(0, text="Ready")
            vstatus_ph  = st.empty()
            vstart_btn  = st.button("▶  ANALYZE VIDEO", use_container_width=True)

        with col_v2:
            st.markdown('<div class="sec-header">Analysis</div>', unsafe_allow_html=True)
            vgauge_ph    = st.empty()
            vtimeline_ph = st.empty()

        if vstart_btn:
            if not models_ready:
                st.error("Model not found — check the sidebar path.")
            else:
                suffix = "." + uploaded.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                try:
                    engine       = load_engine(model_path)
                    cap          = cv2.VideoCapture(tmp_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_scores = []
                    video_events = []
                    fnum         = 0

                    while True:
                        ret, frame = cap.read()
                        if not ret: break
                        fnum += 1

                        result = engine.process_frame(frame)
                        score  = result["fatigue_score"]
                        level  = result["fatigue_level"]
                        video_scores.append(score)

                        if result["face_detected"]:
                            video_events.append({
                                "frame": fnum,
                                "level": level,
                                "score": round(score, 1),
                            })

                        if fnum % 5 == 0:
                            disp = cv2.cvtColor(result["frame"], cv2.COLOR_BGR2RGB)
                            vframe_ph.image(disp, use_container_width=True)
                            vprogress.progress(
                                min(fnum / max(total_frames,1), 1.0),
                                text=f"Frame {fnum}/{total_frames}"
                            )
                            vstatus_ph.markdown(
                                f'<div class="status-badge badge-{level_class(level)}">{level}</div>',
                                unsafe_allow_html=True
                            )
                            vgauge_ph.plotly_chart(
                                make_gauge_chart(score),
                                use_container_width=True, key=f"vg_{fnum}"
                            )
                            vtimeline_ph.plotly_chart(
                                make_timeline_chart(deque(video_scores[-200:], maxlen=200)),
                                use_container_width=True, key=f"vtl_{fnum}"
                            )

                    cap.release()
                    os.unlink(tmp_path)
                    vprogress.progress(1.0, text="✓ Analysis complete!")

                    # ── Summary ───────────────────────────────────
                    if video_scores:
                        st.markdown("---")
                        st.markdown('<div class="sec-header">Video Summary</div>', unsafe_allow_html=True)
                        avg_s      = np.mean(video_scores)
                        max_s      = np.max(video_scores)
                        danger_pct = (np.array(video_scores) >= 60).mean() * 100
                        warn_pct   = ((np.array(video_scores) >= 30) & (np.array(video_scores) < 60)).mean() * 100

                        sm1, sm2, sm3, sm4 = st.columns(4)
                        with sm1: st.markdown(metric_html("AVG SCORE",      f"{avg_s:.0f}"), unsafe_allow_html=True)
                        with sm2: st.markdown(metric_html("PEAK SCORE",     f"{max_s:.0f}", cls="danger" if max_s>=60 else "warn"), unsafe_allow_html=True)
                        with sm3: st.markdown(metric_html("TIME IN DANGER", f"{danger_pct:.1f}%", cls="danger"), unsafe_allow_html=True)
                        with sm4: st.markdown(metric_html("TIME IN WARNING", f"{warn_pct:.1f}%",  cls="warn"),   unsafe_allow_html=True)

                    if video_events:
                        df = pd.DataFrame(video_events)
                        st.dataframe(df, use_container_width=True, height=250)
                        st.download_button(
                            "⬇  Download Event Log (CSV)",
                            data=df.to_csv(index=False),
                            file_name=f"fatigue_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                        )

                except Exception as e:
                    st.error(f"Error processing video: {e}")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)


# ──────────────────────────────────────────────────────────────
# TAB 3: SESSION ANALYTICS
# ──────────────────────────────────────────────────────────────
with tab_analytics:
    st.markdown('<div class="sec-header">Session Analytics</div>', unsafe_allow_html=True)

    log    = st.session_state.event_log
    scores = list(st.session_state.score_history)

    if not log:
        st.markdown("""
        <div style='background:#0d1520;border:1px dashed #1a2e45;border-radius:10px;
                    padding:40px;text-align:center;color:#4a6070;
                    font-family:"Share Tech Mono";letter-spacing:2px;font-size:0.75rem;'>
            NO SESSION DATA YET<br>
            <span style='font-size:0.65rem;opacity:0.5;'>Start the webcam or upload a video</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        df        = pd.DataFrame(log)
        avg_s     = np.mean(scores) if scores else 0
        max_s     = np.max(scores)  if scores else 0
        danger_ct = sum(1 for e in log if e["level"] == "DANGER")
        warn_ct   = sum(1 for e in log if e["level"] == "WARNING")
        dur_s     = int((datetime.now() - st.session_state.session_start).total_seconds()) \
                    if st.session_state.session_start else 0

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        with mc1: st.markdown(metric_html("DURATION",       f"{dur_s//60}:{dur_s%60:02d}", "min:sec"), unsafe_allow_html=True)
        with mc2: st.markdown(metric_html("FRAMES",         f"{st.session_state.frame_count}"), unsafe_allow_html=True)
        with mc3: st.markdown(metric_html("AVG FATIGUE",    f"{avg_s:.0f}", cls="warn" if avg_s>=30 else ""), unsafe_allow_html=True)
        with mc4: st.markdown(metric_html("DANGER EVENTS",  f"{danger_ct}", cls="danger"), unsafe_allow_html=True)
        with mc5: st.markdown(metric_html("WARNING EVENTS", f"{warn_ct}",   cls="warn"),   unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="sec-header">Score Timeline</div>', unsafe_allow_html=True)
        fig = make_timeline_chart(deque(scores, maxlen=len(scores)))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="sec-header">Level Distribution</div>', unsafe_allow_html=True)
        col_pie, col_tbl = st.columns([1, 2])
        with col_pie:
            level_counts = df["level"].value_counts()
            pie = go.Figure(go.Pie(
                labels=level_counts.index.tolist(),
                values=level_counts.values.tolist(),
                marker=dict(colors=["#00ff88","#ffd700","#ff6b35"]),
                hole=0.55,
                textfont=dict(family="Share Tech Mono", size=11),
            ))
            pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c8dce8"),
                margin=dict(l=0,r=0,t=10,b=10), height=220, showlegend=True,
                legend=dict(font=dict(family="Share Tech Mono", size=10))
            )
            st.plotly_chart(pie, use_container_width=True)

        with col_tbl:
            st.markdown("**Recent Events**")
            st.dataframe(df.tail(20)[::-1], use_container_width=True, height=220)

        st.markdown("---")
        st.download_button(
            "⬇  Export Full Session Log (CSV)",
            data=df.to_csv(index=False),
            file_name=f"session_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
