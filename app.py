import os
import base64
import pandas as pd
import altair as alt
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
load_dotenv(override=True)
from tavily import TavilyClient
import gc
import fitz
from modules.ingestion import process_document_pipeline
from modules.retrieval import populate_vector_store, retrieve_context, get_chroma_client
from modules.agent import (
    generate_agent_response, 
    generate_chained_report, 
    classify_query_intent, 
    generate_tabular_response
)

# ==========================================
# 1. ENVIRONMENT & PAGE CONFIGURATION
# ==========================================

# Wide layout to support the multi-chart dashboard cards
st.set_page_config(
    page_title="DocMate AI",
    page_icon="📖",
    layout="wide"
)

# Initialize persistent session states
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# ==========================================
# 2. HERO SECTION & UI DESIGN SYSTEM
# ==========================================

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None

# Load Logo
logo_base64 = get_base64_image("logo.png") 
if logo_base64:
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="max-width: 320px; margin-bottom: 10px; filter: drop-shadow(0px 0px 15px rgba(0, 150, 255, 0.4));">'
else:
    logo_html = '<h1 style="margin:0; padding:0; color: #FFFFFF; text-shadow: 0px 0px 15px rgba(0, 150, 255, 0.8);">📖 DocMate AI</h1>'

# Dynamically generate the Matrix string to fill the background, with each letter wrapped in a span for animation
matrix_word = "DOCMATEAI"
matrix_spans = "".join([f"<span>{c}</span>" for c in matrix_word * 200])
matrix_html = f'<div class="jp-matrix">{matrix_spans}</div>'

# CSS styling for the Matrix background and overall UI design system, including the glassmorphism effect for chat messages and metrics, and the dynamic dashboard cards. 
# The CSS also includes responsive design considerations and interactive elements for the charts and graph exports.
st.markdown(f"""
<style>
/* 0. GLOBAL CSS VARIABLES */
:root {{
    --system-font: -apple-system, BlinkMacSystemFont, "San Francisco", "Helvetica Neue", Helvetica, Arial, sans-serif;
}}

/* 1. NUKE ALL STREAMLIT ROOT BACKGROUNDS */
html, body, #root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"] {{
    background: transparent !important;
    background-color: transparent !important;
}}

/* 2. THE MATRIX BACKGROUND CSS */
.jp-matrix {{
    position: fixed;
    top: 0; left: 0;
    width: 100vw;
    height: 100vh;
    background-color: #05050a;
    overflow: hidden;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(40px, 1fr));
    grid-auto-rows: 40px;
    font-size: 32px;
    color: rgba(0, 150, 255, 0.4);
    font-family: "Courier New", Courier, monospace;
    justify-content: center;
    align-content: center;
    z-index: -1; /* Pushes the matrix behind the UI */
    pointer-events: none; /* Allows you to click through the background */
}}

.jp-matrix > span {{
    text-align: center;
    text-shadow: 0 0 5px rgba(0, 150, 255, 0.5);
    user-select: none;
    transition: color 0.5s, text-shadow 0.5s;
    line-height: 1;
}}

/* Matrix Animation Delays */
.jp-matrix > span:nth-child(19n + 2) {{ animation: smooth-pulse 3.5s ease-in-out infinite 0.2s; }}
.jp-matrix > span:nth-child(29n + 1) {{ animation: smooth-pulse 4.1s ease-in-out infinite 0.7s; }}
.jp-matrix > span:nth-child(11n) {{ color: rgba(100, 200, 255, 0.7); animation: smooth-pulse 2.9s ease-in-out infinite 1.1s; }}
.jp-matrix > span:nth-child(37n + 10) {{ animation: smooth-pulse 5.3s ease-in-out infinite 1.5s; }}
.jp-matrix > span:nth-child(41n + 1) {{ animation: smooth-pulse 3.9s ease-in-out infinite 0.4s; }}
.jp-matrix > span:nth-child(17n + 9) {{ animation: smooth-pulse 2.8s ease-in-out infinite 0.9s; }}
.jp-matrix > span:nth-child(23n + 18) {{ animation: smooth-pulse 4.3s ease-in-out infinite 1.3s; }}
.jp-matrix > span:nth-child(31n + 4) {{ animation: smooth-pulse 5.6s ease-in-out infinite 0.1s; }}
.jp-matrix > span:nth-child(43n + 20) {{ animation: smooth-pulse 3.6s ease-in-out infinite 1.8s; }}
.jp-matrix > span:nth-child(13n + 6) {{ animation: smooth-pulse 3.2s ease-in-out infinite 1.2s; }}
.jp-matrix > span:nth-child(53n + 5) {{ animation: smooth-pulse 4.9s ease-in-out infinite 0.5s; }}
.jp-matrix > span:nth-child(47n + 15) {{ animation: smooth-pulse 5.9s ease-in-out infinite 1s; }}

@keyframes smooth-pulse {{
    0%, 100% {{
        color: rgba(0, 150, 255, 0.4);
        text-shadow: 0 0 5px rgba(0, 150, 255, 0.5);
    }}
    30% {{
        color: rgba(100, 200, 255, 1);
        text-shadow: 0 0 10px rgba(100, 200, 255, 1), 0 0 15px rgba(100, 200, 255, 1);
    }}
    50% {{
        color: rgba(255, 105, 180, 1);
        text-shadow: 0 0 10px rgba(255, 105, 180, 1), 0 0 15px rgba(255, 105, 180, 1);
    }}
    70% {{
        color: #ffffff;
        text-shadow: 0 0 10px #fff, 0 0 15px #fff, 0 0 20px #fff;
    }}
}}

/* 3. SAFE FONT & WHITE TEXT TARGETING */
html, body, p, h1, h2, h3, h4, h5, h6, li, a, label, .stMarkdown {{
    font-family: var(--system-font) !important;
    color: #F8FAFC !important;
}}

.material-symbols-rounded, .icon, [class*="icon"], [data-testid="stSidebarCollapseButton"] *, [data-testid="collapsedControl"] * {{
    font-family: "Material Symbols Rounded" !important;
    font-weight: normal !important;
    font-style: normal !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    color: #F8FAFC !important;
}}

/* 4. UI GLASSMORPHISM */
[data-testid="stSidebar"] {{ 
    background-color: rgba(5, 5, 10, 0.7) !important; 
    backdrop-filter: blur(15px); 
    border-right: 1px solid rgba(0, 150, 255, 0.2); 
}}

[data-testid="stStatusWidget"], [data-testid="stExpander"] details {{
    background-color: rgba(5, 5, 10, 0.8) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(0, 150, 255, 0.2) !important;
    border-radius: 10px !important;
}}

.stSlider div, .stSelectbox div, .stNumberInput div {{ color: white !important; }}
div[data-baseweb="select"] > div, input {{ 
    background-color: rgba(255, 255, 255, 0.05) !important; 
    border: 1px solid rgba(0, 150, 255, 0.3) !important; 
    color: white !important; 
}}
.stButton > button {{ 
    background: rgba(0, 150, 255, 0.1) !important; 
    border: 1px solid rgba(0, 150, 255, 0.3) !important; 
    backdrop-filter: blur(10px); 
    border-radius: 50px; 
    transition: all 0.3s ease; 
    color: white !important;
}}
.stButton > button:hover {{ 
    background: rgba(255, 105, 180, 0.2) !important; 
    border-color: rgba(255, 105, 180, 0.6) !important; 
}}

/* 5. SIDEBAR & BOTTOM BAR ALIGNMENT */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {{
    text-align: center !important; width: 100% !important;
}}
[data-testid="stSidebar"] .stButton {{ display: flex; justify-content: center; width: 100%; }}
[data-testid="stSidebar"] .stButton > button {{ width: 100% !important; max-width: 280px !important; }}
[data-testid="stSidebar"] [data-testid="stAlert"] {{
    display: flex !important; flex-direction: column !important; align-items: center !important; 
    justify-content: center !important; text-align: center !important; padding-top: 1.5rem !important;
}}
[data-testid="stSidebar"] [data-testid="stUploadedFile"] {{
    display: flex !important; flex-direction: column !important; align-items: center !important;
    justify-content: center !important; text-align: center !important; padding-bottom: 15px !important;
}}
[data-testid="stSidebar"] [data-testid="stUploadedFile"] button {{
    position: relative !important; margin-top: 5px !important; right: auto !important; top: auto !important;
}}
[data-testid="stFileUploader"] label {{ display: flex; justify-content: center; text-align: center; width: 100%; }}
[data-testid="stFileUploaderDropzone"] {{ display: flex; flex-direction: column; align-items: center; text-align: center; }}

[data-testid="stBottom"] {{ background: transparent !important; }}
[data-testid="stBottom"] > div {{ background: transparent !important; }}
[data-testid="stChatInput"] {{
    background-color: rgba(5, 5, 10, 0.8) !important;
    backdrop-filter: blur(15px) !important;
    border: 1px solid rgba(0, 150, 255, 0.3) !important;
    border-radius: 50px !important;
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.4) !important;
}}
[data-testid="stChatInput"] textarea {{ color: #FFFFFF !important; }}

/* --- 6. CHAT MESSAGE GLASSMORPHISM --- */
/* Wrap all chat messages in a dark, blurred glass card */
[data-testid="stChatMessage"] {{
    background-color: rgba(5, 5, 10, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(0, 150, 255, 0.2) !important;
    border-radius: 15px !important;
    padding: 1rem 1.5rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.6) !important;
}}

/* Ensure all text inside the chat bubbles is bright white with a drop shadow */
[data-testid="stChatMessage"] p, 
[data-testid="stChatMessage"] li, 
[data-testid="stChatMessage"] h1, 
[data-testid="stChatMessage"] h2, 
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] div {{
    color: #F8FAFC !important;
    text-shadow: 0px 2px 5px rgba(0, 0, 0, 0.9) !important;
}}

/* Style the avatar icons so they match the aesthetic */
[data-testid="stChatMessageAvatar"] {{
    background-color: rgba(0, 150, 255, 0.1) !important;
    border: 1px solid rgba(0, 150, 255, 0.3) !important;
    border-radius: 8px !important;
}}
/* --- 7. METRICS GLASSMORPHISM & CENTERING --- */
/* Wrap the metric boxes in dark frosted glass */
[data-testid="stMetric"] {{
    background-color: rgba(5, 5, 10, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(0, 150, 255, 0.2) !important;
    border-radius: 15px !important;
    padding: 15px !important;
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.6) !important;
    
    /* Force the container to align contents to the center */
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}}

/* Force the tiny upper label to center */
[data-testid="stMetricLabel"] {{
    display: flex !important;
    justify-content: center !important;
    text-align: center !important;
    width: 100% !important;
}}

/* Force the big number value to center */
[data-testid="stMetricValue"] {{
    display: flex !important;
    justify-content: center !important;
    text-align: center !important;
    width: 100% !important;
    text-shadow: 0px 2px 5px rgba(0, 0, 0, 0.9), 0px 0px 15px rgba(0, 150, 255, 0.3) !important;
}}
</style>

{matrix_html}

<div style="
    text-align: center; 
    padding: 60px 20px; 
    margin: 0 auto 20px auto; 
    max-width: 850px;
    background: radial-gradient(circle, rgba(5, 5, 10, 0.95) 0%, rgba(5, 5, 10, 0.7) 40%, transparent 55%);
    border-radius: 50px;
">
    {logo_html}
    <p style="
        margin:0; 
        padding-top: 15px; 
        font-size: 1.15em; 
        font-weight: 500; 
        color: #F8FAFC; 
        text-shadow: 0px 4px 15px rgba(0, 0, 0, 1), 0px 0px 20px rgba(0, 150, 255, 0.5);
    ">Multi-document intelligence engine — query, synthesize, explore</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.processed_files:
    cols = st.columns(4)
    cols[0].metric("Active Documents", len(st.session_state.processed_files))
    cols[1].metric("LLM Status", "Online")
    cols[2].metric("Modules", "Loaded")
    cols[3].metric("Web API Status", "Ready")
st.divider()

# ==========================================
# 3. SIDEBAR: DOCUMENT MANAGEMENT
# ==========================================
if "active_persona" not in st.session_state:
    st.session_state.active_persona = "Analyst"

# SIDEBAR: DOCUMENT MANAGEMENT & COGNITIVE PERSONA SELECTION ALONG WITH PROGRESS TRACKING FOR DOCUMENT PROCESSING. 
with st.sidebar:
    
    st.header("Document Library")
        
    uploaded_files = st.file_uploader(
        "Upload workspace PDFs & CSVs", 
        type=["pdf", "csv"], 
        accept_multiple_files=True
    )
    
    if st.button("Clear Document Library"):
        try:
            get_chroma_client().reset()
        except Exception as e:
            print(f"Database reset log: {e}")
            
        st.cache_resource.clear() 
        st.cache_data.clear() 
        st.session_state.processed_files.clear()
        st.session_state.conversation_memory = []
        
        st.toast("✅ Library wiped clean! Ready for new documents.")
        st.rerun()

    st.info("**Pro Tip:** Our tool is for making things easier, but your effort counts all the way!", icon="💡")
    st.divider()
    st.header("🧠 Cognitive Persona")
    
    if "active_persona" not in st.session_state:
        st.session_state.active_persona = "Analyst"
        
    st.session_state.active_persona = st.selectbox(
        "Select Active Lens",
        ["Analyst", "Researcher", "BI Expert", "Teacher"],
        index=["Analyst", "Researcher", "BI Expert", "Teacher"].index(st.session_state.active_persona),
        help="Changes how the AI interprets and structures the document data."
    )
    st.info("**NOTE:** If graphs on dashboard appear distorted, please adjust your browser zoom level.", icon="🔎",width="stretch")
    # Uploaded files are processed in the main loop below to ensure the progress bar is visible and updates correctly, and to manage memory by processing one batch at a time.
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                
                # 1. Setup UI Progress Tracking
                st.write(f"Processing {uploaded_file.name}...")
                progress_bar = st.progress(0)
                
                temp_path = f"./data/{uploaded_file.name}"
                os.makedirs("./data", exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # 2. Process the file based on its type (PDF or CSV)
                if uploaded_file.name.endswith(".pdf"):
                    with fitz.open(temp_path) as doc:
                        total_pages = len(doc)
                    
                    def update_progress(pages_processed):
                        progress_val = min(pages_processed / total_pages, 1.0)
                        progress_bar.progress(progress_val)
                    
                    for batch_chunks in process_document_pipeline(temp_path, progress_callback=update_progress):
                        if batch_chunks:
                            populate_vector_store(batch_chunks)
                        del batch_chunks
                        gc.collect()
                        
                elif uploaded_file.name.endswith(".csv"):
                    from modules.database import load_csv_to_sqlite
                    load_csv_to_sqlite(temp_path, uploaded_file.name)
                    progress_bar.progress(1.0)
                # --------------------------------------------
                
                # 3. Cleanup
                progress_bar.empty()
                st.session_state.processed_files.add(uploaded_file.name)
        
# ==========================================
# 4. HELPER FUNCTION: RENDER DYNAMIC DASHBOARD
# ==========================================
def render_dashboard_component(payload: dict, message_idx: int = 0):
    """
    Renders an enterprise-grade multi-chart dashboard using Altair.
    Safely melts wide tabular data into long format to map dynamically reasoned charts.
    """
    st.markdown(f"## 📊 {payload.get('dashboard_title', 'Executive Dashboard')}")
    st.write(payload.get("executive_summary", ""))

    # 1. Render Top-Level KPI Metric Cards
    metrics = payload.get("metrics", [])
    if metrics:
        cols = st.columns(len(metrics))
        for idx, m in enumerate(metrics):
            cols[idx].metric(
                label=m.get("label", "Metric"),
                value=f"{m.get('value', 'N/A')} {m.get('unit', '')}",
                delta=m.get('trend', None)
            )
    st.divider()

    # 2. Render Reasoned Multi-Chart Grid with Independent Datasets
    charts = payload.get("charts", [])
    if charts:
        st.markdown("### 📈 Interactive Visual Analytics")
        chart_cols = st.columns(len(charts))
        
        for idx, chart in enumerate(charts):
            with chart_cols[idx]:
                with st.container(border=True):
                    title = chart.get("chart_title", "Data Trend Analysis")
                    reasoning = chart.get("reasoning", "")
                    c_type = chart.get("chart_type", "bar").lower()
                    x_col = chart.get("x_axis", "")
                    y_cols = chart.get("y_axes", [])
                    
                    # Extract the dataset SPECIFIC to this chart
                    dataset_raw = chart.get("dataset", [])
                    df = pd.DataFrame(dataset_raw)

                    st.markdown(f"**{title}**")
                    st.caption(f"*{reasoning}*")

                    # Generate One-Click Export for this specific chart's data
                    if not df.empty:
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Export Data (CSV)",
                            data=csv_data,
                            file_name=f"chart_data_{message_idx}_{idx}.csv",
                            mime="text/csv",
                            key=f"export_btn_{message_idx}_{idx}"
                        )

                    # Safety check
                    if not df.empty and x_col in df.columns and all(y in df.columns for y in y_cols):
                        melted_df = df.melt(id_vars=[x_col], value_vars=y_cols, var_name="Metric", value_name="Value")
                        
                        if c_type == "line":
                            alt_chart = alt.Chart(melted_df).mark_line(point=True).encode(
                                x=alt.X(f"{x_col}:N", title=x_col),
                                y=alt.Y("Value:Q", title="Value"),
                                color="Metric:N",
                                tooltip=[x_col, "Metric", "Value"]
                            ).interactive()
                        elif c_type == "scatter":
                            alt_chart = alt.Chart(melted_df).mark_circle(size=60).encode(
                                x=alt.X(f"{x_col}:N", title=x_col),
                                y=alt.Y("Value:Q", title="Value"),
                                color="Metric:N",
                                tooltip=[x_col, "Metric", "Value"]
                            ).interactive()
                        else:
                            alt_chart = alt.Chart(melted_df).mark_bar().encode(
                                x=alt.X(f"{x_col}:N", title=x_col),
                                y=alt.Y("Value:Q", title="Value"),
                                color="Metric:N",
                                tooltip=[x_col, "Metric", "Value"]
                            ).interactive()

                        st.altair_chart(alt_chart, use_container_width=True)
                    else:
                        st.error(f"⚠️ **Chart Mismatch**\n\nAI generated Axes (X: `{x_col}`, Y: `{y_cols}`).\n\nDataset only contains columns: `{list(df.columns) if not df.empty else 'No Data'}`")

# ==========================================
# 5. HISTORICAL MEMORY RENDERING
# ==========================================
for idx, message in enumerate(st.session_state.conversation_memory): 
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Check for visual payloads in the message and render accordingly
        if "visual_payload" in message:
            h_intent = message["visual_payload"]["intent"]
            h_payload = message["visual_payload"]["payload"]
            # Depending on the intent, render the appropriate component with the provided payload
            if h_intent == "VISUAL_GRAPH":
                with st.expander("📊 View Cross-Document Entity Graph", expanded=False):
                    
                    mermaid_html = f"""
                    <style>
                        .toolbar {{ display: flex; justify-content: flex-end; gap: 10px; margin-bottom: 10px; }}
                        .action-btn {{ 
                            background: rgba(0, 150, 255, 0.1); border: 1px solid rgba(0, 150, 255, 0.4); 
                            color: #F8FAFC; padding: 6px 12px; border-radius: 8px; cursor: pointer; 
                            font-family: sans-serif; font-size: 12px; transition: all 0.3s ease;
                        }}
                        .action-btn:hover {{ background: rgba(0, 150, 255, 0.3); border-color: rgba(0, 150, 255, 0.8); box-shadow: 0px 0px 10px rgba(0, 150, 255, 0.5); }}
                    </style>
                    
                    <div class="toolbar">
                        <button class="action-btn" onclick="toggleFullScreen()">⛶ Fullscreen</button>
                        <button class="action-btn" onclick="downloadPNG()">📥 Download PNG</button>
                    </div>
                    
                    <div class="mermaid" id="graph-container" style="width: 100%; display: flex; justify-content: center; background-color: transparent;">
                        {h_payload}
                    </div>
                    
                    <script type="module">
                        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                        mermaid.initialize({{ startOnLoad: true, theme: 'dark', flowchart: {{ useMaxWidth: true }} }});
                        
                        window.toggleFullScreen = function() {{
                            const elem = document.documentElement;
                            if (!document.fullscreenElement) {{
                                elem.requestFullscreen().catch(err => console.log(err));
                            }} else {{
                                document.exitFullscreen();
                            }}
                        }};

                        window.downloadPNG = function() {{
                            const svg = document.querySelector('.mermaid svg');
                            if (!svg) return;
                            
                            // 1. Extract absolute dimensions from the viewBox
                            let width = 800;
                            let height = 600;
                            const viewBox = svg.getAttribute('viewBox');
                            
                            if (viewBox) {{
                                const parts = viewBox.split(' ');
                                if (parts.length === 4) {{
                                    width = parseFloat(parts[2]);
                                    height = parseFloat(parts[3]);
                                }}
                            }}

                            // 2. Clone the SVG and force absolute dimensions so it doesn't compress
                            const clone = svg.cloneNode(true);
                            clone.setAttribute('width', width);
                            clone.setAttribute('height', height);

                            const svgData = new XMLSerializer().serializeToString(clone);
                            const canvas = document.createElement("canvas");
                            const ctx = canvas.getContext("2d");
                            const img = new Image();
                            
                            img.onload = function() {{
                                // 3. Set canvas dimensions with comfortable padding
                                const padding = 80;
                                canvas.width = width + padding;
                                canvas.height = height + padding;
                                
                                // Fill the dark background
                                ctx.fillStyle = "#05050a"; 
                                ctx.fillRect(0, 0, canvas.width, canvas.height);
                                
                                // Draw the SVG un-squished at its original aspect ratio
                                ctx.drawImage(img, padding / 2, padding / 2, width, height);
                                
                                // 4. Trigger download
                                const a = document.createElement("a");
                                a.download = "DocMate_Flowchart.png";
                                a.href = canvas.toDataURL("image/png");
                                a.click();
                            }};
                            img.src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svgData)));
                        }};
                    </script>
                    """
                    components.html(mermaid_html, height=600, scrolling=True)
                    
            elif h_intent == "METRICS_DASHBOARD":
                with st.container():
                    render_dashboard_component(h_payload, message_idx=idx) 
            
            elif h_intent == "TABULAR_SQL":
                # Render historical DataFrame
                if "sql_data" in message["visual_payload"]:
                    df = pd.DataFrame(message["visual_payload"]["sql_data"])
                    st.dataframe(df, use_container_width=True)
                    
                with st.expander("🗄️ View Generated SQL Plan", expanded=False):
                    st.code(h_payload, language="sql")

# ==========================================
# 6. LIVE CHAT ROUTING & ORCHESTRATION
# ==========================================
# The chat input is the main orchestrator that routes user queries through different execution paths based on command parsing and intent classification. 
# It also manages the conversation memory and ensures that the appropriate components are rendered based on the AI's response payloads.
if user_query := st.chat_input("Ask a question, or use commands: /dash, /flow, /table, /find, /help..."):
    
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.conversation_memory.append({"role": "user", "content": user_query})
    
    # --- 1. INSTANT UTILITY COMMANDS (Bypass AI) ---
    # The /help command is designed to provide users with immediate guidance on how to use the various commands without needing to wait for an AI response. It directly renders a markdown list of available commands and their descriptions, and then stops further execution to prevent any unnecessary processing or database queries.
    if user_query.strip().lower() == "/help":
        help_text = """
        **🛠️ DocMate Command Palette**
        * **`/find [query]`** - Strictly extract exact quotes and numbers (no filler).
        * **`/explain [concept]`** - Break down a complex topic into simple terms.
        * **`/summarize [topic]`** - Get a rapid 3-4 bullet point overview.
        * **`/table [query]`** - Extract data and format it strictly as a Markdown table.
        * **`/report [topic]`** - Trigger a multi-step AI chain to write a comprehensive document.
        * **`/dash [query]`** - Generate a quick dashboard with KPI metrics and 1 chart.
        * **`/mdash [query]`** - Generate a deep dashboard with multiple interacting charts.
        * **`/flow [concept]`** - Build a Mermaid.js network graph or flowchart.
        * **`/sql [query]`** - Evaluate tabular constraints and write a SQL execution plan.
        * **`/search [query]`** - Search the live internet instead of local documents.
        """
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(help_text)
        st.session_state.conversation_memory.append({"role": "assistant", "content": help_text})
        st.stop() # Stop execution here so it doesn't hit the database

    # --- 2. COMMAND PARSING & INTENT OVERRIDES ---
    intent_override = None
    clean_query = user_query
    
    is_web_search = user_query.startswith("/search ")
    is_report = user_query.startswith("/report ")
    is_sql = user_query.startswith("/sql ")
    
    if user_query.startswith("/dash "):
        intent_override = "DASH_BASIC"
        clean_query = user_query.replace("/dash ", "").strip()
    elif user_query.startswith("/mdash "):
        intent_override = "DASH_MULTI"
        clean_query = user_query.replace("/mdash ", "").strip()
    elif user_query.startswith("/flow "):
        intent_override = "VISUAL_GRAPH"
        clean_query = user_query.replace("/flow ", "").strip()
    elif user_query.startswith("/table "):
        intent_override = "TABLE_GENERATION"
        clean_query = user_query.replace("/table ", "").strip()
    elif user_query.startswith("/find "):
        intent_override = "STRICT_EXTRACTION"
        clean_query = user_query.replace("/find ", "").strip()
    elif user_query.startswith("/explain "):
        intent_override = "EXPLANATION"
        clean_query = user_query.replace("/explain ", "").strip()
    elif user_query.startswith("/summarize "):
        intent_override = "SUMMARIZATION"
        clean_query = user_query.replace("/summarize ", "").strip()
    
    # --- 3. EXECUTION ROUTING ---
    if not st.session_state.processed_files and not is_web_search:
        with st.chat_message("assistant"):
            st.warning("Please upload at least one PDF document to the Library sidebar before querying, or use /search to browse the web.")
    else:
        context_chunks = []
        response_payload = None
        
        if is_web_search:
            search_term = user_query.replace("/search ", "").strip()
            with st.status(f"🌐 Searching the web for: '{search_term}'...", expanded=True) as status:
                tavily_api_key = os.environ.get("TAVILY_API_KEY")
                
                # 1. Immediate Gatekeeper
                if not tavily_api_key:
                    status.update(label="API Key Missing", state="error")
                    st.error("TAVILY_API_KEY is missing. Please add it to your .env file.")
                    st.stop() 
                
                tavily = TavilyClient(api_key=tavily_api_key)
                search_response = None
                max_retries = 1 # One retry is usually enough to bypass a temporary timeout
                
                # 2. Retry Pipeline with Depth Fallback
                for attempt in range(max_retries + 1):
                    try:
                        # Attempt 0 uses 'advanced' (deep scrape). Attempt 1 falls back to 'basic' (fast snippet).
                        depth = "advanced" if attempt == 0 else "basic"
                        
                        if attempt > 0:
                            st.toast(f"⚠️ Advanced search timed out. Retrying with basic depth...")
                            import time
                            time.sleep(1) # Brief pause to reset the connection
                            
                        search_response = tavily.search(query=search_term, search_depth=depth, max_results=3)
                        break # If successful, break out of the retry loop
                        
                    except Exception as e:
                        if attempt == max_retries:
                            status.update(label="Search failed after retries.", state="error", expanded=False)
                            st.error(f"Tavily API is currently unreachable: {e}")
                            st.stop() # Stop execution so the LLM doesn't hallucinate an answer
                
                # 3. Data Validation
                if not search_response or not search_response.get("results"):
                    status.update(label="No results found.", state="error")
                    st.warning("Could not find any relevant web results.")
                    st.stop()
                    
                # 4. Context Injection
                for i, res in enumerate(search_response.get("results", [])):
                    context_chunks.append({
                        "text": res.get('content', ''),
                        "metadata": {"source_file": res.get('url', 'Web Link'), "page_number": f"Result {i+1}"}
                    })
                status.update(label="Web search complete!", state="complete", expanded=False)

            # 5. THE MISSING PIECE: Send the web data to the AI
            cleaned_history = [
                {"role": m["role"], "content": m["content"]} 
                for m in st.session_state.conversation_memory[:-1]
            ]
            
            with st.spinner(f"Analyzing live web data as {st.session_state.active_persona}..."):
                response_payload = generate_agent_response(
                    query=clean_query, 
                    context=context_chunks, 
                    memory=cleaned_history,
                    enforced_intent=intent_override,
                    active_persona=st.session_state.active_persona 
                )
                        
        elif is_report: # This triggers the multi-step chain that performs iterative retrieval, reasoning, and composition to produce a comprehensive report. It's designed for complex queries that require synthesizing information across multiple documents and sections.
            task_query = user_query.replace("/report ", "").strip()
            with st.status(f"📝 Executing Multi-Step Chain...", expanded=True) as status:
                context_chunks = retrieve_context(query=task_query, top_k=20)
                response_payload = generate_chained_report(task_query, context_chunks)
                status.update(label="Chained execution complete!", state="complete", expanded=False)
                
        elif is_sql:
            sql_query = user_query.replace("/sql ", "").strip()
            with st.status("🗄️ Querying Live Database...", expanded=True) as status:
                from modules.database import get_db_schema, execute_sql
                
                real_schema = get_db_schema()
                if "No tables available." in real_schema or "No database found" in real_schema:
                    status.update(label="Database Empty", state="error")
                    st.error("Please upload a CSV file to the library before running SQL queries.")
                    st.stop()
                    
                # Ask the LLM to generate the SQL based on the REAL schema
                response_payload = generate_tabular_response(sql_query, real_schema)
                
                if response_payload.get("intent") == "TABULAR_SQL":
                    # Clean the markdown wrappers the LLM sometimes adds
                    clean_sql = response_payload["payload"].replace("```sql", "").replace("```", "").strip()
                    
                    # Execute against SQLite
                    df_result, error = execute_sql(clean_sql)
                    
                    if error:
                        response_payload["payload"] = f"SQL Execution Error: {error}\n\nGenerated SQL:\n```sql\n{clean_sql}\n```"
                        response_payload["intent"] = "ERROR"
                    else:
                        # Convert DataFrame to a dictionary so it can be saved in session_state memory
                        response_payload["sql_data"] = df_result.to_dict('records') 
                        response_payload["payload"] = clean_sql
                        
                status.update(label="SQL execution complete!", state="complete", expanded=False)
                
        else: # For standard queries that require retrieval from the document vector space, we perform the retrieval and then pass the chunks to the LLM for response generation. The LLM will use the retrieved context to formulate a comprehensive answer, and the intent override allows us to steer the response format if the user used a command.
            with st.status("📚 Querying Document Vector Space...", expanded=True) as status:
                context_chunks = retrieve_context(query=clean_query, top_k=15)
                status.update(label="Document search complete!", state="complete", expanded=False)
            
            cleaned_history = [
                {"role": m["role"], "content": m["content"]} 
                for m in st.session_state.conversation_memory[:-1]
            ]
            
            with st.spinner(f"Analyzing data as {st.session_state.active_persona}..."):
                response_payload = generate_agent_response(
                    query=clean_query, 
                    context=context_chunks, 
                    memory=cleaned_history,
                    enforced_intent=intent_override,
                    active_persona=st.session_state.active_persona # Pass the state here
                )

        # ==========================================
        # 7. RESPONSE PARSING & DISPLAY
        # ==========================================
        intent = response_payload.get("intent", "STANDARD_QA")
        payload = response_payload.get("payload", "")
        traced_context = response_payload.get("traced_context", [])

        with st.chat_message("assistant", avatar="🤖"):
            assistant_record = {"role": "assistant"}
            
            if intent in ["STANDARD_QA", "TABLE_GENERATION", "STRICT_EXTRACTION", "EXPLANATION", "SUMMARIZATION"]:
                
                # Split the raw payload by the new XML tags
                thinking_content, draft_content, audit_content, final_text = "", "", "", payload
                
                if "</audit>" in payload:
                    parts = payload.split("</audit>")
                    final_text = parts[-1].strip()
                    internal_process = parts[0]
                    
                    # Extract Audit
                    if "<audit>" in internal_process:
                        audit_parts = internal_process.split("<audit>")
                        audit_content = audit_parts[-1].strip()
                        internal_process = audit_parts[0]
                    
                    # Extract Thinking & Draft
                    if "<draft>" in internal_process:
                        draft_parts = internal_process.split("<draft>")
                        draft_content = draft_parts[-1].replace("</draft>", "").strip()
                        internal_process = draft_parts[0]
                    
                    if "<thinking>" in internal_process:
                        thinking_content = internal_process.replace("<thinking>", "").replace("</thinking>", "").strip()
                
                # Render the Audit Trail Expanders
                if thinking_content or audit_content:
                    cols = st.columns(2)
                    with cols[0]:
                        with st.expander("🧠 Internal Logic & Draft", expanded=False):
                            st.markdown("**1. Initial Plan**")
                            st.write(thinking_content)
                            st.markdown("**2. Raw Draft**")
                            st.write(draft_content)
                    with cols[1]:
                        with st.expander("🛡️ Quality Audit", expanded=False):
                            st.write(audit_content)
                
                # Render the final clean text
                safe_text = final_text.replace("$", "\\$")
                st.markdown(safe_text)
                assistant_record["content"] = safe_text
                    
            elif intent == "METRICS_DASHBOARD":
                with st.container():
                    # 🛠️ Pass a unique index for the current new message
                    render_dashboard_component(payload, message_idx=len(st.session_state.conversation_memory) + 999)
                assistant_record["content"] = "Dashboard and dataset extracted successfully."
                assistant_record["visual_payload"] = {"intent": intent, "payload": payload}
                
            elif intent == "TABULAR_SQL":
                text_intro = "I have generated and executed the SQL plan based on your uploaded CSV data."
                st.markdown(text_intro)
                
                # 1. Render the live DataFrame
                if "sql_data" in response_payload:
                    df = pd.DataFrame(response_payload["sql_data"])
                    st.dataframe(df, use_container_width=True)
                
                # 2. Keep the SQL code visible in an expander for auditing
                with st.expander("🗄️ View Generated SQL Plan", expanded=False):
                    st.code(payload, language="sql")
                    
                assistant_record["content"] = text_intro
                assistant_record["visual_payload"] = {
                    "intent": intent, 
                    "payload": payload,
                    "sql_data": response_payload.get("sql_data", [])
                }
            
            elif intent == "VISUAL_GRAPH":
                mermaid_string = "graph TD\n"
                node_map = {}
                node_counter = 0
                edges = payload.get("edges", [])
                
                if not edges:
                    st.warning("The AI did not extract enough structured data to draw a graph.")
                    assistant_record["content"] = "Graph generation failed."
                else:
                    for edge in edges:
                        src_clean = str(edge.get('source', 'Node')).replace('"', '').replace("'", "")
                        tgt_clean = str(edge.get('target', 'Node')).replace('"', '').replace("'", "")
                        rel_clean = str(edge.get('relationship', 'Connects')).replace('"', '').replace("'", "")
                        
                        if src_clean not in node_map:
                            node_map[src_clean] = f"NODE_{node_counter}"
                            node_counter += 1
                        src_id = node_map[src_clean]
                        
                        if tgt_clean not in node_map:
                            node_map[tgt_clean] = f"NODE_{node_counter}"
                            node_counter += 1
                        tgt_id = node_map[tgt_clean]
                        
                        mermaid_string += f'    {src_id}["{src_clean}"] -->|"{rel_clean}"| {tgt_id}["{tgt_clean}"]\n'
                    
                    text_intro = "I have compiled a cross-document network graph mapping the core relationships relative to your query."
                    st.markdown(text_intro)
                    assistant_record["content"] = text_intro
                    assistant_record["visual_payload"] = {"intent": intent, "payload": mermaid_string}
                    
                    with st.expander("📊 View Cross-Document Entity Graph", expanded=True):
                        mermaid_html = f"""
                        <style>
                            .toolbar {{ display: flex; justify-content: flex-end; gap: 10px; margin-bottom: 10px; }}
                            .action-btn {{ 
                                background: rgba(0, 150, 255, 0.1); border: 1px solid rgba(0, 150, 255, 0.4); 
                                color: #F8FAFC; padding: 6px 12px; border-radius: 8px; cursor: pointer; 
                                font-family: sans-serif; font-size: 12px; transition: all 0.3s ease;
                            }}
                            .action-btn:hover {{ background: rgba(0, 150, 255, 0.3); border-color: rgba(0, 150, 255, 0.8); box-shadow: 0px 0px 10px rgba(0, 150, 255, 0.5); }}
                        </style>
                        
                        <div class="toolbar">
                            <button class="action-btn" onclick="toggleFullScreen()">⛶ Fullscreen</button>
                            <button class="action-btn" onclick="downloadPNG()">📥 Download PNG</button>
                        </div>
                        
                        <div class="mermaid" id="graph-container" style="width: 100%; display: flex; justify-content: center; background-color: transparent;">
                            {mermaid_string}
                        </div>
                        
                        <script type="module">
                            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                            mermaid.initialize({{ startOnLoad: true, theme: 'dark', flowchart: {{ useMaxWidth: true }} }});
                            
                            window.toggleFullScreen = function() {{
                                const elem = document.documentElement;
                                if (!document.fullscreenElement) {{
                                    elem.requestFullscreen().catch(err => console.log(err));
                                }} else {{
                                    document.exitFullscreen();
                                }}
                            }};

                            window.downloadPNG = function() {{
                            const svg = document.querySelector('.mermaid svg');
                            if (!svg) return;
                            
                            // 1. Extract absolute dimensions from the viewBox
                            let width = 800;
                            let height = 600;
                            const viewBox = svg.getAttribute('viewBox');
                            
                            if (viewBox) {{
                                const parts = viewBox.split(' ');
                                if (parts.length === 4) {{
                                    width = parseFloat(parts[2]);
                                    height = parseFloat(parts[3]);
                                }}
                            }}

                            // 2. Clone the SVG and force absolute dimensions so it doesn't compress
                            const clone = svg.cloneNode(true);
                            clone.setAttribute('width', width);
                            clone.setAttribute('height', height);

                            const svgData = new XMLSerializer().serializeToString(clone);
                            const canvas = document.createElement("canvas");
                            const ctx = canvas.getContext("2d");
                            const img = new Image();
                            
                            img.onload = function() {{
                                // 3. Set canvas dimensions with comfortable padding
                                const padding = 80;
                                canvas.width = width + padding;
                                canvas.height = height + padding;
                                
                                // Fill the dark background
                                ctx.fillStyle = "#05050a"; 
                                ctx.fillRect(0, 0, canvas.width, canvas.height);
                                
                                // Draw the SVG un-squished at its original aspect ratio
                                ctx.drawImage(img, padding / 2, padding / 2, width, height);
                                
                                // 4. Trigger download
                                const a = document.createElement("a");
                                a.download = "DocMate_Flowchart.png";
                                a.href = canvas.toDataURL("image/png");
                                a.click();
                            }};
                            img.src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svgData)));
                        }};
                        </script>
                        """
                        components.html(mermaid_html, height=600, scrolling=True)
            
            elif intent == "ERROR":
                st.error(f"🚨 Backend Error: {payload}")
                assistant_record["content"] = f"Error: {payload}"
            
            # --- GLASS-BOX AUDIT LAYER ---
            # If the LLM provided a trace of which document chunks it used to arrive at its answer, we render that in an expander for transparency. This allows users to audit the AI's reasoning and verify that it's grounding its response in actual data from the documents, rather than hallucinating.
            if traced_context:
                with st.expander("🔍 Audit Thought Process & Sources", expanded=False):
                    st.write("The agent synthesized this response using the following factual fragments:")
                    for chunk in traced_context:
                        meta = chunk["metadata"]
                        st.info(f"**Source:** {meta.get('source_file', 'Unknown')} (Page {meta.get('page_number', 'N/A')})\n\n*{chunk.get('text', '')}*")
            
            st.session_state.conversation_memory.append(assistant_record)
