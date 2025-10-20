import os
import io
import re
import ast
import json
import uuid
import yaml
import random
import datetime as dt
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

# Optional imports with graceful fallback
def try_import(module_name, alias=None):
    try:
        m = __import__(module_name) if alias is None else __import__(module_name, fromlist=[alias])
        return m if alias is None else getattr(m, alias)
    except Exception:
        return None

px = try_import("plotly.express")
pdfplumber = try_import("pdfplumber")
pytesseract = try_import("pytesseract")
pdf2image = try_import("pdf2image")
PIL_Image = None
try:
    from PIL import Image as PIL_Image
except Exception:
    PIL_Image = None

# LLM SDKs (lazy)
google_genai = None
openai_module = None
xai_Client = None
xai_user = None
xai_system = None
try:
    from xai_sdk import Client
    from xai_sdk.chat import user, system
    xai_Client = Client
    xai_user = user
    xai_system = system
except ImportError:
    pass

# ----------------------------
# App configuration
# ----------------------------
st.set_page_config(page_title="Agentic Mind Studio", page_icon="🧠", layout="wide")

# ----------------------------
# Custom CSS (WOW UI)
# ----------------------------
PRIMARY = "#7C4DFF"
ACCENT = "#FF7F50"  # Coral
BG_GRADIENT = "linear-gradient(135deg, rgba(16,20,33,1) 0%, rgba(18,18,18,1) 35%, rgba(38,35,63,1) 100%)"
CARD_BG = "rgba(255,255,255,0.06)"
BORDER = "rgba(255,255,255,0.15)"

st.markdown(f"""
<style>
    .stApp {{
        background: {BG_GRADIENT};
        color: #EDEDED;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }}
    .wow-header {{
        padding: 16px 24px;
        border-radius: 16px;
        background: {CARD_BG};
        border: 1px solid {BORDER};
        backdrop-filter: blur(10px);
        margin-bottom: 12px;
    }}
    .wow-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        color: #fff;
        background: {PRIMARY};
        margin-right: 8px;
    }}
    .card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }}
    .metric-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 14px 16px;
        text-align: center;
    }}
    .metric-value {{
        font-size: 26px;
        font-weight: 700;
        color: #FFFFFF;
    }}
    .metric-label {{
        font-size: 12px;
        opacity: 0.8;
    }}
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-bottom: 8px;
    }}
    .small {{
        font-size: 12px;
        opacity: 0.85;
    }}
    .coral {{ color: {ACCENT}; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# State initialization
# ----------------------------
if "datasets" not in st.session_state:
    st.session_state.datasets = {}
if "relationships" not in st.session_state:
    st.session_state.relationships = []
if "graph_theme" not in st.session_state:
    st.session_state.graph_theme = "Midnight"
if "agents_config" not in st.session_state:
    st.session_state.agents_config = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []

def log_event(kind: str, message: str, extra: Optional[dict] = None):
    st.session_state.logs.append({
        "ts": dt.datetime.utcnow().isoformat() + "Z",
        "kind": kind,
        "message": message,
        "extra": extra or {}
    })

# ----------------------------
# Sidebar: Providers, Models, API Keys
# ----------------------------
st.sidebar.subheader("🔧 Model Providers")

provider = st.sidebar.selectbox("Provider", ["Gemini", "OpenAI", "Grok"], index=0)

gemini_key = None
openai_key = None
grok_key = None

if provider == "Gemini":
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        gemini_key = st.sidebar.text_input("Gemini API Key", type="password", key="gemini_key_input", help="Add GEMINI_API_KEY to your environment variables to avoid this.")
    else:
        st.sidebar.success("Found Gemini API Key in environment.")
    gemini_model = st.sidebar.selectbox("Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    if gemini_key:
        try:
            import google.generativeai as genai
            google_genai = genai
            google_genai.configure(api_key=gemini_key)
        except Exception as e:
            st.sidebar.error(f"Gemini SDK error: {e}")

elif provider == "OpenAI":
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        openai_key = st.sidebar.text_input("OpenAI API Key", type="password", key="openai_key_input", help="Add OPENAI_API_KEY to your environment variables to avoid this.")
    else:
        st.sidebar.success("Found OpenAI API Key in environment.")
    openai_model = st.sidebar.selectbox("OpenAI Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])
    if openai_key:
        try:
            import openai
            openai_module = openai
            openai_module.api_key = openai_key
        except Exception as e:
            st.sidebar.error(f"OpenAI SDK error: {e}")
else: # Grok
    grok_key = os.getenv("XAI_API_KEY")
    if not grok_key:
        grok_key = st.sidebar.text_input("Grok API Key", type="password", key="grok_key_input", help="Add XAI_API_KEY to your environment variables to avoid this.")
    else:
        st.sidebar.success("Found Grok API Key in environment.")
    grok_model = st.sidebar.selectbox("Grok Model", ["grok-1.5-flash", "grok-1.5"])
    if not xai_Client:
        st.sidebar.warning("`xai-sdk` not found. Please install it to use Grok.")


# Theme for graph
st.sidebar.subheader("🎨 Theme & Colors")
themes = {
    "Midnight": {"bg": "#0F1220", "edge": "#7C4DFF", "font": "#EDEDED"},
    "Sky Blue": {"bg": "#E6F7FF", "edge": "#3399FF", "font": "#003366"},
    "Deep Sea": {"bg": "#001F3F", "edge": "#0074D9", "font": "#7FDBFF"},
    "Fendi Luxury": {"bg": "#2b2520", "edge": "#6D4C41", "font": "#FFF0D8"},
}
theme_choice = st.sidebar.selectbox("Graph Theme", list(themes.keys()), 
                                     index=list(themes.keys()).index(st.session_state.graph_theme))
st.session_state.graph_theme = theme_choice
graph_theme = themes[theme_choice]
node_color = st.sidebar.color_picker("Node color", ACCENT)

# ----------------------------
# Utility functions
# ----------------------------
def parse_csv_or_lines(text: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.StringIO(text))
    except Exception:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return pd.DataFrame({"record": lines})

def read_any_dataset(file, pasted_text: Optional[str]) -> Tuple[Optional[pd.DataFrame], Optional[list]]:
    if file is not None:
        name = file.name.lower()
        try:
            if name.endswith(".csv"):
                df = pd.read_csv(file)
            elif name.endswith(".json"):
                data = json.load(file)
                df = pd.DataFrame(data if isinstance(data, list) else [data])
            elif name.endswith((".txt", ".md", ".markdown")):
                content = file.read().decode("utf-8", errors="ignore")
                df = parse_csv_or_lines(content)
            else:
                st.error("Unsupported file format.")
                return None, None
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None, None
    elif pasted_text:
        try:
            data = json.loads(pasted_text)
            df = pd.DataFrame(data if isinstance(data, list) else [data])
        except Exception:
            try:
                df = pd.read_csv(io.StringIO(pasted_text))
            except Exception:
                df = parse_csv_or_lines(pasted_text)
    else:
        return None, None
    
    if df is not None:
        return df, df.to_dict(orient="records")
    return None, None

def parse_pdf_pages(pages_spec: str) -> List[int]:
    pages = set()
    for part in re.split(r"[,\s]+", pages_spec.strip()):
        if not part:
            continue
        if "-" in part:
            try:
                a, b = part.split("-", 1)
                a, b = int(a), int(b)
                for p in range(min(a, b), max(a, b) + 1):
                    pages.add(p)
            except Exception:
                pass
        else:
            try:
                pages.add(int(part))
            except Exception:
                pass
    return sorted(list(pages))

def extract_pdf_text(file, use_ocr: bool, pages: List[int]) -> str:
    try:
        buf = file.read()
        bio = io.BytesIO(buf)
        
        if not use_ocr and pdfplumber is not None:
            with pdfplumber.open(bio) as pdf:
                selected_pages = pages if pages else list(range(1, len(pdf.pages) + 1))
                texts = []
                for p in selected_pages:
                    if 1 <= p <= len(pdf.pages):
                        page_text = pdf.pages[p-1].extract_text()
                        if page_text:
                            texts.append(page_text)
                return "\n".join(texts).strip()
        
        # OCR path
        if pdf2image is None or pytesseract is None:
            return "ERROR: OCR libraries not available. Install pdf2image, pytesseract, and tesseract."
        
        images = pdf2image.convert_from_bytes(buf)
        selected_pages = pages if pages else list(range(1, len(images) + 1))
        texts = []
        for p in selected_pages:
            if 1 <= p <= len(images):
                img = images[p-1]
                text = pytesseract.image_to_string(img)
                texts.append(text)
        return "\n".join(texts).strip()
    except Exception as e:
        return f"ERROR during PDF extraction: {e}"

def coral_highlight_markdown(text: str, keywords: List[str], color_hex: str = ACCENT) -> str:
    if not keywords:
        return text
    escaped = [re.escape(k) for k in keywords if k.strip()]
    if not escaped:
        return text
    pattern = r"(" + r"|".join(escaped) + r")"
    def repl(m):
        return f'<span style="color:{color_hex}; font-weight:600">{m.group(1)}</span>'
    try:
        return re.sub(pattern, repl, text, flags=re.IGNORECASE)
    except Exception:
        return text

def naive_keywords(text: str, topn: int = 20) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text.lower())
    stop = set("""
        the of a an and or to is are am be in on for with by as at that this these those it from
        was were will would can could should may might have has had into about across between
        you your we they them he she his her their our not no yes if else when where which who
        """.split())
    freq = {}
    for w in words:
        if w not in stop:
            freq[w] = freq.get(w, 0) + 1
    return [w for w, c in sorted(freq.items(), key=lambda x: -x[1])[:topn]]

def heuristic_entities(text: str, n: int = 100):
    kws = naive_keywords(text, topn=n)
    return [{"entity": k, "type": "keyword", "salience": round(random.uniform(0.2, 0.99), 2)} for k in kws]

def call_llm_text(provider: str, model: str, prompt: str, temperature: float = 0.2, max_tokens: int = 3000) -> str:
    try:
        if provider == "Gemini" and google_genai:
            gm = google_genai.GenerativeModel(model)
            resp = gm.generate_content(prompt)
            return getattr(resp, "text", "").strip()
        elif provider == "OpenAI" and openai_module:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        elif provider == "Grok" and grok_key and xai_Client and xai_user:
            client = xai_Client(
                api_key=grok_key,
                timeout=3600
            )
            chat = client.chat.create(model=model)
            # For simplicity and consistency with other providers in this function,
            # the entire prompt is passed as a user message.
            # The sample code separates system/user prompts, which can be adapted here if needed.
            chat.append(xai_user(prompt))
            response = chat.sample()
            return response.content.strip()
        else:
            if provider == "Grok" and not xai_Client:
                return "Grok provider selected, but `xai-sdk` is not installed. Please run `pip install xai-sdk`."
            return "LLM not configured. Please provide an API key in the sidebar."
    except Exception as e:
        return f"LLM error ({provider}): {str(e)}"

def call_openai_prompt_api(prompt_id: str, prompt_version: str, openai_key: str) -> dict:
    """Call OpenAI Prompt API with given prompt ID and version"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        
        response = client.responses.create(
            prompt={
                "id": prompt_id,
                "version": prompt_version
            }
        )
        
        return {
            "success": True,
            "response": response,
            "text": getattr(response, 'text', str(response))
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "text": f"Error calling OpenAI Prompt API: {e}"
        }

def call_llm_structured(provider: str, model: str, instruction: str, text_block: str) -> Dict[str, Any]:
    prompt = f"""
You are a precise extractor and summarizer. Using the INPUT below, produce a JSON object with:
- "entities": array of 100 objects with keys ["entity","type","salience"]
- "qa": array of 30 objects with keys ["question","answer"]
- "followups": array of 10 strings
- "summary_markdown": a well-structured Markdown summary with sections and tables where helpful.

Return ONLY valid JSON, no backticks, no commentary.

INPUT:
{text_block[:8000]}
"""
    out = call_llm_text(provider, model, prompt, temperature=0.2, max_tokens=6000)
    try:
        if out.strip().startswith("```"):
            out = "\n".join(out.strip().splitlines()[1:-1])
        data = json.loads(out)
        data.setdefault("entities", [])
        data.setdefault("qa", [])
        data.setdefault("followups", [])
        data.setdefault("summary_markdown", "")
        return data
    except Exception:
        ents = heuristic_entities(text_block, n=100)
        qas = [{"question": f"Q{i+1}: What is '{e['entity']}'?", "answer": "Context-dependent."} 
               for i, e in enumerate(ents[:30])]
        fus = [f"Follow-up {i+1}: Explore relationships of key terms." for i in range(10)]
        summary = "## Summary\n\n- Heuristic summary based on keyword frequency.\n"
        return {"entities": ents, "qa": qas, "followups": fus, "summary_markdown": summary}

def heuristic_relationships(records: List[dict]):
    pairs = set()
    for r in records:
        project = r.get("project") or r.get("title")
        team = r.get("team") or r.get("department")
        author = r.get("author")
        if project and team: 
            pairs.add((str(project), str(team)))
        if project and author: 
            pairs.add((str(project), f"Author: {author}"))
    return list(pairs)

def build_graph_elements(relationships, node_color_hex, theme: dict):
    nodes_set = set()
    for s, t in relationships:
        nodes_set.add(s)
        nodes_set.add(t)
    nodes = [Node(id=i, label=name, size=24, color=node_color_hex) 
             for i, name in enumerate(sorted(nodes_set))]
    idx = {n.label: n.id for n in nodes}
    edges = [Edge(source=idx.get(s), target=idx.get(t), color=theme["edge"]) 
             for s, t in relationships if s in idx and t in idx]
    return nodes, edges, idx

def df_to_markdown(df: pd.DataFrame, max_rows=50):
    sample = df.head(max_rows)
    out = "|" + "|".join(sample.columns.astype(str)) + "|\n"
    out += "|" + "|".join(["---"] * len(sample.columns)) + "|\n"
    for _, row in sample.iterrows():
        out += "|" + "|".join([str(v) for v in row.values]) + "|\n"
    return out

def safe_df(df):
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def status_chip(text: str, color: str = "#3DDC97"):
    st.markdown(f'<span class="wow-badge" style="background:{color}">{text}</span>', unsafe_allow_html=True)

def require_agents_yaml() -> dict:
    cfg = st.session_state.agents_config
    if cfg:
        return cfg
    default_path = "agents.yaml"
    if os.path.exists(default_path):
        try:
            with open(default_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                st.session_state.agents_config = cfg
                return cfg
        except Exception as e:
            st.warning(f"Failed to read agents.yaml: {e}")
    upl = st.file_uploader("Upload agents.yaml", type=["yaml", "yml"], key="agents_yaml_upl")
    if upl:
        try:
            cfg = yaml.safe_load(upl.read().decode("utf-8")) or {}
            st.session_state.agents_config = cfg
            return cfg
        except Exception as e:
            st.error(f"Invalid YAML: {e}")
    return {}

# ----------------------------
# Header
# ----------------------------
st.markdown("""
<div class="wow-header">
  <span class="wow-badge">🧠 Agentic Mind Studio</span>
  <span class="small">Streamlit • Multi-Provider LLM • OpenAI Prompt API • OCR • Mind Graph • Dashboard</span>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Tabs
# ----------------------------
tab_home, tab_prompt, tab_file, tab_datasets, tab_graph, tab_agents, tab_logs = st.tabs([
    "📊 Dashboard", "🎯 OpenAI Prompt API", "📄 File → Markdown", "📊 Multi-Dataset", "🕸️ Mind Graph", "🤖 Agents", "📋 Logs"
])

# ----------------------------
# Dashboard Tab
# ----------------------------
with tab_home:
    total_datasets = len(st.session_state.datasets)
    total_records = sum(len(v.get("df", [])) for v in st.session_state.datasets.values())
    total_nodes = len(set([n for pair in st.session_state.relationships for n in pair])) if st.session_state.relationships else 0
    total_edges = len(st.session_state.relationships)

    st.markdown("### 📈 Overview")
    cols = st.columns(4)
    metrics = [
        ("Datasets", total_datasets, "📁"),
        ("Records", total_records, "📝"),
        ("Graph Nodes", total_nodes, "🔵"),
        ("Graph Edges", total_edges, "🔗"),
    ]
    for col, (label, value, icon) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{icon} {label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    if px and total_datasets > 0:
        sizes, names = [], []
        for ds_id, meta in st.session_state.datasets.items():
            sizes.append(len(meta.get("df", [])))
            names.append(meta.get("name", ds_id))
        fig = px.bar(x=names, y=sizes, title="Rows per Dataset", labels={"x": "Dataset", "y": "Rows"})
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔔 Status")
    cols = st.columns(4)
    with cols: status_chip("✅ Ready", "#3DDC97")
    with cols: status_chip(f"OCR: {'✅' if pdfplumber or (pdf2image and pytesseract) else '⚠️'}", "#FFB020")
    with cols: status_chip(f"Provider: {provider}", "#3388FF")
    with cols: status_chip(f"Theme: {st.session_state.graph_theme}", "#AA66FF")

# ----------------------------
# OpenAI Prompt API Tab
# ----------------------------
with tab_prompt:
    st.markdown("### 🎯 OpenAI Prompt API Executor")
    st.markdown("Execute stored prompts using OpenAI's Prompt API with ID and version.")
    
    col1, col2 = st.columns()
    
    with col1:
        st.markdown("#### 📝 Enter Prompt Details")
        prompt_request = st.text_area(
            "Paste your prompt request/details",
            height=150,
            placeholder="Enter prompt ID, version, or full request details...",
            key="prompt_request_input"
        )
        
        # Auto-parse prompt ID and version if formatted correctly
        default_id = ""
        default_version = "1"
        if prompt_request:
            id_match = re.search(r'["\']?id["\']?\s*:\s*["\']([^"\']+)["\']', prompt_request)
            ver_match = re.search(r'["\']?version["\']?\s*:\s*["\']?(\d+)["\']?', prompt_request)
            if id_match:
                default_id = id_match.group(1)
            if ver_match:
                default_version = ver_match.group(1)
        
        prompt_id = st.text_input(
            "Prompt ID", 
            value=default_id,
            placeholder="pmpt_xxxxxxxxxxxxxxxxxx",
            help="The prompt ID from OpenAI"
        )
        
        prompt_version = st.text_input(
            "Prompt Version", 
            value=default_version,
            placeholder="1",
            help="Version number of the prompt"
        )
    
    with col2:
        st.markdown("#### ⚙️ Settings")
        use_openai_key = st.checkbox("Use OpenAI API", value=True)
        
        if use_openai_key and not openai_key:
            st.warning("⚠️ Please enter or set OpenAI API key in sidebar")
        
        st.markdown("#### 📊 Execution Stats")
        st.metric("Prompts Executed", len(st.session_state.prompt_history))
    
    st.markdown("---")
    
    execute_col1, execute_col2, execute_col3 = st.columns()
    
    with execute_col1:
        execute_btn = st.button("🚀 Execute Prompt", type="primary", use_container_width=True)
    
    with execute_col2:
        clear_btn = st.button("🗑️ Clear History", use_container_width=True)
        if clear_btn:
            st.session_state.prompt_history = []
            st.rerun()
    
    if execute_btn:
        if not prompt_id:
            st.error("❌ Please enter a Prompt ID")
        elif not prompt_version:
            st.error("❌ Please enter a Prompt Version")
        elif not openai_key:
            st.error("❌ Please enter or set OpenAI API key in sidebar")
        else:
            with st.status("🔄 Executing OpenAI Prompt API...", expanded=True) as status:
                st.write(f"📌 Prompt ID: `{prompt_id}`")
                st.write(f"📌 Version: `{prompt_version}`")
                
                result = call_openai_prompt_api(prompt_id, prompt_version, openai_key)
                
                if result["success"]:
                    status.update(label="✅ Prompt executed successfully!", state="complete")
                    log_event("prompt_api", f"Executed prompt {prompt_id} v{prompt_version}", 
                             {"prompt_id": prompt_id, "version": prompt_version})
                else:
                    status.update(label="❌ Execution failed", state="error")
                    log_event("prompt_api_error", f"Failed to execute prompt {prompt_id}", 
                             {"error": result.get("error")})
            
            # Store in history
            st.session_state.prompt_history.append({
                "timestamp": dt.datetime.now().isoformat(),
                "prompt_id": prompt_id,
                "version": prompt_version,
                "success": result["success"],
                "result": result
            })
            
            st.markdown("---")
            st.markdown("### 📤 Response")
            
            if result["success"]:
                st.success("✅ Execution successful!")
                
                response_text = result["text"]
                edited_response = st.text_area(
                    "Response (editable)", 
                    value=response_text,
                    height=300,
                    key="prompt_response_editor"
                )
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.download_button(
                        "💾 Download Response",
                        edited_response,
                        file_name=f"prompt_response_{prompt_id}_{prompt_version}.txt",
                        mime="text/plain"
                    ):
                        st.success("Downloaded!")
                
                with col_b:
                    if st.button("📋 Copy to Clipboard"):
                        st.code(edited_response)
                        st.info("Response displayed above - copy manually")
                
                # Generate follow-up questions
                st.markdown("### 💭 AI-Generated Follow-up Questions")
                
                if provider and (gemini_key or openai_key or grok_key):
                    with st.spinner("Generating follow-up questions..."):
                        followup_prompt = f"""Based on this response, generate 5 insightful follow-up questions that would help explore the topic deeper:

Response: {response_text[:2000]}

Provide exactly 5 questions, numbered 1-5."""
                        
                        if provider == "Gemini":
                            followup_model = gemini_model
                        elif provider == "OpenAI":
                            followup_model = openai_model
                        else:
                            followup_model = grok_model
                        
                        followups = call_llm_text(provider, followup_model, followup_prompt, temperature=0.7)
                        
                        if followups and not followups.startswith("LLM error"):
                            st.markdown(followups)
                        else:
                            st.info("Follow-up generation not available. Configure LLM provider in sidebar.")
                else:
                    st.info("💡 Configure an LLM provider in the sidebar to generate follow-up questions")
                
            else:
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                st.code(result.get("text", "No details available"))
    
    # History section
    if st.session_state.prompt_history:
        st.markdown("---")
        st.markdown("### 📜 Execution History")
        
        for i, entry in enumerate(reversed(st.session_state.prompt_history[-10:])):
            with st.expander(
                f"{'✅' if entry['success'] else '❌'} {entry['prompt_id']} v{entry['version']} - {entry['timestamp'][:19]}"
            ):
                st.json(entry)

# ----------------------------
# File → Markdown Tab
# ----------------------------
with tab_file:
    st.markdown("#### 📄 Upload or Paste File (txt, md, pdf)")
    colA, colB = st.columns()
    with colA:
        file = st.file_uploader("Upload TXT/MD/PDF", type=["txt", "md", "markdown", "pdf"], key="single_file_upl")
        pasted = st.text_area("Or paste text", height=180, key="single_paste_text")
    with colB:
        use_ocr = st.toggle("Use OCR for PDF", value=False)
        pages_spec = st.text_input("OCR Pages (e.g., 1-3,5)", value="")
        manual_keywords = st.text_input("Highlight keywords (comma-separated)", value="")
        coral_color = st.color_picker("Keyword color", ACCENT)

    extracted_text = ""
    if file is not None and file.name.lower().endswith(".pdf"):
        pages = parse_pdf_pages(pages_spec) if pages_spec.strip() else []
        with st.status("Extracting from PDF...", expanded=False) as status:
            extracted_text = extract_pdf_text(file, use_ocr, pages)
            if extracted_text.startswith("ERROR"):
                status.update(label="PDF extraction error", state="error")
            else:
                status.update(label="PDF extraction done", state="complete")
    elif file is not None:
        content = file.read().decode("utf-8", errors="ignore")
        extracted_text = content
    elif pasted.strip():
        extracted_text = pasted

    if extracted_text:
        st.markdown("##### Raw Text (editable)")
        edited_text = st.text_area("Edit extracted text", value=extracted_text, height=220, key="edited_text_area")
        kws = [k.strip() for k in manual_keywords.split(",")] if manual_keywords else naive_keywords(edited_text, topn=20)
        highlighted_md = coral_highlight_markdown(edited_text, kws, coral_color)
        st.markdown("##### Markdown with Highlights")
        st.markdown(highlighted_md, unsafe_allow_html=True)

        if st.button("Generate Summary + Entities + Q&A", type="primary"):
            with st.status("Summarizing...", expanded=False) as status:
                if provider == "Gemini":
                    mdl = gemini_model
                elif provider == "OpenAI":
                    mdl = openai_model
                else:
                    mdl = grok_model
                data = call_llm_structured(provider, mdl, "summarize", edited_text)
                status.update(label="Summary generated", state="complete")
            
            st.markdown("##### Summary (Markdown)")
            summary_md = st.text_area("Edit summary", value=data.get("summary_markdown", ""), height=260, key="summary_md_editor")
            st.markdown(summary_md)

            st.markdown("##### Entities (100)")
            ents_df = pd.DataFrame(data.get("entities", []))
            ents_edit = st.data_editor(ents_df, use_container_width=True, num_rows="dynamic", key="ents_editor")
            
            st.markdown("##### Q&A (30)")
            qa_df = pd.DataFrame(data.get("qa", []))
            qa_edit = st.data_editor(qa_df, use_container_width=True, num_rows="dynamic", key="qa_editor")
            
            st.markdown("##### Follow-up Questions (10)")
            fu_list = data.get("followups", [])
            fu_text = st.text_area("Edit follow-ups (one per line)", value="\n".join(fu_list), height=140)

            st.session_state.last_file_summary = {
                "summary_markdown": summary_md,
                "entities": ents_edit.to_dict(orient="records"),
                "qa": qa_edit.to_dict(orient="records"),
                "followups": [x.strip() for x in fu_text.splitlines() if x.strip()],
                "source_text": edited_text
            }
            st.success("Summary artifacts ready!")

# ----------------------------
# Multi-Dataset Analysis Tab
# ----------------------------
with tab_datasets:
    st.markdown("#### 📊 Upload Multiple Datasets")
    files = st.file_uploader("Upload files", type=["txt", "csv", "json"], accept_multiple_files=True, key="multi_up")
    pasted_multi = st.text_area("Paste dataset content", height=120, key="multi_paste")

    add_btn = st.button("Add datasets")
    if add_btn:
        added = 0
        if files:
            for f in files:
                df, recs = read_any_dataset(f, None)
                if df is not None:
                    ds_id = str(uuid.uuid4())[:8]
                    st.session_state.datasets[ds_id] = {"name": f.name, "df": df, "json": df.to_dict(orient="records")}
                    added += 1
        if pasted_multi.strip():
            df, recs = read_any_dataset(None, pasted_multi)
            if df is not None:
                ds_id = str(uuid.uuid4())[:8]
                st.session_state.datasets[ds_id] = {"name": f"Pasted-{ds_id}", "df": df, "json": df.to_dict(orient="records")}
                added += 1
        if added > 0:
            st.success(f"Added {added} dataset(s).")
            log_event("datasets:add", f"Added {added} datasets.")

    if st.session_state.datasets:
        st.markdown("##### Datasets")
        for ds_id, meta in list(st.session_state.datasets.items()):
            with st.expander(f"{meta.get('name', ds_id)} (id={ds_id})", expanded=False):
                df = safe_df(meta.get("df"))
                st.dataframe(df.head(100), use_container_width=True)
                st.markdown("Markdown Preview")
                st.code(df_to_markdown(df), language="markdown")
                st.markdown("Edit Table")
                edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, key=f"edit_{ds_id}")
                st.session_state.datasets[ds_id]["df"] = edited
                st.markdown("JSON Preview")
                json_text = st.text_area("JSON", value=json.dumps(edited.to_dict(orient="records"), ensure_ascii=False, indent=2), height=160, key=f"json_{ds_id}")
                try:
                    st.session_state.datasets[ds_id]["json"] = json.loads(json_text)
                except Exception as e:
                    st.warning(f"JSON invalid: {e}")

        if st.button("Analyze Across Datasets", type="primary"):
            with st.status("Analyzing...", expanded=False) as status:
                context_chunks = []
                for ds_id, meta in st.session_state.datasets.items():
                    name = meta.get("name", ds_id)
                    df = safe_df(meta.get("df"))
                    sample_json = json.dumps(df.head(50).to_dict(orient="records"), ensure_ascii=False)
                    context_chunks.append(f"DATASET {name}:\n{sample_json}")
                context = "\n\n".join(context_chunks)[:10000]
                
                if provider == "Gemini":
                    mdl = gemini_model
                elif provider == "OpenAI":
                    mdl = openai_model
                else:
                    mdl = grok_model
                    
                analyze_prompt = f"""
Analyze these datasets and produce JSON with:
- "report_markdown": comprehensive analysis
- "viz_suggestions": 5 visualization ideas
- "entities": 100 entities
- "qa": 30 Q&A pairs
- "followups": 10 follow-up questions

INPUT:
{context}
"""
                out = call_llm_text(provider, mdl, analyze_prompt, temperature=0.2)
                try:
                    if out.strip().startswith("```"):
                        out = "\n".join(out.strip().splitlines()[1:-1])
                    data = json.loads(out)
                except Exception:
                    data = {
                        "report_markdown": "## Cross-Dataset Report\n\nBasic analysis.\n",
                        "viz_suggestions": ["Bar chart", "Scatter plot", "Heatmap", "Timeline", "Treemap"],
                        "entities": heuristic_entities(context, n=100),
                        "qa": [{"question": f"Q{i+1}", "answer": "A"} for i in range(30)],
                        "followups": [f"Follow-up {i+1}" for i in range(10)]
                    }
                status.update(label="Analysis complete", state="complete")

            st.markdown("##### Report")
            report_md = st.text_area("Edit report", value=data.get("report_markdown", ""), height=280, key="cross_report_md")
            st.markdown(report_md)

            st.markdown("##### Visualization Suggestions")
            vs_txt = st.text_area("Edit viz", value="\n".join(data.get("viz_suggestions", [])), height=120, key="viz_suggestions")
            
            st.markdown("##### Entities (100)")
            ents_edit = st.data_editor(pd.DataFrame(data.get("entities", [])), use_container_width=True, num_rows="dynamic", key="cross_entities")
            
            st.markdown("##### Q&A (30)")
            qa_edit = st.data_editor(pd.DataFrame(data.get("qa", [])), use_container_width=True, num_rows="dynamic", key="cross_qa")
            
            st.markdown("##### Follow-ups (10)")
            fu_txt = st.text_area("Edit follow-ups", value="\n".join(data.get("followups", [])), height=140, key="cross_followups")

            st.session_state.cross_analysis = {
                "report_markdown": report_md,
                "viz_suggestions": [x.strip() for x in vs_txt.splitlines() if x.strip()],
                "entities": ents_edit.to_dict(orient="records"),
                "qa": qa_edit.to_dict(orient="records"),
                "followups": [x.strip() for x in fu_txt.splitlines() if x.strip()],
            }
            st.success("Cross-dataset analysis ready!")

# ----------------------------
# Mind Graph Tab
# ----------------------------
with tab_graph:
    st.markdown("#### 🕸️ Build Mind Graph")
    colX, colY = st.columns([2, 1])
    with colX:
        uploaded = st.file_uploader("Upload dataset", type=["csv","json","txt"], key="graph_upl")
        pasted_text = st.text_area("Or paste content", height=120, key="graph_paste")
    with colY:
        infer_button = st.button("Infer Relationships", type="primary")
        filter_kw = st.text_input("Filter nodes", value="")

    df, records = read_any_dataset(uploaded, pasted_text)
    if df is not None:
        st.dataframe(df.head(), use_container_width=True)

        if not st.session_state.relationships:
            st.session_state.relationships = heuristic_relationships(records)

        if infer_button:
            with st.status("Inferring...", expanded=False) as s:
                ctx = "\n".join([json.dumps(r, ensure_ascii=False) for r in records[:50]])
                prompt = f"Extract relationships as Python list of tuples: [('A','B'), ...]\n\n{ctx}"
                
                if provider == "Gemini" and google_genai:
                    try:
                        model = google_genai.GenerativeModel(gemini_model)
                        resp = model.generate_content(prompt)
                        text = getattr(resp, "text", "").strip()
                        if text.startswith("```"): 
                            text = "\n".join(text.splitlines()[1:-1])
                        st.session_state.relationships = ast.literal_eval(text)
                    except Exception:
                        st.session_state.relationships = heuristic_relationships(records)
                else:
                    st.session_state.relationships = heuristic_relationships(records)
                s.update(label="Done", state="complete")

        rels_df = pd.DataFrame(st.session_state.relationships, columns=["source","target"])
        edited_df = st.data_editor(rels_df, num_rows="dynamic", use_container_width=True, key="graph_editor")
        st.session_state.relationships = list(edited_df.itertuples(index=False, name=None))

        relationships = st.session_state.relationships
        if filter_kw.strip():
            relationships = [pair for pair in relationships if filter_kw.lower() in pair.lower() or filter_kw.lower() in pair.lower()]

        nodes, edges, node_index = build_graph_elements(relationships, node_color, graph_theme)
        config = Config(
            width=1100,
            height=600,
            directed=True,
            physics=True,
            nodeHighlightBehavior=True,
            highlightColor=graph_theme["edge"],
            bgcolor=graph_theme["bg"],
            font={"color": graph_theme["font"], "size": 12}
        )
        st.subheader("Interactive Graph")
        selected = agraph(nodes=nodes, edges=edges, config=config)
    else:
        st.info("Upload dataset to build graph")

# ----------------------------
# Agents Tab
# ----------------------------
with tab_agents:
    st.markdown("#### 🤖 Agent Execution")
    cfg = require_agents_yaml()
    if not cfg:
        st.info("Provide agents.yaml")
    else:
        agents = cfg.get("agents", [])
        names = [a.get("name", f"agent_{i}") for i, a in enumerate(agents)]
        if not names:
            st.warning("No agents in agents.yaml")
        else:
            agent_name = st.selectbox("Select agent", names)
            agent = next((a for a in agents if a.get("name") == agent_name), None) or {}

            raw_yaml = st.text_area("Edit agent", value=yaml.safe_dump(agent, allow_unicode=True, sort_keys=False), height=220)
            try:
                agent = yaml.safe_load(raw_yaml) or {}
            except Exception as e:
                st.error(f"Invalid YAML: {e}")

            agent_provider = st.selectbox("Provider", ["Gemini", "OpenAI", "Grok"], index=0, key="agent_provider")
            if agent_provider == "Gemini":
                agent_model = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"], key="agent_model_gemini")
            elif agent_provider == "OpenAI":
                agent_model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], key="agent_model_openai")
            else:
                agent_model = st.selectbox("Model", ["grok-1.5-flash", "grok-1.5"], key="agent_model_grok")

            system_prompt = agent.get("system_prompt", "You are helpful.")
            user_prompt = st.text_area("User prompt", value=agent.get("user_prompt", ""), height=160)

            src_opt = st.radio("Input", ["Manual", "File Summary", "Cross-Dataset"])
            if src_opt == "Manual":
                input_text = st.text_area("Input", height=180, key="agent_manual")
            elif src_opt == "File Summary":
                input_text = st.session_state.get("last_file_summary", {}).get("summary_markdown", "")
                st.text_area("Preview", value=input_text, height=160, disabled=True)
            else:
                input_text = st.session_state.get("cross_analysis", {}).get("report_markdown", "")
                st.text_area("Preview", value=input_text, height=160, disabled=True)

            temp = st.slider("Temperature", 0.0, 1.0, float(agent.get("temperature", 0.2)), 0.05)
            
            if st.button("Run Agent", type="primary"):
                with st.status("Executing...", expanded=False) as status:
                    full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nContext:\n{input_text}"
                    resp = call_llm_text(agent_provider, agent_model, full_prompt, temperature=temp)
                    status.update(label="Complete", state="complete")
                
                st.markdown("##### Response")
                resp_edit = st.text_area("Edit response", value=resp, height=260, key="agent_resp")
                
                # Generate follow-up questions
                st.markdown("##### Follow-up Questions")
                with st.spinner("Generating..."):
                    fu_prompt = f"Generate 5 follow-up questions for:\n\n{resp[:1000]}"
                    followups = call_llm_text(agent_provider, agent_model, fu_prompt, temperature=0.7)
                    st.markdown(followups)

# ----------------------------
# Logs Tab
# ----------------------------
with tab_logs:
    st.markdown("#### 📋 Activity Log")
    if st.session_state.logs:
        for log in reversed(st.session_state.logs[-50:]):
            with st.expander(f"{log['kind']} - {log['ts'][:19]}"):
                st.json(log)
    else:
        st.info("No logs yet")
