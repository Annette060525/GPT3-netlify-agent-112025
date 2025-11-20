import os
import time
import textwrap
from typing import Dict, List, Literal

import yaml
import streamlit as st

# Optional but recommended for dashboard charts
try:
    import pandas as pd
    import altair as alt

    HAS_CHARTS = True
except Exception:
    HAS_CHARTS = False

# -------------------------------------------------------------
# 1. Basic config
# -------------------------------------------------------------
st.set_page_config(
    page_title="FlowerMind Agent Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# 2. Theme & Language Configuration
# -------------------------------------------------------------
ThemeMode = Literal["light", "dark"]
LanguageCode = Literal["en", "zh-TW"]

FLOWER_THEMES: Dict[str, Dict[str, str]] = {
    # id: { name_en, name_zh, primary, secondary, accent, bg_light, bg_dark }
    "sakura": {
        "name_en": "Sakura",
        "name_zh": "櫻花",
        "primary": "#ff80ab",
        "secondary": "#ffe4ec",
        "accent": "#d81b60",
        "bg_light": "#fff7fb",
        "bg_dark": "#2b1020",
    },
    "lotus": {
        "name_en": "Lotus",
        "name_zh": "蓮花",
        "primary": "#ffb3c1",
        "secondary": "#c8f7dc",
        "accent": "#2e7d32",
        "bg_light": "#f7fff9",
        "bg_dark": "#11261a",
    },
    "peony": {
        "name_en": "Peony",
        "name_zh": "牡丹",
        "primary": "#f06292",
        "secondary": "#fce4ec",
        "accent": "#ad1457",
        "bg_light": "#fff5f9",
        "bg_dark": "#280b19",
    },
    "orchid": {
        "name_en": "Orchid",
        "name_zh": "蘭花",
        "primary": "#ba68c8",
        "secondary": "#ede7f6",
        "accent": "#6a1b9a",
        "bg_light": "#faf5ff",
        "bg_dark": "#1e1030",
    },
    "plum": {
        "name_en": "Plum Blossom",
        "name_zh": "梅花",
        "primary": "#ef5350",
        "secondary": "#f3e5f5",
        "accent": "#8e24aa",
        "bg_light": "#fff5f5",
        "bg_dark": "#241015",
    },
    "chrysanthemum": {
        "name_en": "Chrysanthemum",
        "name_zh": "菊花",
        "primary": "#ffca28",
        "secondary": "#fff8e1",
        "accent": "#f9a825",
        "bg_light": "#fffdf5",
        "bg_dark": "#261c06",
    },
    "rose": {
        "name_en": "Rose",
        "name_zh": "玫瑰",
        "primary": "#e53935",
        "secondary": "#ffebee",
        "accent": "#b71c1c",
        "bg_light": "#fff5f6",
        "bg_dark": "#25080a",
    },
    "sunflower": {
        "name_en": "Sunflower",
        "name_zh": "向日葵",
        "primary": "#ffeb3b",
        "secondary": "#fffde7",
        "accent": "#fbc02d",
        "bg_light": "#fffef5",
        "bg_dark": "#262008",
    },
    "lavender": {
        "name_en": "Lavender",
        "name_zh": "薰衣草",
        "primary": "#9575cd",
        "secondary": "#ede7f6",
        "accent": "#5e35b1",
        "bg_light": "#f4f2ff",
        "bg_dark": "#17132b",
    },
    "camellia": {
        "name_en": "Camellia",
        "name_zh": "山茶花",
        "primary": "#ef9a9a",
        "secondary": "#fbe9e7",
        "accent": "#c62828",
        "bg_light": "#fff6f3",
        "bg_dark": "#2b1511",
    },
    "hydrangea": {
        "name_en": "Hydrangea",
        "name_zh": "繡球花",
        "primary": "#64b5f6",
        "secondary": "#e3f2fd",
        "accent": "#1976d2",
        "bg_light": "#f3f8ff",
        "bg_dark": "#111c2b",
    },
    "magnolia": {
        "name_en": "Magnolia",
        "name_zh": "木蘭",
        "primary": "#ffcc80",
        "secondary": "#fff3e0",
        "accent": "#fb8c00",
        "bg_light": "#fffaf3",
        "bg_dark": "#261a0c",
    },
    "jasmine": {
        "name_en": "Jasmine",
        "name_zh": "茉莉",
        "primary": "#c5e1a5",
        "secondary": "#f1f8e9",
        "accent": "#558b2f",
        "bg_light": "#f9fff5",
        "bg_dark": "#161f11",
    },
    "wisteria": {
        "name_en": "Wisteria",
        "name_zh": "紫藤",
        "primary": "#ce93d8",
        "secondary": "#f3e5f5",
        "accent": "#8e24aa",
        "bg_light": "#faf3ff",
        "bg_dark": "#211127",
    },
    "azalea": {
        "name_en": "Azalea",
        "name_zh": "杜鵑",
        "primary": "#f48fb1",
        "secondary": "#fce4ec",
        "accent": "#d81b60",
        "bg_light": "#fff4f9",
        "bg_dark": "#26111c",
    },
    "daffodil": {
        "name_en": "Daffodil",
        "name_zh": "水仙",
        "primary": "#fff176",
        "secondary": "#fffde7",
        "accent": "#fbc02d",
        "bg_light": "#fffef5",
        "bg_dark": "#241f09",
    },
    "iris": {
        "name_en": "Iris",
        "name_zh": "鳶尾花",
        "primary": "#7986cb",
        "secondary": "#e8eaf6",
        "accent": "#303f9f",
        "bg_light": "#f5f6ff",
        "bg_dark": "#12172b",
    },
    "poppy": {
        "name_en": "Poppy",
        "name_zh": "罌粟",
        "primary": "#ff8a65",
        "secondary": "#fbe9e7",
        "accent": "#e64a19",
        "bg_light": "#fff6f3",
        "bg_dark": "#28160f",
    },
    "tulip": {
        "name_en": "Tulip",
        "name_zh": "鬱金香",
        "primary": "#ff8a80",
        "secondary": "#fbe9e7",
        "accent": "#d32f2f",
        "bg_light": "#fff7f5",
        "bg_dark": "#281111",
    },
    "lotus_night": {
        "name_en": "Lotus Night",
        "name_zh": "夜蓮",
        "primary": "#ff80ab",
        "secondary": "#263238",
        "accent": "#00bfa5",
        "bg_light": "#f6f8ff",
        "bg_dark": "#020817",
    },
}

# UI labels for EN / Traditional Chinese
UI_TEXT: Dict[LanguageCode, Dict[str, str]] = {
    "en": {
        "app_title": "FlowerMind Agent Studio",
        "tab_agents": "Agents Console",
        "tab_attachment": "Attachment Chat",
        "tab_notes": "Notes Studio",
        "tab_dashboard": "Dashboard",
        "sidebar_controls": "Control Panel",
        "provider": "Provider",
        "model": "Model",
        "max_tokens": "Max tokens",
        "temperature": "Temperature",
        "prompt_id": "Prompt ID",
        "system_prompt": "System Prompt",
        "api_keys": "API Keys (local only)",
        "gemini_key": "Gemini API Key",
        "openai_key": "OpenAI API Key",
        "anthropic_key": "Anthropic API Key",
        "language": "Language",
        "theme_mode": "Theme mode",
        "theme_light": "Light",
        "theme_dark": "Dark",
        "flower_style": "Flower style",
        "agents_header": "Agents Console",
        "your_message": "Your message",
        "run_agent": "Run Agent",
        "attachment_header": "Attachment Chat",
        "upload_file": "Upload a file (PDF, TXT, DOCX, etc.)",
        "ask_attachment": "Ask about this attachment",
        "ask_btn": "Ask Attachment",
        "notes_header": "Notes Studio",
        "raw_notes": "Raw Notes",
        "notes_placeholder": "Type or paste your notes here",
        "btn_notes_to_md": "Transform to Markdown (AI)",
        "btn_notes_format": "AI Improve Formatting",
        "keywords": "Keywords (comma-separated)",
        "keyword_color": "Color for these keywords",
        "md_preview": "Markdown Preview",
        "dashboard_header": "Dashboard & Wow Indicators",
        "metrics_usage": "Usage Overview",
        "tips": "Tips",
        "q": "Q",
        "a": "A",
    },
    "zh-TW": {
        "app_title": "FlowerMind 智能代理工作室",
        "tab_agents": "智能代理主控台",
        "tab_attachment": "附件對話",
        "tab_notes": "筆記工作室",
        "tab_dashboard": "儀表板",
        "sidebar_controls": "控制面板",
        "provider": "模型提供者",
        "model": "模型",
        "max_tokens": "最大 Token 數",
        "temperature": "溫度 (隨機度)",
        "prompt_id": "提示 ID",
        "system_prompt": "系統提示",
        "api_keys": "API 金鑰（僅本機儲存）",
        "gemini_key": "Gemini API 金鑰",
        "openai_key": "OpenAI API 金鑰",
        "anthropic_key": "Anthropic API 金鑰",
        "language": "介面語言",
        "theme_mode": "主題模式",
        "theme_light": "亮色",
        "theme_dark": "暗色",
        "flower_style": "花卉主題",
        "agents_header": "智能代理主控台",
        "your_message": "您的訊息",
        "run_agent": "執行代理",
        "attachment_header": "附件對話",
        "upload_file": "上傳檔案（PDF, TXT, DOCX 等）",
        "ask_attachment": "就附件內容發問",
        "ask_btn": "詢問附件",
        "notes_header": "筆記工作室",
        "raw_notes": "原始筆記",
        "notes_placeholder": "在此輸入或貼上您的筆記",
        "btn_notes_to_md": "轉換為 Markdown（AI）",
        "btn_notes_format": "AI 排版強化",
        "keywords": "關鍵字（以逗號分隔）",
        "keyword_color": "關鍵字顏色",
        "md_preview": "Markdown 預覽",
        "dashboard_header": "儀表板與 Wow 指標",
        "metrics_usage": "使用概況",
        "tips": "小提示",
        "q": "問",
        "a": "答",
    },
}


def get_text(key: str) -> str:
    lang: LanguageCode = st.session_state.get("ui_lang", "en")
    return UI_TEXT.get(lang, UI_TEXT["en"]).get(key, key)


def apply_theme():
    mode: ThemeMode = st.session_state.get("ui_theme_mode", "light")
    theme_id: str = st.session_state.get("ui_flower_theme", "sakura")
    theme = FLOWER_THEMES.get(theme_id, FLOWER_THEMES["sakura"])

    bg_color = theme["bg_dark"] if mode == "dark" else theme["bg_light"]
    primary = theme["primary"]
    accent = theme["accent"]

    css = f"""
    <style>
    .stApp {{
        background: radial-gradient(circle at top left, {theme["secondary"]} 0, {bg_color} 40%, {bg_color} 100%);
        color: {'#f9fafb' if mode == 'dark' else '#111827'};
    }}
    .stSidebar {{
        background-color: rgba(15, 23, 42, {0.85 if mode == 'dark' else 0.05}) !important;
    }}
    section.main > div {{
        background-color: rgba(255,255,255,{0.02 if mode == 'dark' else 0.85});
        backdrop-filter: blur(18px);
        border-radius: 18px;
        padding: 1.2rem 1.4rem 2rem 1.4rem;
        border: 1px solid rgba(148, 163, 184, {0.45 if mode=='dark' else 0.4});
    }}
    .wow-primary {{
        color: {primary} !important;
    }}
    .wow-accent {{
        color: {accent} !important;
    }}
    .wow-pill {{
        border-radius: 999px;
        padding: 0.25rem 0.9rem;
        border: 1px solid {accent}55;
        background: linear-gradient(90deg, {accent}11, transparent);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .wow-card {{
        border-radius: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.4);
        padding: 1rem 1.25rem;
        background: rgba(15, 23, 42, {0.65 if mode == 'dark' else 0.02});
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# -------------------------------------------------------------
# 3. Helpers: agents config, API keys, LLM call, metrics
# -------------------------------------------------------------
@st.cache_data
def load_agents_config(path: str = "agents.yaml") -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


def get_api_key(env_name: str, ui_label: str, session_key: str) -> str:
    """
    Priority:
      1. Environment variable (never shown).
      2. User-entered key (password input).
    """
    env_val = os.getenv(env_name)
    if env_val:
        return env_val

    if session_key not in st.session_state:
        st.session_state[session_key] = ""

    user_val = st.text_input(ui_label, type="password", value=st.session_state[session_key])
    if user_val:
        st.session_state[session_key] = user_val
    return st.session_state[session_key]


def estimate_tokens(messages: List[Dict[str, str]]) -> int:
    chars = sum(len(m.get("content", "")) for m in messages)
    return max(1, chars // 4)


def init_metrics():
    if "metrics" not in st.session_state:
        st.session_state.metrics = {
            "total_requests": 0,
            "agents_requests": 0,
            "attachment_requests": 0,
            "notes_requests": 0,
            "tokens_estimated": 0,
            "latencies": [],
        }


def record_metric(context: str, tokens: int, latency: float):
    init_metrics()
    m = st.session_state.metrics
    m["total_requests"] += 1
    m["tokens_estimated"] += tokens
    m["latencies"].append(latency)
    if context == "agents":
        m["agents_requests"] += 1
    elif context == "attachment":
        m["attachment_requests"] += 1
    elif context == "notes":
        m["notes_requests"] += 1


def call_model(
    provider: str,
    model_name: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    api_keys: Dict[str, str],
) -> str:
    """
    Central place to call Gemini / OpenAI / Anthropic.

    For now this is a demo implementation that echoes the last user message.
    Integrate your real SDK calls here using `api_keys`.
    """
    user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    demo_response = textwrap.shorten(user_msg, width=300, placeholder="...")
    return f"[Demo response from {provider}/{model_name}] {demo_response}"


def highlight_keywords_in_markdown(text: str, keyword_color_map: Dict[str, str]) -> str:
    if not keyword_color_map:
        return text
    for kw, color in keyword_color_map.items():
        if not kw.strip():
            continue
        span = f"<span style='color:{color}'>{kw}</span>"
        text = text.replace(kw, span)
    return text


# -------------------------------------------------------------
# 4. Session defaults: language & theme
# -------------------------------------------------------------
if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = "en"
if "ui_theme_mode" not in st.session_state:
    st.session_state.ui_theme_mode = "light"
if "ui_flower_theme" not in st.session_state:
    st.session_state.ui_flower_theme = "sakura"

apply_theme()  # must be early

# -------------------------------------------------------------
# 5. Sidebar controls (language, theme, models, prompts, keys)
# -------------------------------------------------------------
agents_cfg = load_agents_config()

with st.sidebar:
    st.markdown(f"### 🌸 {get_text('app_title')}")
    st.markdown("---")

    # Language
    lang = st.selectbox(
        get_text("language"),
        options=["en", "zh-TW"],
        format_func=lambda x: "English" if x == "en" else "繁體中文",
        index=0 if st.session_state.ui_lang == "en" else 1,
    )
    st.session_state.ui_lang = lang  # update
    # Theme mode
    mode = st.radio(
        get_text("theme_mode"),
        options=["light", "dark"],
        format_func=lambda x: get_text("theme_light") if x == "light" else get_text("theme_dark"),
        horizontal=True,
        index=0 if st.session_state.ui_theme_mode == "light" else 1,
    )
    st.session_state.ui_theme_mode = mode

    # Flower theme select
    def flower_label(fid: str) -> str:
        t = FLOWER_THEMES[fid]
        return f"{t['name_en']} / {t['name_zh']}"

    flower_id = st.selectbox(
        get_text("flower_style"),
        options=list(FLOWER_THEMES.keys()),
        format_func=flower_label,
        index=list(FLOWER_THEMES.keys()).index(st.session_state.ui_flower_theme)
        if st.session_state.ui_flower_theme in FLOWER_THEMES
        else 0,
    )
    st.session_state.ui_flower_theme = flower_id

    st.markdown("---")
    st.markdown(f"#### ⚙️ {get_text('sidebar_controls')}")

    provider = st.selectbox(get_text("provider"), ["Gemini", "OpenAI", "Anthropic"])
    model_options = {
        "Gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "OpenAI": ["gpt-5-nano", "gpt-4o-mini", "gpt-4.1-mini"],
        "Anthropic": ["claude-3-5-sonnet", "claude-3-5-haiku"],
    }[provider]
    model_name = st.selectbox(get_text("model"), model_options)

    max_tokens = st.slider(get_text("max_tokens"), 128, 8192, 1024, 128)
    temperature = st.slider(get_text("temperature"), 0.0, 1.5, 0.7, 0.1)

    # Prompt IDs
    prompt_id = st.selectbox(get_text("prompt_id"), ["Custom", "Daily", "pmpt_69A", "pmpt_69B"])

    prompt_templates = {
        "Daily": "You are my daily assistant. Help me plan, prioritize, and summarize today clearly.",
        "pmpt_69A": (
            "You are an expert reasoning assistant. Think step-by-step, "
            "explain assumptions, and verify each step before answering."
        ),
        "pmpt_69B": (
            "You are a creative ideation assistant. Generate multiple diverse ideas, "
            "evaluate pros and cons, and suggest next actions."
        ),
    }
    default_system_prompt = prompt_templates.get(prompt_id, "You are a helpful AI assistant.")
    system_prompt = st.text_area(
        get_text("system_prompt"),
        value=default_system_prompt,
        height=150,
    )

    # API keys
    st.markdown(f"#### 🔐 {get_text('api_keys')}")
    gemini_key = get_api_key("GEMINI_API_KEY", get_text("gemini_key"), "gemini_api_key")
    openai_key = get_api_key("OPENAI_API_KEY", get_text("openai_key"), "openai_api_key")
    anthropic_key = get_api_key("ANTHROPIC_API_KEY", get_text("anthropic_key"), "anthropic_api_key")

    api_keys = {"Gemini": gemini_key, "OpenAI": openai_key, "Anthropic": anthropic_key}

# -------------------------------------------------------------
# 6. Main layout: tabs
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        f"🧠 {get_text('tab_agents')}",
        f"📎 {get_text('tab_attachment')}",
        f"📝 {get_text('tab_notes')}",
        f"📊 {get_text('tab_dashboard')}",
    ]
)

# -------------------------------------------------------------
# 6.1 Agents Console
# -------------------------------------------------------------
with tab1:
    st.markdown(
        f"<div class='wow-pill'>Agents</div>", unsafe_allow_html=True
    )
    st.markdown(f"## {get_text('agents_header')}")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Optional: pick agent from agents.yaml, if present
    if agents_cfg:
        agent_names = list(agents_cfg.keys())
        selected_agent = st.selectbox("Agent (from agents.yaml)", agent_names)
        st.caption(f"Loaded from agents.yaml → `{selected_agent}`")
    else:
        selected_agent = None
        st.caption("No agents.yaml found – using generic assistant.")

    user_message = st.text_area(get_text("your_message"), "", height=140)

    col_run, col_clear = st.columns([1, 1])
    with col_run:
        if st.button(get_text("run_agent"), use_container_width=True):
            if not user_message.strip():
                st.warning("Please enter a message.")
            else:
                st.session_state.chat_history.append({"role": "user", "content": user_message})
                messages = [{"role": "system", "content": system_prompt}] + st.session_state.chat_history

                start = time.time()
                response = call_model(
                    provider=provider,
                    model_name=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    api_keys=api_keys,
                )
                latency = time.time() - start

                st.session_state.chat_history.append({"role": "assistant", "content": response})
                tokens_est = estimate_tokens(messages)
                record_metric("agents", tokens_est, latency)

                st.success(f"Response in {latency:.2f}s • ~{tokens_est} tokens")
    with col_clear:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []

    st.markdown("---")
    for i, msg in enumerate(st.session_state.chat_history):
        speaker = "👤 You" if msg["role"] == "user" else "🤖 Assistant"
        box_class = "wow-card"
        st.markdown(
            f"<div class='{box_class}'><strong>{speaker}:</strong><br>{msg['content']}</div>",
            unsafe_allow_html=True,
        )

# -------------------------------------------------------------
# 6.2 Attachment Chat
# -------------------------------------------------------------
with tab2:
    st.markdown(
        f"<div class='wow-pill'>Documents</div>", unsafe_allow_html=True
    )
    st.markdown(f"## {get_text('attachment_header')}")

    if "attachment_chat" not in st.session_state:
        st.session_state.attachment_chat = []
        st.session_state.attachment_content = ""

    uploaded_file = st.file_uploader(get_text("upload_file"), type=None)

    if uploaded_file is not None:
        raw_bytes = uploaded_file.read()
        try:
            decoded = raw_bytes.decode("utf-8", errors="ignore")
        except Exception:
            decoded = str(raw_bytes)
        st.session_state.attachment_content = decoded
        st.info(f"Loaded attachment: {uploaded_file.name} ({len(decoded)} characters)")

    st.markdown(f"**{get_text('ask_attachment')}**")
    chat_on_attachment = st.text_area("", "", height=100)

    col_ask, col_clear2 = st.columns([1, 1])
    with col_ask:
        if st.button(get_text("ask_btn"), use_container_width=True):
            if not st.session_state.attachment_content:
                st.warning("Please upload an attachment first.")
            elif not chat_on_attachment.strip():
                st.warning("Please enter your question.")
            else:
                context_snippet = st.session_state.attachment_content[:6000]
                user_prompt = (
                    "You are given the following document content:\n\n"
                    f"{context_snippet}\n\n"
                    "Please answer the user's question based strictly on the document when possible.\n\n"
                    f"User question:\n{chat_on_attachment}"
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                start = time.time()
                response = call_model(
                    provider=provider,
                    model_name=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    api_keys=api_keys,
                )
                latency = time.time() - start
                tokens_est = estimate_tokens(messages)
                record_metric("attachment", tokens_est, latency)

                st.session_state.attachment_chat.append(
                    {"question": chat_on_attachment, "answer": response, "latency": latency}
                )
                st.success(f"Response in {latency:.2f}s • ~{tokens_est} tokens")
                chat_on_attachment = ""

    with col_clear2:
        if st.button("Clear history", use_container_width=True):
            st.session_state.attachment_chat = []

    st.markdown("---")
    for turn in st.session_state.attachment_chat:
        st.markdown(
            f"<div class='wow-card'><strong>{get_text('q')}:</strong> {turn['question']}<br>"
            f"<strong>{get_text('a')}:</strong> {turn['answer']}</div>",
            unsafe_allow_html=True,
        )

# -------------------------------------------------------------
# 6.3 Notes Studio
# -------------------------------------------------------------
with tab3:
    st.markdown(
        f"<div class='wow-pill'>Notes</div>", unsafe_allow_html=True
    )
    st.markdown(f"## {get_text('notes_header')}")

    if "notes_text" not in st.session_state:
        st.session_state.notes_text = ""
    if "notes_markdown" not in st.session_state:
        st.session_state.notes_markdown = ""

    st.subheader(f"📝 {get_text('raw_notes')}")
    st.session_state.notes_text = st.text_area(
        get_text("notes_placeholder"),
        value=st.session_state.notes_text,
        height=220,
    )

    col_md, col_fmt = st.columns(2)
    with col_md:
        run_ai_markdown = st.button(get_text("btn_notes_to_md"), use_container_width=True)
    with col_fmt:
        run_ai_formatting = st.button(get_text("btn_notes_format"), use_container_width=True)

    if run_ai_markdown:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a note formatting assistant. "
                    "Convert the user's notes into clean, well-structured Markdown. "
                    "Use headings, bullet lists, code blocks, and tables when appropriate. "
                    "Do not add extra commentary beyond formatting."
                ),
            },
            {"role": "user", "content": st.session_state.notes_text},
        ]
        start = time.time()
        md = call_model(
            provider=provider,
            model_name=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
            api_keys=api_keys,
        )
        latency = time.time() - start
        tokens_est = estimate_tokens(messages)
        record_metric("notes", tokens_est, latency)
        st.session_state.notes_markdown = md
        st.success(f"Markdown created in {latency:.2f}s • ~{tokens_est} tokens")

    if run_ai_formatting and st.session_state.notes_markdown.strip():
        messages = [
            {
                "role": "system",
                "content": (
                    "Improve the formatting of the given Markdown. "
                    "Keep the same meaning, but enhance structure, headings, "
                    "and readability. Output Markdown only."
                ),
            },
            {"role": "user", "content": st.session_state.notes_markdown},
        ]
        start = time.time()
        improved_md = call_model(
            provider=provider,
            model_name=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.4,
            api_keys=api_keys,
        )
        latency = time.time() - start
        tokens_est = estimate_tokens(messages)
        record_metric("notes", tokens_est, latency)
        st.session_state.notes_markdown = improved_md
        st.success(f"Formatting improved in {latency:.2f}s • ~{tokens_est} tokens")

    st.markdown("---")
    st.subheader(f"🔍 {get_text('keywords')}")
    kw_input = st.text_input(get_text("keywords"), "TODO,Important,Deadline")
    kw_color = st.color_picker(get_text("keyword_color"), "#ff4081")

    keyword_map: Dict[str, str] = {}
    if kw_input.strip():
        for kw in [k.strip() for k in kw_input.split(",")]:
            if kw:
                keyword_map[kw] = kw_color

    st.subheader(f"📄 {get_text('md_preview')}")
    if not st.session_state.notes_markdown.strip():
        st.info("Use AI to generate Markdown from your notes first.")
    else:
        colored_md = highlight_keywords_in_markdown(st.session_state.notes_markdown, keyword_map)
        st.markdown(colored_md, unsafe_allow_html=True)

# -------------------------------------------------------------
# 6.4 Dashboard & Wow Indicators
# -------------------------------------------------------------
with tab4:
    st.markdown(
        f"<div class='wow-pill'>Insights</div>", unsafe_allow_html=True
    )
    st.markdown(f"## {get_text('dashboard_header')}")

    init_metrics()
    m = st.session_state.metrics
    total_req = m["total_requests"]
    agents_req = m["agents_requests"]
    attach_req = m["attachment_requests"]
    notes_req = m["notes_requests"]
    tokens_total = m["tokens_estimated"]
    avg_latency = sum(m["latencies"]) / len(m["latencies"]) if m["latencies"] else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Requests", total_req)
    c2.metric("Agents Chats", agents_req)
    c3.metric("Attachment Q&A", attach_req)
    c4.metric("Notes Actions", notes_req)

    st.markdown("### ⏱ Performance")
    colA, colB = st.columns(2)
    with colA:
        st.metric("Average Latency (s)", f"{avg_latency:.2f}")
    with colB:
        st.metric("Estimated Tokens Used", tokens_total)

    st.markdown(f"### 🔭 {get_text('metrics_usage')}")
    st.progress(min(total_req / 50.0, 1.0))

    if HAS_CHARTS and total_req > 0:
        st.markdown("#### Activity by Context")
        data = pd.DataFrame(
            {
                "Context": ["Agents", "Attachment", "Notes"],
                "Requests": [agents_req, attach_req, notes_req],
            }
        )
        chart = (
            alt.Chart(data)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(x="Context", y="Requests", color="Context")
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown(f"### 💡 {get_text('tips')}")
    st.markdown(
        """
- Try different Prompt IDs (**Daily**, **pmpt_69A**, **pmpt_69B**) to see how behavior changes.
- Use **Notes Studio** to turn rough notes into clean Markdown, then highlight key terms for quick review.
- **Attachment Chat** helps you quickly understand long documents – upload and ask direct questions.
- Switch **Light/Dark** and experiment with the **20 flower themes** to find a look you enjoy for long sessions.
"""
    )
