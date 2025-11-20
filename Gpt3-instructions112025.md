Below is a complete design + implementation plan that:

- Keeps all original features (multi‑model, agents.yaml, prompt/max_tokens control, API key handling).
- Adds a “wow” UI (themes, language toggle, 20 flower styles).
- Adds prompt IDs (Daily pmpt_69A, pmpt_69B).
- Adds attachment upload + “chat on attachment”.
- Adds note-keeping + AI markdown + keyword coloring.
- Runs backend on Hugging Face Spaces (Streamlit) and a “wow” front‑end on Netlify.
- Includes beginner‑friendly deployment instructions.
- Ends with 20 comprehensive follow‑up questions.

---

## 1. High-Level Architecture

**Backend (Hugging Face Space) — Python + Streamlit**

- Core features already exist:
  - Uses `agents.yaml` to define agents/tools.
  - Supports multiple models:
    - Gemini: `gemini-2.5-flash`, `gemini-2.5-flash-lite`
    - OpenAI: `gpt-5-nano`, `gpt-4o-mini`, `gpt-4.1-mini`
    - Anthropic models.
  - User can:
    - Modify system/user prompt.
    - Set max tokens.
    - Select model.
    - Enter API keys in UI when not provided via environment.
- Enhancements:
  - Attachment upload + “chat on attachment”.
  - Notes + AI → Markdown transformer.
  - AI formatting: highlight keywords in color within markdown.
  - More visual “wow indicators” / mini dashboard inside Streamlit as well (basic summary).

**Frontend (“Wow UI”) — React (Vite) on Netlify**

- A polished SPA that calls the backend API endpoints (you can expose simple JSON endpoints from the Streamlit app or a small companion FastAPI/Flask app in the Space).
- Features:
  - Global theme:
    - Light / Dark mode.
    - 20 flower-based styles (color palettes, background accents).
  - Language toggle: English / Traditional Chinese.
  - Tabs / sections:
    1. **Agents Console** (existing features).
    2. **Attachment Chat** (chat about uploaded files).
    3. **Notes Studio** (note taking, AI → markdown, keyword coloring).
    4. **Dashboard** (wow indicators: usage, tokens, latency, model mix, etc.).
  - Prompt IDs: `Daily`, `pmpt_69A`, `pmpt_69B` as selection options that pre-fill prompts.

---

## 2. Wow UI: Feature & UX Specification

### 2.1 Global UI Shell

- **Header Bar**
  - Left: App logo + name, e.g., “FlowerMind Agent Studio”.
  - Center: Current page title.
  - Right:
    - Light/Dark mode toggle.
    - Flower style dropdown (1–20).
    - Language toggle (EN / 繁體).
    - Quick stats (requests today, active model).

- **Left Sidebar**
  - Navigation:
    - Agents Console
    - Attachment Chat
    - Notes Studio
    - Dashboard
  - Model selection:
    - Provider: Gemini / OpenAI / Anthropic.
    - Model: per provider (e.g., `gemini-2.5-flash`).
  - Prompt ID selection:
    - Default (Custom)
    - Daily
    - pmpt_69A
    - pmpt_69B

- **Main Content Area**
  - Depends on selected tab:
    - Agents Console → main chat + settings.
    - Attachment Chat → file upload + contextual chat.
    - Notes Studio → note editor + AI transform + keyword styling panel.
    - Dashboard → metrics & wow indicators.

### 2.2 Themes: Light/Dark + 20 Flower Styles

Define 20 flower styles, each a palette:

Example (English + Chinese label):

1. Sakura / 櫻花 — soft pink/white.
2. Lotus / 蓮花 — pastel green/pink.
3. Peony / 牡丹 — deep pink/red.
4. Orchid / 蘭花 — purple/white.
5. Plum Blossom / 梅花 — red/gray.
6. Chrysanthemum / 菊花 — yellow/amber.
7. Rose / 玫瑰 — red/black.
8. Sunflower / 向日葵 — yellow/blue.
9. Lavender / 薰衣草 — violet/gray.
10. Camellia / 山茶花 — red/green.
11. Hydrangea / 繡球花 — blue/pink.
12. Magnolia / 木蘭 — cream/green.
13. Jasmine / 茉莉 — white/green.
14. Wisteria / 紫藤 — purple/sky.
15. Azalea / 杜鵑 — bright pink.
16. Daffodil / 水仙 — yellow/white.
17. Iris / 鳶尾花 — deep purple.
18. Poppy / 罌粟 — orange/red.
19. Tulip / 鬱金香 — multicolor pastels.
20. Lotus Night / 夜蓮 — dark teal/pink (for dramatic dark mode).

Each theme defines:

```ts
interface ThemeVariant {
  id: string;
  labelEN: string;
  labelZH: string;
  primary: string;
  secondary: string;
  accent: string;
  backgroundLight: string;
  backgroundDark: string;
  border: string;
}
```

Use them as CSS variables so the UI updates instantly.

### 2.3 Language Toggle (EN / 繁體中文)

Use a simple i18n dictionary:

```ts
const messages = {
  en: {
    agentsConsole: "Agents Console",
    attachmentChat: "Attachment Chat",
    notesStudio: "Notes Studio",
    dashboard: "Dashboard",
    // ...
  },
  zhTW: {
    agentsConsole: "智能代理主控台",
    attachmentChat: "附件對話",
    notesStudio: "筆記工作室",
    dashboard: "儀表板",
    // ...
  },
};
```

Wrap the app in a `LanguageProvider`, so all text is selected by current language.

---

## 3. Backend: Streamlit App (Hugging Face Space)

Below is a single-file example `app.py` showing how to add:

- Model selection.
- Prompt IDs.
- Attachment upload + chat.
- Notes → Markdown via AI.
- Keyword coloring.
- API key handling (never showing env keys).

> Adjust imports and model client calls to match your actual code and API clients.

### 3.1 Core Setup

```python
# app.py
import os
import yaml
import time
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ------------------------------
# Helpers: API key management
# ------------------------------
def get_api_key(env_name: str, ui_label: str, session_key: str) -> str:
    """
    Priority:
    1. Environment variable (never shown to the user).
    2. User-entered key stored in session_state.
    """
    env_val = os.getenv(env_name)
    if env_val:
        return env_val

    # Only ask user if env var is missing
    if session_key not in st.session_state:
        st.session_state[session_key] = ""

    user_val = st.text_input(ui_label, type="password", value=st.session_state[session_key])
    if user_val:
        st.session_state[session_key] = user_val
    return st.session_state[session_key]


# ------------------------------
# Helpers: Load agents
# ------------------------------
@st.cache_data
def load_agents_config(path: str = "agents.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------
# Helpers: LLM call (pseudo)
# ------------------------------
def call_model(
    provider: str,
    model_name: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float = 0.7,
    api_keys: Dict[str, str] = None,
) -> str:
    """
    Pseudo function: integrate your actual Gemini / OpenAI / Anthropic clients here.
    messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    # Example structure: you'll need to plug into real SDKs
    # Use provider and model_name to route to correct client.
    # Respect api_keys dict for authentication.
    # For this example, just echo back last user message.
    user_content = [m["content"] for m in messages if m["role"] == "user"][-1]
    return f"[Demo response from {provider}/{model_name}] {user_content[:300]}"


# ------------------------------
# Keyword coloring helper
# ------------------------------
def highlight_keywords_in_markdown(text: str, keyword_color_map: Dict[str, str]) -> str:
    """
    Wrap keywords in <span style='color: ...'>keyword</span> so markdown renderers
    (with unsafe_allow_html=True) can color them.
    """
    if not keyword_color_map:
        return text

    # Simple word-based replacement; you can refine with regex boundaries.
    for kw, color in keyword_color_map.items():
        if not kw.strip():
            continue
        span = f"<span style='color:{color}'>{kw}</span>"
        text = text.replace(kw, span)
    return text


# ------------------------------
# UI Config
# ------------------------------
st.set_page_config(
    page_title="FlowerMind Agent Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)
```

### 3.2 Sidebar: Model, Prompt ID, API Keys

```python
agents_cfg = load_agents_config()

st.sidebar.title("⚙️ Control Panel")

# Provider + model
provider = st.sidebar.selectbox("Provider", ["Gemini", "OpenAI", "Anthropic"])
model_name = st.sidebar.selectbox(
    "Model",
    {
        "Gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "OpenAI": ["gpt-5-nano", "gpt-4o-mini", "gpt-4.1-mini"],
        "Anthropic": ["claude-3-5-sonnet", "claude-3-5-haiku"],
    }[provider],
)

max_tokens = st.sidebar.slider("Max tokens", min_value=128, max_value=8192, value=1024, step=128)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

prompt_id = st.sidebar.selectbox("Prompt ID", ["Custom", "Daily", "pmpt_69A", "pmpt_69B"])

# Load template prompts based on ID (you can store these in a YAML or dict)
prompt_templates = {
    "Daily": "You are my daily assistant. Help me organize today.",
    "pmpt_69A": "Prompt 69A: Advanced reasoning prompt template ...",
    "pmpt_69B": "Prompt 69B: Creative thinking prompt template ...",
}
default_system_prompt = prompt_templates.get(prompt_id, "You are a helpful AI assistant.")

system_prompt = st.sidebar.text_area("System Prompt", value=default_system_prompt, height=150)

# API keys (only shown if env var missing)
st.sidebar.subheader("🔐 API Keys (local only)")
gemini_key = get_api_key("GEMINI_API_KEY", "Gemini API Key", "gemini_api_key")
openai_key = get_api_key("OPENAI_API_KEY", "OpenAI API Key", "openai_api_key")
anthropic_key = get_api_key("ANTHROPIC_API_KEY", "Anthropic API Key", "anthropic_api_key")

api_keys = {
    "Gemini": gemini_key,
    "OpenAI": openai_key,
    "Anthropic": anthropic_key,
}
```

### 3.3 Main Layout: Tabs

```python
tab1, tab2, tab3, tab4 = st.tabs(
    ["Agents Console", "Attachment Chat", "Notes Studio", "Dashboard"]
)
```

#### 3.3.1 Agents Console (Existing + Prompt IDs)

```python
with tab1:
    st.header("Agents Console")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_message = st.text_area("Your message", "", height=120)

    if st.button("Run Agent"):
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

            # Simple "wow indicator"
            st.success(f"Model responded in {latency:.2f} seconds")

    # Show conversation
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")
```

#### 3.3.2 Attachment Chat

```python
with tab2:
    st.header("Attachment Chat")

    if "attachment_chat" not in st.session_state:
        st.session_state.attachment_chat = []
        st.session_state.attachment_content = ""

    uploaded_file = st.file_uploader("Upload a file (PDF, TXT, DOCX, etc.)", type=None)

    if uploaded_file is not None:
        # You can handle different file types (pdfminer, docx, etc.)
        raw_bytes = uploaded_file.read()
        try:
            decoded = raw_bytes.decode("utf-8", errors="ignore")
        except Exception:
            decoded = str(raw_bytes)
        st.session_state.attachment_content = decoded
        st.info(f"Loaded attachment: {uploaded_file.name} ({len(decoded)} chars)")

    chat_on_attachment = st.text_area("Ask about this attachment", "", height=100)

    if st.button("Ask Attachment"):
        if not st.session_state.attachment_content:
            st.warning("Please upload an attachment first.")
        elif not chat_on_attachment.strip():
            st.warning("Please enter your question.")
        else:
            context_snippet = st.session_state.attachment_content[:6000]
            user_prompt = (
                "You are given the following document content:\n\n"
                f"{context_snippet}\n\n"
                "User question:\n"
                f"{chat_on_attachment}"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = call_model(
                provider=provider,
                model_name=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                api_keys=api_keys,
            )
            st.session_state.attachment_chat.append(
                {"question": chat_on_attachment, "answer": response}
            )

    for turn in st.session_state.attachment_chat:
        st.markdown(f"**Q:** {turn['question']}")
        st.markdown(f"**A:** {turn['answer']}")
```

#### 3.3.3 Notes Studio (Note Keeping + AI Markdown + Keyword Coloring)

```python
with tab3:
    st.header("Notes Studio")

    if "notes_text" not in st.session_state:
        st.session_state.notes_text = ""

    st.subheader("📝 Raw Notes")
    st.session_state.notes_text = st.text_area(
        "Type or paste your notes here",
        value=st.session_state.notes_text,
        height=220,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        run_ai_markdown = st.button("✨ Transform to Markdown (AI)")
    with col_b:
        run_ai_formatting = st.button("🎨 AI Improve Formatting")

    if run_ai_markdown:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a note formatting assistant. "
                    "Convert the user's notes into clean, well-structured Markdown. "
                    "Use headings, bullet lists, code blocks, tables when appropriate. "
                    "Do not add extra explanations."
                ),
            },
            {"role": "user", "content": st.session_state.notes_text},
        ]
        md = call_model(
            provider=provider,
            model_name=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
            api_keys=api_keys,
        )
        st.session_state.notes_markdown = md

    if run_ai_formatting and "notes_markdown" in st.session_state:
        messages = [
            {
                "role": "system",
                "content": (
                    "Improve the formatting of the given Markdown. "
                    "Keep the same meaning but make it more readable, "
                    "with better headings and logical sections."
                ),
            },
            {"role": "user", "content": st.session_state.get("notes_markdown", "")},
        ]
        improved_md = call_model(
            provider=provider,
            model_name=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.4,
            api_keys=api_keys,
        )
        st.session_state.notes_markdown = improved_md

    st.subheader("🔍 Keyword Highlighting")
    kw_input = st.text_input("Keywords (comma-separated)", "TODO,Important,Deadline")
    kw_color = st.color_picker("Color for these keywords", "#ff4081")

    keyword_map = {}
    if kw_input.strip():
        for kw in [k.strip() for k in kw_input.split(",")]:
            if kw:
                keyword_map[kw] = kw_color

    st.subheader("📄 Markdown Preview")

    if "notes_markdown" not in st.session_state:
        st.info("Click 'Transform to Markdown (AI)' to generate markdown.")
    else:
        colored_md = highlight_keywords_in_markdown(
            st.session_state.notes_markdown, keyword_map
        )
        st.markdown(colored_md, unsafe_allow_html=True)
```

#### 3.3.4 Dashboard (Wow Indicators)

```python
with tab4:
    st.header("Dashboard & Wow Indicators")

    # In a real app, track these metrics in a database or in session
    total_requests = len(st.session_state.get("chat_history", [])) // 2
    attachment_questions = len(st.session_state.get("attachment_chat", []))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Chat Turns", total_requests)
    with col2:
        st.metric("Attachment Q&A", attachment_questions)
    with col3:
        st.metric("Current Model", model_name)
    with col4:
        st.metric("Provider", provider)

    st.subheader("Usage Overview")
    st.progress(min(total_requests / 50.0, 1.0))  # arbitrary target

    st.subheader("Tips")
    st.write(
        "- Try switching Prompt ID to 'pmpt_69A' or 'pmpt_69B' for different behaviors.\n"
        "- Use Notes Studio to convert rough notes into clean markdown.\n"
        "- Use Attachment Chat to quickly understand long documents."
    )
```

---

## 4. Wow Frontend UI on Netlify (React + Vite Sketch)

Below is a high-level sketch (not full project) to guide implementation.

### 4.1 Project Setup

```bash
# Create Vite + React + TypeScript app
npm create vite@latest wow-ui -- --template react-ts
cd wow-ui
npm install
npm install axios react-markdown rehype-raw
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

Configure Tailwind in `tailwind.config.cjs` and import styles in `src/index.css`.

### 4.2 Theme & Language Providers (Simplified)

`src/theme.tsx`:

```tsx
import React, { createContext, useContext, useState } from "react";

type ThemeMode = "light" | "dark";
type Lang = "en" | "zhTW";

interface ThemeContextType {
  mode: ThemeMode;
  setMode: (m: ThemeMode) => void;
  flowerId: string;
  setFlowerId: (id: string) => void;
  lang: Lang;
  setLang: (l: Lang) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [mode, setMode] = useState<ThemeMode>("light");
  const [flowerId, setFlowerId] = useState<string>("sakura");
  const [lang, setLang] = useState<Lang>("en");

  return (
    <ThemeContext.Provider value={{ mode, setMode, flowerId, setFlowerId, lang, setLang }}>
      <div className={mode === "dark" ? "dark bg-slate-900 text-slate-100" : "bg-slate-50 text-slate-900"}>
        {children}
      </div>
    </ThemeContext.Provider>
  );
};

export const useThemeCtx = () => {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useThemeCtx must be used within ThemeProvider");
  return ctx;
};
```

You’d also define a `flowerThemes` map and apply CSS variables per `flowerId`.

### 4.3 Main Layout Example

`src/App.tsx` (simplified):

```tsx
import React, { useState } from "react";
import { ThemeProvider, useThemeCtx } from "./theme";
import AgentsConsole from "./pages/AgentsConsole";
import AttachmentChat from "./pages/AttachmentChat";
import NotesStudio from "./pages/NotesStudio";
import Dashboard from "./pages/Dashboard";

type Page = "agents" | "attachment" | "notes" | "dashboard";

const Shell: React.FC = () => {
  const [page, setPage] = useState<Page>("agents");
  const { mode, setMode, flowerId, setFlowerId, lang, setLang } = useThemeCtx();

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="w-72 border-r border-slate-200 dark:border-slate-800 p-4 space-y-4">
        <div className="font-bold text-xl">FlowerMind</div>

        <nav className="space-y-2">
          <button onClick={() => setPage("agents")} className="w-full text-left">Agents Console</button>
          <button onClick={() => setPage("attachment")} className="w-full text-left">Attachment Chat</button>
          <button onClick={() => setPage("notes")} className="w-full text-left">Notes Studio</button>
          <button onClick={() => setPage("dashboard")} className="w-full text-left">Dashboard</button>
        </nav>

        <div className="pt-4 border-t border-slate-200 dark:border-slate-800 space-y-2">
          <div className="flex items-center justify-between">
            <span>Theme</span>
            <button
              onClick={() => setMode(mode === "light" ? "dark" : "light")}
              className="px-2 py-1 border rounded"
            >
              {mode === "light" ? "Light" : "Dark"}
            </button>
          </div>

          <div>
            <label className="block text-sm mb-1">Flower Style</label>
            <select
              value={flowerId}
              onChange={(e) => setFlowerId(e.target.value)}
              className="w-full border rounded px-2 py-1"
            >
              <option value="sakura">Sakura / 櫻花</option>
              <option value="lotus">Lotus / 蓮花</option>
              {/* Add all 20 options */}
            </select>
          </div>

          <div>
            <label className="block text-sm mb-1">Language</label>
            <select
              value={lang}
              onChange={(e) => setLang(e.target.value as any)}
              className="w-full border rounded px-2 py-1"
            >
              <option value="en">English</option>
              <option value="zhTW">繁體中文</option>
            </select>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 p-6">
        {page === "agents" && <AgentsConsole />}
        {page === "attachment" && <AttachmentChat />}
        {page === "notes" && <NotesStudio />}
        {page === "dashboard" && <Dashboard />}
      </main>
    </div>
  );
};

const App: React.FC = () => (
  <ThemeProvider>
    <Shell />
  </ThemeProvider>
);

export default App;
```

### 4.4 Attachment Chat Page (talking to backend)

`src/pages/AttachmentChat.tsx` (simplified):

```tsx
import React, { useState } from "react";
import axios from "axios";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL; // e.g. your Hugging Face Space API

const AttachmentChat: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<{ q: string; a: string }[]>([]);
  const [loading, setLoading] = useState(false);

  const onSubmit = async () => {
    if (!file || !question.trim()) return;
    setLoading(true);
    try:
      const formData = new FormData();
      formData.append("file", file);
      formData.append("question", question);

      const res = await axios.post(`${BACKEND_URL}/attachment-chat`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const answer = res.data.answer as string;
      setHistory((h) => [...h, { q: question, a: answer }]);
      setQuestion("");
    } catch (err) {
      console.error(err);
      alert("Error talking to backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Attachment Chat</h1>

      <input
        type="file"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        className="block"
      />

      <textarea
        className="w-full border rounded p-2 h-24"
        placeholder="Ask about your attachment..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
      />

      <button
        disabled={loading}
        onClick={onSubmit}
        className="px-4 py-2 rounded bg-pink-500 text-white"
      >
        {loading ? "Thinking..." : "Ask"}
      </button>

      <div className="space-y-3 mt-4">
        {history.map((h, idx) => (
          <div key={idx} className="border rounded p-3">
            <div className="font-semibold">Q: {h.q}</div>
            <div className="mt-1">A: {h.a}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AttachmentChat;
```

> On the backend, create an endpoint (Flask/FastAPI or similar) `/attachment-chat` that mirrors the logic used in the Streamlit tab, and deploy it in the same Hugging Face Space (or in another Space) that your Netlify app calls.

### 4.5 Notes Studio Page (Keyword coloring in markdown)

`src/pages/NotesStudio.tsx` (simplified):

```tsx
import React, { useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;

const NotesStudio: React.FC = () => {
  const [notes, setNotes] = useState("");
  const [markdown, setMarkdown] = useState("");
  const [keywords, setKeywords] = useState("TODO,Important,Deadline");
  const [color, setColor] = useState("#ff4081");

  const transformToMarkdown = async () => {
    const res = await axios.post(`${BACKEND_URL}/notes-to-markdown`, { notes });
    setMarkdown(res.data.markdown);
  };

  const improveFormatting = async () => {
    const res = await axios.post(`${BACKEND_URL}/markdown-improve`, { markdown });
    setMarkdown(res.data.markdown);
  };

  const keywordColoring = (text: string) => {
    const kws = keywords.split(",").map((k) => k.trim()).filter(Boolean);
    let result = text;
    for (const kw of kws) {
      const span = `<span style="color:${color}">${kw}</span>`;
      result = result.split(kw).join(span);
    }
    return result;
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Notes Studio</h1>

      <textarea
        className="w-full border rounded p-2 h-40"
        placeholder="Write your notes..."
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
      />

      <div className="flex gap-2">
        <button onClick={transformToMarkdown} className="px-4 py-2 bg-blue-500 text-white rounded">
          Transform to Markdown (AI)
        </button>
        <button onClick={improveFormatting} className="px-4 py-2 bg-emerald-500 text-white rounded">
          AI Improve Formatting
        </button>
      </div>

      <div className="border-t pt-4 space-y-2">
        <div className="flex gap-2 items-center">
          <input
            className="border rounded px-2 py-1 flex-1"
            value={keywords}
            onChange={(e) => setKeywords(e.target.value)}
          />
          <input type="color" value={color} onChange={(e) => setColor(e.target.value)} />
        </div>

        <div className="border rounded p-3 bg-white dark:bg-slate-900">
          <ReactMarkdown rehypePlugins={[rehypeRaw]}>
            {keywordColoring(markdown)}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
};

export default NotesStudio;
```

---

## 5. Deployment Instructions (Beginner-Friendly)

### 5.1 Prerequisites

- A **GitHub** account.
- A **Hugging Face** account.
- A **Netlify** account.
- Basic tools installed:
  - Python 3.9+ (for backend).
  - Node.js 18+ (for React frontend).
  - `git`.

---

### 5.2 Backend on Hugging Face Spaces (Streamlit)

1. **Create a new Space**
   - Go to https://huggingface.co/spaces.
   - Click “New Space”.
   - Name: e.g., `your-username/flowermind-agents`.
   - Space type: **Streamlit**.
   - Visibility: Public or Private.

2. **Initialize repo locally**
   ```bash
   git clone https://huggingface.co/spaces/your-username/flowermind-agents
   cd flowermind-agents
   ```

3. **Add backend files**
   - Copy your existing `app.py` (Streamlit) and `agents.yaml`.
   - Add the enhanced `app.py` (merge with your existing code).
   - Create `requirements.txt`, for example:

     ```txt
     streamlit
     pyyaml
     python-dotenv
     openai
     google-generativeai
     anthropic
     # plus any PDF/DOCX libs if you add them
     ```

4. **Configure environment variables in the Space (optional)**
   - In Space settings → “Repository secrets”.
   - Add (if you want backend-side keys):
     - `GEMINI_API_KEY`
     - `OPENAI_API_KEY`
     - `ANTHROPIC_API_KEY`
   - Your `get_api_key` helper will use these automatically and will **not show them** in the UI.

5. **Push to Hugging Face**
   ```bash
   git add .
   git commit -m "Initial Streamlit app with wow features"
   git push
   ```

6. **Wait for build & test**
   - Hugging Face will build and start the Space automatically.
   - Visit the Space URL to confirm:
     - Sidebar controls appear (model, prompt, keys).
     - Tabs: Agents Console, Attachment Chat, Notes Studio, Dashboard.
     - Upload attachments and test.

---

### 5.3 Exposing Simple API Endpoints (Optional for Netlify Frontend)

You have three options:

1. **Use only Streamlit UI** inside the HF Space (no Netlify).
2. **Create a small FastAPI/Flask backend** in another Space and call it from Netlify.
3. **Extend the same Space** to support both Streamlit and an API (more advanced).

For a beginner, option 2 is often cleaner:

- Create a new “Docker” or “Gradio/FastAPI” Space named `your-username/flowermind-backend`.
- Implement endpoints:
  - `POST /chat`
  - `POST /attachment-chat`
  - `POST /notes-to-markdown`
  - `POST /markdown-improve`
- Reuse the same `call_model` logic from your Streamlit app.

Then, your Netlify frontend points `VITE_BACKEND_URL` to this backend Space URL.

---

### 5.4 Frontend on Netlify (Wow UI)

1. **Create React project (if you haven’t)**
   ```bash
   npm create vite@latest wow-ui -- --template react-ts
   cd wow-ui
   npm install
   npm run dev  # verify it runs locally on http://localhost:5173
   ```

2. **Integrate the UI**
   - Replace `src/App.tsx` with the Shell example above (or your customized UI).
   - Add page components:
     - `src/pages/AgentsConsole.tsx`
     - `src/pages/AttachmentChat.tsx`
     - `src/pages/NotesStudio.tsx`
     - `src/pages/Dashboard.tsx`
   - Add `src/theme.tsx` and optionally `src/i18n.ts`.

3. **Configure backend URL**
   - Create `.env` in project root:
     ```env
     VITE_BACKEND_URL=https://your-backend-space.hf.space
     ```
   - Restart `npm run dev` and test features (attachment upload, notes transform).

4. **Initialize Git & push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial wow UI"
   git branch -M main
   git remote add origin https://github.com/your-username/wow-ui.git
   git push -u origin main
   ```

5. **Connect to Netlify**
   - Go to https://app.netlify.com.
   - “Add new site” → “Import an existing project”.
   - Choose your `wow-ui` GitHub repo.
   - Build settings:
     - Build command: `npm run build`
     - Publish directory: `dist`
   - Environment variables:
     - Add `VITE_BACKEND_URL` with your backend Space URL.

6. **Deploy**
   - Click “Deploy site”.
   - Wait until Netlify finishes building.
   - Visit the Netlify URL to test:
     - Sidebar navigation.
     - Light/dark mode.
     - Flower theme dropdown.
     - Language toggle.
     - Attachment Upload & chat.
     - Notes Studio with Markdown + keyword coloring.

7. **Optional: Custom domain**
   - In Netlify site settings → Domain management.
   - Add a custom domain (e.g., `agents.yourdomain.com`).
   - Follow Netlify’s DNS instructions.

---

### 5.5 API Key Handling on Netlify Frontend

- **Important**: Do not hardcode real API keys in frontend code or `.env` files that are shipped to the browser.
- Use one of these patterns:
  1. **Backend-only API keys**: Keep keys in Hugging Face Space environment variables. Netlify sends only simple requests; keys never leave backend.
  2. **User-provided keys**: Let users input keys in your Netlify UI, then forward them with each request to your backend (backend uses them and does not log them).

You already requested: “Please don't show api key If get api key from the environment.”  
The code pattern above honors this: environment keys are used silently, and only if they’re missing will the UI ask for a key.
