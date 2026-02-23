import hashlib
import json
import re
import os
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# =========================
# Page configuration
# =========================
st.set_page_config(page_title="Literature searching assistant", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #fcfcfc;
    }

    .main-title {
        text-align: center;
        color: #1E3A8A; 
        font-weight: 800;
        padding-top: 10px;
        padding-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #64748B;
        margin-bottom: 30px;
        font-size: 1.1rem;
    }

    button[kind="primary"], div.stButton > button[data-testid="stFormSubmitButton"] {
        background-color: #1E3A8A !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    button[kind="primary"]:hover, div.stButton > button[data-testid="stFormSubmitButton"]:hover {
        background-color: #2563EB !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }

    div.stButton > button {
        border-radius: 8px;
    }

    div[data-testid="stVerticalBlock"] > div.element-container div.stMarkdown div.stContainer {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
        margin-bottom: 20px;
    }

    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Literature Searching Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by Semantic Scholar and Gemini AI</p>', unsafe_allow_html=True)

# =========================
# Secrets
# =========================
def get_secret(name: str) -> str:
    try:
        v = st.secrets.get(name, "")
        if v:
            return v
    except Exception:
        pass
    return os.getenv(name, "")

S2_KEY = get_secret("SEMANTIC_SCHOLAR_API_KEY")
GEMINI_KEY = get_secret("GEMINI_API_KEY")

# =========================
# Helper functions
# =========================
def md5_key(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

def format_authors(authors: List[Dict[str, Any]], max_n: int = 6) -> str:
    names = [a.get("name", "") for a in (authors or []) if a.get("name")]
    if len(names) > max_n:
        return ", ".join(names[:max_n]) + ", et al."
    return ", ".join(names) if names else "Unknown authors"

def paper_key(p: Dict[str, Any]) -> str:
    ext = p.get("externalIds") or {}
    return p.get("paperId") or ext.get("DOI") or p.get("title", "paper")

def clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# =========================
# Logging
# =========================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

EVENT_LOG_PATH = LOG_DIR / "event_log.jsonl"
SESSION_SUMMARY_PATH = LOG_DIR / "session_summary.jsonl"

def ensure_state():
    defaults = {
        # Core app state
        "saved": [],
        "saved_ids": set(),
        "last_query": "",
        "last_results": [],
        "chat_messages": [],

        # Experiment/session state
        "participant_id": "",
        "assigned_group": "A",   # A=no chatbot, B=chatbot
        "session_id": str(uuid.uuid4())[:8],
        "task_started": False,
        "task_start_ts": None,
        "result_page_enter_ts": None,

        # Logging counters
        "ai_summary_view_count": 0,
        "chat_question_count": 0,
        "chat_response_count": 0,
        "paper_click_count": 0,

        # UI toggle state for abstracts
        "visible_abstracts": set(),

        # One-time log guard
        "page_load_logged": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

ensure_state()

def task_duration_sec() -> Optional[float]:
    ts = st.session_state.get("task_start_ts")
    if ts is None:
        return None
    return round(time.time() - ts, 2)

def result_page_dwell_sec() -> Optional[float]:
    ts = st.session_state.get("result_page_enter_ts")
    if ts is None:
        return None
    return round(time.time() - ts, 2)

def json_safe(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, list):
        return [json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): json_safe(val) for k, val in v.items()}
    if isinstance(v, set):
        return [json_safe(x) for x in v]
    return str(v)

def base_log_record(event_type: str) -> Dict[str, Any]:
    return {
        "timestamp_utc": utc_now_iso(),
        "event_type": event_type,
        "participant_id": st.session_state.get("participant_id", ""),
        "assigned_group": st.session_state.get("assigned_group", ""),
        "session_id": st.session_state.get("session_id", ""),
        "search_queries_submitted_count": None,  # optional aggregate if you later add counter
        "paper_click_count": st.session_state.get("paper_click_count", 0),
        "time_spent_on_result_page_sec": result_page_dwell_sec(),
        "ai_summary_viewed_count": st.session_state.get("ai_summary_view_count", 0),
        "chatbot_question_count": st.session_state.get("chat_question_count", 0),
        "chatbot_response_count": st.session_state.get("chat_response_count", 0),
        "task_duration_sec": task_duration_sec(),
    }

def log_event(event_type: str, **kwargs):
    rec = base_log_record(event_type)
    for k, v in kwargs.items():
        rec[k] = json_safe(v)
    with EVENT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def log_session_summary(note: str = ""):
    rec = base_log_record("session_summary")
    rec["note"] = note
    rec["final_last_query"] = st.session_state.get("last_query", "")
    rec["final_results_count"] = len(st.session_state.get("last_results", []))
    rec["saved_papers_count"] = len(st.session_state.get("saved", []))
    with SESSION_SUMMARY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if not st.session_state.page_load_logged:
    log_event("page_load")
    st.session_state.page_load_logged = True

# =========================
# Sidebar (settings + experiment controls)
# =========================
with st.sidebar:
    st.header("Settings")
    st.caption("Configuration from `.streamlit/secrets.toml`")
    if bool(GEMINI_KEY):
        st.success("Gemini API: Connected")
    else:
        st.error("Gemini API: Missing")
    if bool(S2_KEY):
        st.success("Semantic Scholar API: Connected")
    else:
        st.warning("S2 API: Using Rate-Limited Public Access")

    st.divider()
    st.subheader("Experiment Setup")

    pid_input = st.text_input(
        "Participant ID",
        value=st.session_state.participant_id,
        placeholder="e.g., P001"
    )

    group_input = st.selectbox(
        "Assigned Group",
        ["A", "B"],
        index=0 if st.session_state.assigned_group == "A" else 1,
        help="A = no chatbot, B = chatbot enabled"
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start Task"):
            st.session_state.participant_id = clean_ws(pid_input) or f"P_{st.session_state.session_id}"
            st.session_state.assigned_group = group_input
            st.session_state.task_started = True
            st.session_state.task_start_ts = time.time()
            st.session_state.result_page_enter_ts = None
            st.session_state.ai_summary_view_count = 0
            st.session_state.chat_question_count = 0
            st.session_state.chat_response_count = 0
            st.session_state.paper_click_count = 0
            st.session_state.chat_messages = []
            log_event("task_start")
            st.success("Task started and logging reset.")
    with c2:
        if st.button("End Task"):
            log_event("task_end")
            log_session_summary(note="Manual end task")
            st.success("Task summary logged.")

    st.caption(
        f"Session: {st.session_state.session_id} | "
        f"Group: {st.session_state.assigned_group}"
    )
    st.caption(
        f"Task duration: {task_duration_sec()} sec | "
        f"Result page time: {result_page_dwell_sec()} sec"
    )

# =========================
# S2 API
# =========================
S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "paperId,title,abstract,year,venue,authors,citationCount,url,openAccessPdf,externalIds"

@st.cache_data(ttl=3600, show_spinner=False)
def s2_search_papers(query: str, api_key: str, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
    url = f"{S2_BASE}/paper/search"
    params = {"query": query, "limit": limit, "offset": offset, "fields": S2_FIELDS}
    headers = {"x-api-key": api_key} if api_key else {}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"S2 error {r.status_code}: {r.text[:400]}")
    return r.json()

# =========================
# Gemini API
# =========================
GEMINI_ENDPOINT_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"

def _extract_gemini_text(resp_json: Dict[str, Any]) -> str:
    try:
        parts = resp_json["candidates"][0]["content"]["parts"]
        return "\n".join([p["text"] for p in parts if "text" in p]).strip()
    except Exception:
        return json.dumps(resp_json)[:1500]

def gemini_generate_text(contents, api_key, model, system_instruction=None, temperature=0.4, max_output_tokens=1500):
    if not api_key:
        return {"ok": False, "raw_text": "", "error": "Missing GEMINI_API_KEY"}
    url = GEMINI_ENDPOINT_TMPL.format(model=model, key=api_key)
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens
        }
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=90)
    if r.status_code != 200:
        return {"ok": False, "raw_text": r.text[:900], "error": f"Gemini HTTP {r.status_code}"}
    return {"ok": True, "raw_text": _extract_gemini_text(r.json()), "error": None}

def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    return None

# =========================
# Visualization
# =========================
def render_visualizations(papers: List[Dict[str, Any]]):
    if not papers:
        return
    df = pd.DataFrame(papers)

    if "year" in df.columns:
        st.write("#### Result Insights")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.caption("Publication Trend (Yearly)")
            year_counts = df["year"].dropna().astype(int).value_counts().sort_index()
            st.bar_chart(year_counts, color="#3358BE")
        with c2:
            st.caption("Top Publication Venues")
            venue_counts = df["venue"].replace("", "Unknown").value_counts().head(5)
            st.dataframe(venue_counts, use_container_width=True)

# =========================
# Context for synthesis/chat
# =========================
def build_papers_context(papers, max_papers=None, include_abstract=True, abstract_char_limit=650):
    chosen = papers if max_papers is None else papers[:max_papers]
    blocks, labels = [], []
    for i, p in enumerate(chosen, start=1):
        label = f"P{i}"
        labels.append(label)
        abstract = clean_ws(p.get("abstract", ""))
        if include_abstract:
            abstract = abstract[:abstract_char_limit] + ("..." if len(abstract) > abstract_char_limit else "")
        else:
            abstract = ""
        blocks.append(
            f"[{label}] {p.get('title')}\n"
            f"Year: {p.get('year')} | Venue: {p.get('venue')}\n"
            f"Authors: {format_authors(p.get('authors'))}\n"
            f"Abstract: {abstract}"
        )
    return "\n---\n".join(blocks).strip(), labels

def make_synthesis_prompt(query: str, context: str) -> str:
    return f"""
User query: {query}

Analyze the provided papers and provide:
1) 5–10 sentence synthesis (cite like [P1]).
2) 8–12 research keywords.
3) 3 refined search queries.
4) Inconsistencies or Controversies: Identify conflicting findings, differing methodologies, or unresolved debates among these papers.

Return ONLY JSON with keys: summary, keywords, refined_queries, controversies.

PAPERS:
{context}
""".strip()

# =========================
# Search
# =========================
st.subheader("Search")
with st.form("search_form", clear_on_submit=False):
    col1, col2, col3 = st.columns([6, 1.4, 1.4])
    query = col1.text_input(
        "Keyword query",
        value=st.session_state.last_query,
        placeholder="e.g., impact of AI on management"
    )
    limit = col2.selectbox("Results", [5, 10, 20, 30, 50], index=1)
    page = col3.number_input("Page", min_value=1, value=1)
    submitted = st.form_submit_button("Search")

if submitted:
    if not query.strip():
        st.warning("Please enter a query.")
        log_event("search_submit_blocked", reason="empty_query")
    else:
        with st.spinner("Searching..."):
            try:
                results = s2_search_papers(query, S2_KEY, limit=int(limit), offset=(int(page)-1)*int(limit))
                st.session_state.last_query = query
                st.session_state.last_results = results.get("data", [])
                st.session_state.result_page_enter_ts = time.time()  # start dwell time on result page
                log_event(
                    "search_submit",
                    query=query,
                    limit=int(limit),
                    page=int(page),
                    n_results=len(st.session_state.last_results)
                )
            except Exception as e:
                st.error(f"Error: {e}")
                log_event("search_error", query=query, error=str(e))

# =========================
# AI analysis (visualizations)
# =========================
if st.session_state.last_results:
    with st.expander("View Research Trends", expanded=True):
        render_visualizations(st.session_state.last_results)

# =========================
# AI Insights (summary synthesis)
# =========================
st.subheader("AI Insights")
ai_cols = st.columns([3, 3, 4])
target = ai_cols[0].selectbox("Source set", ["Current page results", "Saved papers"])
model_ai = ai_cols[1].text_input("Gemini model", value=DEFAULT_GEMINI_MODEL)

run_ai = st.button("Generate Synthesis!", type="primary", disabled=not bool(GEMINI_KEY))

if run_ai:
    papers = st.session_state.last_results if target == "Current page results" else st.session_state.saved
    if not papers:
        st.warning("No papers to summarize.")
        log_event("ai_summary_blocked", reason="no_papers", source_set=target)
    else:
        ctx, _ = build_papers_context(papers, max_papers=None)
        prompt = make_synthesis_prompt(st.session_state.last_query, ctx)

        log_event("ai_summary_requested", source_set=target, n_papers=len(papers), model=model_ai)

        with st.spinner(f"Gemini is synthesizing {len(papers)} papers..."):
            res = gemini_generate_text(
                [{"role": "user", "parts": [{"text": prompt}]}],
                GEMINI_KEY,
                model_ai,
                system_instruction="Output valid JSON only."
            )
            if res["ok"]:
                parsed = try_parse_json(res["raw_text"])
                if parsed:
                    st.session_state.ai_summary_view_count += 1  # log summary viewed
                    log_event(
                        "ai_summary_viewed",
                        source_set=target,
                        n_papers=len(papers),
                        summary_chars=len(parsed.get("summary", "")),
                        controversies_chars=len(str(parsed.get("controversies", "")))
                    )

                    st.info("### Synthesis\n" + parsed.get("summary", ""))

                    st.markdown("### Controversies & Inconsistencies")
                    st.warning(parsed.get("controversies", "No major inconsistencies identified in this subset."))

                    st.write("**Keywords:** " + ", ".join(parsed.get("keywords", [])))
                    st.write("**Refined Queries:**")
                    for q in parsed.get("refined_queries", []):
                        st.write(f"- {q}")
                else:
                    st.error("Gemini output could not be parsed as JSON.")
                    st.code(res["raw_text"][:1000])
                    log_event("ai_summary_parse_error", raw_preview=res["raw_text"][:300])
            else:
                st.error(f"Gemini error: {res['error']}")
                log_event("ai_summary_error", error=res["error"], raw_preview=res["raw_text"][:300])

# =========================
# Result list (paper interactions)
# =========================
st.divider()
if st.session_state.last_results:
    for p in st.session_state.last_results:
        pid = paper_key(p)
        pid_hash = md5_key(str(pid))
        abs_toggle_key = f"show_abs_{pid_hash}"

        if abs_toggle_key not in st.session_state:
            st.session_state[abs_toggle_key] = False

        with st.container():
            c1, c2, c3, c4 = st.columns([8, 1.2, 1.2, 1.4])

            c1.markdown(f"### {p.get('title')}")
            c1.write(f"**{format_authors(p.get('authors'))}** | {p.get('year')} • {p.get('venue')}")

            # Save button
            if c2.button("Save", key=f"s_{pid_hash}", disabled=pid in st.session_state.saved_ids):
                st.session_state.saved.append(p)
                st.session_state.saved_ids.add(pid)
                st.toast("Saved!")
                log_event(
                    "paper_saved",
                    paper_id=pid,
                    paper_title=p.get("title", ""),
                    year=p.get("year"),
                    venue=p.get("venue", "")
                )

            # "View" button (logs paper click)
            if c3.button("View", key=f"v_{pid_hash}"):
                st.session_state[abs_toggle_key] = not st.session_state[abs_toggle_key]
                st.session_state.paper_click_count += 1
                log_event(
                    "paper_clicked",
                    paper_id=pid,
                    paper_title=p.get("title", ""),
                    year=p.get("year"),
                    venue=p.get("venue", ""),
                    action="toggle_abstract"
                )

            # PDF link (cannot directly capture click event in Streamlit)
            if p.get("openAccessPdf"):
                c4.link_button("PDF", p["openAccessPdf"]["url"])

            if st.session_state[abs_toggle_key]:
                st.write(p.get("abstract") or "No abstract available.")

# =========================
# Chat with Gemini (Group B only)
# =========================
st.divider()
st.subheader("Chat with Results?")

chat_enabled = st.session_state.assigned_group == "B"

if not chat_enabled:
    st.info("Chatbot is disabled in Group A.")
else:
    if st.session_state.last_results or st.session_state.saved:
        # Render previous chat history
        for msg in st.session_state.chat_messages:
            role = "assistant" if msg["role"] == "model" else "user"
            with st.chat_message(role):
                st.write(msg["text"])

        user_input = st.chat_input("Ask a question about these results...")
        if user_input:
            st.session_state.chat_messages.append({"role": "user", "text": user_input})
            st.session_state.chat_question_count += 1
            log_event(
                "chat_question",
                query_text=user_input,
                source_set=target,
                model=model_ai
            )

            with st.chat_message("user"):
                st.write(user_input)

            papers_chat = st.session_state.last_results if target == "Current page results" else st.session_state.saved
            ctx_chat, _ = build_papers_context(papers_chat, max_papers=10)  # cap context for reliability

            full_prompt = f"CONTEXT:\n{ctx_chat}\n\nQUESTION: {user_input}"

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    res = gemini_generate_text(
                        [{"role": "user", "parts": [{"text": full_prompt}]}],
                        GEMINI_KEY,
                        model_ai,
                        system_instruction=(
                            "Academic assistant. Answer ONLY using the provided CONTEXT. "
                            "Do not use outside knowledge. If the answer is not in the context, "
                            "say 'I cannot answer from the retrieved papers.' "
                            "Use [P1]-style citations where possible. Be concise."
                        ),
                    )
                    if res["ok"]:
                        st.write(res["raw_text"])
                        st.session_state.chat_messages.append({"role": "model", "text": res["raw_text"]})
                        st.session_state.chat_response_count += 1
                        log_event(
                            "chat_response",
                            response_chars=len(res["raw_text"]),
                            ok=True
                        )
                    else:
                        st.error(f"Gemini error: {res['error']}")
                        log_event(
                            "chat_response_error",
                            ok=False,
                            error=res["error"],
                            raw_preview=res.get("raw_text", "")[:300]
                        )
    else:
        st.caption("Search or save papers first to enable chat.")

# =========================
# Footer panel for quick logging summary
# =========================
st.divider()
st.subheader("Logging Summary (Current Session)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Paper Clicks", st.session_state.paper_click_count)
m2.metric("AI Summaries Viewed", st.session_state.ai_summary_view_count)
m3.metric("Chat Qs", st.session_state.chat_question_count)
m4.metric("Chat Responses", st.session_state.chat_response_count)

st.caption(
    f"Task duration: {task_duration_sec()} sec | "
    f"Result-page dwell time: {result_page_dwell_sec()} sec | "
    f"Participant ID: {st.session_state.participant_id or '(not set)'} | "
    f"Group: {st.session_state.assigned_group}"
)