# streamlit_app.py
# 🔐 Energy Sector Cybersecurity Risk Mitigation Bot (RAG UI)

import json
import time
import uuid
import streamlit as st  # type: ignore[import]
from app.rag import generate_grounded_response

st.set_page_config(
    page_title="🔐 Energy Cybersecurity RAG",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# State
# =========================
def _init_state():
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None

    if "settings" not in st.session_state:
        st.session_state.settings = {
            "top_k": 8,
            "min_recurring_reviews": 2,
            "debug_on": False,
        }


def _new_chat():
    cid = str(uuid.uuid4())
    st.session_state.conversations[cid] = {
        "title": "New cybersecurity query",
        "messages": [],
        "created": time.time(),
    }
    st.session_state.active_chat_id = cid


def _ensure_active():
    if not st.session_state.conversations:
        _new_chat()
        return

    if st.session_state.active_chat_id not in st.session_state.conversations:
        newest = max(
            st.session_state.conversations.items(),
            key=lambda x: x[1]["created"]
        )[0]
        st.session_state.active_chat_id = newest


def _sorted():
    return sorted(
        st.session_state.conversations.items(),
        key=lambda x: x[1]["created"],
        reverse=True
    )


def _delete(cid):
    st.session_state.conversations.pop(cid, None)
    if not st.session_state.conversations:
        _new_chat()
    else:
        _ensure_active()


# =========================
# Init
# =========================
_init_state()
if not st.session_state.active_chat_id:
    _new_chat()
_ensure_active()

cid = st.session_state.active_chat_id
convo = st.session_state.conversations[cid]

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("🔐 Cybersecurity Chats")

    if st.button("➕ New analysis", use_container_width=True):
        _new_chat()
        st.rerun()

    st.divider()

    ids = [i[0] for i in _sorted()]

    selected = st.radio(
        "History",
        ids,
        index=ids.index(cid) if cid in ids else 0,
        format_func=lambda x: st.session_state.conversations[x]["title"],
    )
    st.session_state.active_chat_id = selected
    convo = st.session_state.conversations[selected]

    st.divider()

    with st.expander("⚙️ Settings"):
        st.session_state.settings["top_k"] = st.slider(
            "Top-K retrieval chunks",
            3, 30,
            st.session_state.settings["top_k"]
        )
        st.session_state.settings["debug_on"] = st.toggle(
            "Debug mode",
            st.session_state.settings["debug_on"]
        )

# =========================
# Title
# =========================
st.title("🔐 Energy Sector Cybersecurity Risk Mitigation Assistant")
st.caption(
    "Grounded RAG over NERC CIP, NIST CSF, DOE, CISA, OT/ICS security documents."
)

# =========================
# Render history
# =========================
for m in convo["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about cyber risk, controls, or compliance gaps...")

# =========================
# Rendering helpers
# =========================
def render_response(out: dict):

    summary = out.get("answer_summary") or out.get("summary") or ""
    if summary:
        st.subheader("Answer")
        st.write(summary)
        st.divider()

    points = out.get("key_points", [])
    if points:
        st.subheader("Key Points")
        for p in points:
            if isinstance(p, dict):
                text = p.get("point") or p.get("text") or str(p)
                st.markdown(f"- **{text}**")
                if p.get("source_chunk_ids"):
                    st.caption(f"Chunks: {p.get('source_chunk_ids', [])}")
            else:
                st.markdown(f"- **{p}**")

        st.divider()

    reqs = out.get("key_requirements", [])
    if reqs:
        st.subheader("Key Requirements")
        for r in reqs:
            if isinstance(r, dict):
                st.markdown(f"- **{r.get('requirement', '')}**")
                st.caption(f"Chunks: {r.get('evidence', [])}")
            else:
                st.markdown(f"- **{r}**")

        st.divider()

    recs = out.get("policy_recommendations", [])
    if recs:
        st.subheader("🛡️ Policy Recommendations")
        for r in recs:
            if isinstance(r, dict):
                st.markdown(f"**→ {r.get('recommendation', '')}**")
                st.write(r.get("justification", ""))
                st.caption(f"Chunks: {r.get('evidence', [])}")
            elif isinstance(r, str):
                st.markdown(f"**→ {r}**")
            else:
                st.write(r)

        st.divider()

    draft = out.get("draft_policy_language", [])
    if draft:
        st.subheader("📄 Draft Policy Language")
        for d in draft:
            if isinstance(d, dict):
                st.markdown(f"### {d.get('section', 'Section')}")
                st.code(d.get("text", ""))
            else:
                st.code(str(d))
        st.divider()

    st.subheader("Sources (chunk IDs)")
    sources = set()
    for r in out.get("key_requirements", []):
        if isinstance(r, dict):
            sources.update(r.get("evidence", []))

    for r in out.get("policy_recommendations", []):
        if isinstance(r, dict):
            sources.update(r.get("evidence", []))

    for p in out.get("key_points", []):
        if isinstance(p, dict):
            sources.update(p.get("source_chunk_ids", []))

    for s in out.get("sources", []):
        if isinstance(s, dict):
            chunk_id = s.get("chunk_id")
            if chunk_id:
                sources.add(chunk_id)
        elif isinstance(s, str):
            sources.add(s)

    if sources:
        st.code("\n".join(sorted(sources)))
    else:
        st.write("_No sources returned._")

    if st.session_state.settings["debug_on"]:
        with st.expander("Debug JSON"):
            st.code(json.dumps(out, indent=2))


# =========================
# Chat flow
# =========================
if prompt:
    convo["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving NERC/NIST/CISA evidence..."):
            out = generate_grounded_response(
                query=prompt,
                top_k=st.session_state.settings["top_k"],
                min_recurring_reviews=2,
                include_debug=st.session_state.settings["debug_on"],
            )

        if "error" in out:
            st.error(out["error"])
            st.code(out.get("raw_output", ""))
        else:
            render_response(out)

            convo["messages"].append({
                "role": "assistant",
                "content": out.get("answer_summary", "") or out.get("summary", "")
            })

    # -------------------------
    # General answer points
    # -------------------------
    points = out.get("key_points", [])
    if points:
        st.subheader("Key Points")
        for p in points:
            if isinstance(p, dict):
                text = p.get("point") or p.get("text") or str(p)
                st.markdown(f"- **{text}**")
                if p.get("source_chunk_ids"):
                    st.caption(f"Chunks: {p.get('source_chunk_ids', [])}")
            else:
                st.markdown(f"- **{p}**")

        st.divider()

    # -------------------------
    # Key requirements
    # -------------------------
    st.subheader("Key Requirements")
    reqs = out.get("key_requirements", [])
    if not reqs:
        st.write("_None found._")
    else:
        for r in reqs:
            if isinstance(r, dict):
                st.markdown(f"- **{r.get('requirement', '')}**")
                st.caption(f"Chunks: {r.get('source_chunk_ids', [])}")
            else:
                st.markdown(f"- **{r}**")

    # -------------------------
    # Recommendations
    # -------------------------
    st.subheader("🛡️ Policy Recommendations")
    recs = out.get("policy_recommendations", [])
    if not recs:
        st.write("_None found._")
    else:
        for r in recs:
            if isinstance(r, dict):
                st.markdown(f"**→ {r.get('recommendation', '')}**")
                st.write(r.get("justification", ""))
                st.caption(f"Chunks: {r.get('source_chunk_ids', [])}")

            elif isinstance(r, str):
                st.markdown(f"**→ {r}**")

            else:
                st.write(r)

    # -------------------------
    # Draft policy
    # -------------------------
    st.subheader("📄 Draft Policy Language")
    draft = out.get("draft_policy_language", [])
    if not draft:
        st.write("_No draft language generated._")
    else:
        for d in draft:
            if isinstance(d, dict):
                st.markdown(f"### {d.get('section', 'Section')}")
                st.code(d.get("text", ""))
            else:
                st.code(str(d))
    # -------------------------
    # Sources
    # -------------------------
    st.subheader("Sources (chunk IDs)")
    sources = set()
    for r in out.get("key_requirements", []):
        if isinstance(r, dict):
            sources.update(r.get("source_chunk_ids", []))

    for r in out.get("policy_recommendations", []):
        if isinstance(r, dict):
            sources.update(r.get("source_chunk_ids", []))

    for p in out.get("key_points", []):
        if isinstance(p, dict):
            sources.update(p.get("source_chunk_ids", []))

    for s in out.get("sources", []):
        if isinstance(s, dict):
            chunk_id = s.get("chunk_id")
            if chunk_id:
                sources.add(chunk_id)
        elif isinstance(s, str):
            sources.add(s)

    if sources:
        st.code("\n".join(sorted(sources)))
    else:
        st.write("_No sources returned._")

    # -------------------------
    # Debug
    # -------------------------
    if st.session_state.settings["debug_on"]:
        with st.expander("Debug JSON"):
            st.code(json.dumps(out, indent=2))


# =========================
# Chat flow
# =========================
if prompt:
    convo["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving NERC/NIST/CISA evidence..."):
            out = generate_grounded_response(
                query=prompt,
                top_k=st.session_state.settings["top_k"],
                min_recurring_reviews=2,
                include_debug=st.session_state.settings["debug_on"],
            )

        if "error" in out:
            st.error(out["error"])
            st.code(out.get("raw_output", ""))
        else:
            render_response(out)

            convo["messages"].append({
                "role": "assistant",
                "content": out.get("answer_summary", "")
            })
