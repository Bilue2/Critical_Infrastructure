import os
import json
import re
from typing import Any, Dict, List, Optional
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from control_families import detect_controls

load_dotenv()

EMBED_MODEL_DEFAULT = "text-embedding-3-small"


# =========================================================
# ENV
# =========================================================
def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


# =========================================================
# UTIL
# =========================================================
def _extract_json_object(text: str) -> str:
    t = (text or "").strip()

    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)

    first = t.find("{")
    last = t.rfind("}")

    if first != -1 and last != -1 and last > first:
        return t[first:last + 1]

    return t


def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj

    return {
        "id": getattr(obj, "id", None),
        "score": getattr(obj, "score", 0.0),
        "metadata": getattr(obj, "metadata", {}) or {},
    }


# =========================================================
# EMBEDDING
# =========================================================
def embed(client: OpenAI, text: str, model: str) -> List[float]:
    return client.embeddings.create(
        model=model,
        input=[text]
    ).data[0].embedding


# =========================================================
# QUERY INTENT
# =========================================================
def detect_organization(query: str) -> Optional[str]:
    orgs = {
        "nerc": "NERC",
        "nist": "NIST",
        "cisa": "CISA",
        "ferc": "FERC",
        "doe": "DOE",
    }

    q = query.lower()
    for k, v in orgs.items():
        if k in q:
            return v
    return None


def wants_workflow_output(query: str) -> bool:
    q = query.lower()
    triggers = [
        "recommend", "action plan", "fix", "improve",
        "next steps", "ops", "operations", "draft"
    ]
    return any(t in q for t in triggers)


# =========================================================
# RETRIEVAL
# =========================================================
def retrieve(
    query: str,
    top_k: int = 8,
    min_score: float = 0.65,
    exclude_owner_responses: bool = True,
) -> List[Dict[str, Any]]:

    client = OpenAI(api_key=_env("OPENAI_API_KEY"))
    pc = Pinecone(api_key=_env("PINECONE_API_KEY"))
    index = pc.Index(_env("PINECONE_INDEX"))

    qvec = embed(client, query, os.getenv("OPENAI_EMBED_MODEL", EMBED_MODEL_DEFAULT))

    filt = {}

    org = detect_organization(query)
    if org:
        filt["organization"] = {"$eq": org}

    if exclude_owner_responses:
        filt["is_owner_response"] = {"$ne": True}

    res = index.query(
    vector=qvec,
    top_k=top_k,
    include_metadata=True,
    include_values=False,
    filter=filt if filt else None,
    namespace=""
)

    matches = getattr(res, "matches", []) or []

    contexts = []

    for m in matches:
        md = _as_dict(m)
        meta = md.get("metadata", {}) or {}

        score = float(md.get("score", 0.0))
        if score < min_score:
            continue

        text = meta.get("text", "")

        controls, _ = detect_controls(text)

        contexts.append({
            "id": md.get("id"),
            "score": score,
            "chunk_id": meta.get("chunk_id"),
            "source_file": meta.get("source_file"),
            "document_title": meta.get("document_title"),
            "organization": meta.get("organization"),
            "document_type": meta.get("document_type"),
            "page_number": meta.get("page_number"),
            "text": text,
            "controls": controls,   # ✅ FIXED
        })

    # sort + dedupe
    seen = set()
    deduped = []

    for c in sorted(contexts, key=lambda x: x["score"], reverse=True):
        if c["chunk_id"] in seen:
            continue
        seen.add(c["chunk_id"])
        deduped.append(c)

    return deduped


# =========================================================
# AGGREGATION
# =========================================================
def aggregate_contexts(contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    controls = defaultdict(int)

    for c in contexts:
        for ctrl in c.get("controls", []) or []:
            controls[ctrl] += 1

    return {
        "control_families": sorted(
            [{"control": k, "count": v} for k, v in controls.items()],
            key=lambda x: x["count"],
            reverse=True,
        )
    }


# =========================================================
# PROMPT
# =========================================================
def build_prompt(query, contexts, agg, workflow):

    ctx = []
    for c in contexts:
        txt = (c.get("text") or "")[:800]

        ctx.append(f"""
[chunk_id] {c.get("chunk_id")}
[org] {c.get("organization")}
[page] {c.get("page_number")}
[controls] {c.get("controls")}

{txt}
""")

    system = """
You are a Cybersecurity Compliance Expert (Energy Sector) assistant. Your job is to answer the user's question using ONLY the CONTEXT provided in the "CONTEXT" section. Do NOT hallucinate, invent requirements, or add facts not present in the provided context.

Hard rules:
- Use ONLY provided context. If the answer cannot be supported entirely by the provided context, respond with a clear INSUFFICIENT_CONTEXT marker: {"INSUFFICIENT_CONTEXT": true, "clarifying_question": "<one short question to ask the user>"} (when JSON output is required) or a short clarifying question otherwise.
- Every factual claim, requirement, recommendation, or quote must cite at least one chunk_id from the CONTEXT. Use the exact chunk_id values.
- When you include a citation, include: chunk_id, page_number (if available), and a one-sentence excerpt (<= 40 words) from the cited context that supports the claim.
- When asked for recommendations or an action plan, return prioritized, actionable steps with an estimated effort level (Low/Med/High), the role(s) that should own the step, and the minimal evidence (chunk_id list) supporting each step.
- When asked for policy or draft language, produce the draft text and then list the exact supporting chunk_ids and short justification lines tying each clause to the cited chunks.
- NEVER reveal chain-of-thought. You may provide a brief, concise rationale for your answer (1–3 sentences) that cites the supporting chunk_ids, but do not reveal internal deliberations.
- If asked to return JSON, return valid JSON only (no extraneous commentary). If asked for free text, structure your response in sections: Summary, Evidence (with chunk_ids), Recommendations, and Appendix (optional).
- Output a "confidence" field (Low/Medium/High) when returning recommendations or requirements, based solely on how many independent supporting chunks (distinct chunk_ids) support the assertion.

Formatting rules:
- If workflow output is requested (the user requested "recommend", "action plan", "next steps", etc.), produce JSON with:
  {
    "answer_summary": "<concise summary>",
    "key_requirements": [{"requirement": "...", "evidence": ["chunk_id", ...], "confidence": "Low|Medium|High"}],
    "policy_recommendations": [{"recommendation": "...", "priority": "High|Med|Low", "owner": "...", "effort": "Low|Med|High", "evidence": ["chunk_id", ...]}],
    "draft_policy_language": ["<policy paragraph 1>", ...]
  }
- If workflow output is not requested, produce JSON with:
  {
    "answer_summary": "<concise summary>",
    "key_points": ["..."],
    "sources": [{"chunk_id":"...", "document_title":"...", "page_number":..., "excerpt":"..."}]
  }
- Always include a top-level "used_chunk_ids" array listing chunk_ids referenced in the response and a "confidence" field for the overall answer.
- Keep each excerpt <= 40 words and escape newline characters inside JSON strings.

Practical guidance:
- Prefer the most recent, highest-scoring chunks when multiple chunks support the same claim; cite at least two independent chunks for High confidence.
- When recommending fixes, provide short, actionable steps (max 8 steps), estimate effort, and map them to a control family when possible.
- If the user's request is ambiguous or missing scope (e.g., which organization, timeframe, or system), ask a single focused clarifying question before answering.
- If the context contains conflicts, identify the conflict and cite the conflicting chunk_ids.

Tone and audience:
- Use precise, professional language suitable for security teams and compliance officers.
- Provide plain-language summaries (1–2 sentences) for non-technical stakeholders, and a technical appendix for engineers when relevant.

Error handling:
- If your output cannot be expressed as valid JSON (when JSON is requested), return:
  {"error":"INVALID_JSON_OUTPUT", "raw": "<first 2000 chars of the raw generation>"}
- If no context chunks are provided, respond with:
  {"INSUFFICIENT_CONTEXT": true, "clarifying_question":"Please provide relevant documents or clarify scope."}

Follow these rules exactly. Responses that ignore these constraints should be avoided.
"""

    if workflow:
        schema = """
Return JSON:
{
  "answer_summary": "",
  "key_requirements": [],
  "policy_recommendations": [],
  "draft_policy_language": []
}
"""
    else:
        schema = """
Return JSON:
{
  "answer_summary": "",
  "key_points": [],
  "sources": []
}
"""

    user = f"""
QUESTION:
{query}

CONTROL SIGNALS:
{json.dumps(agg, indent=2)}

CONTEXT:
{chr(10).join(ctx)}

{schema}
"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# =========================================================
# MAIN
# =========================================================
def generate_grounded_response(
    query: str,
    top_k: int = 8,
    min_recurring_reviews: int = 2,
    include_debug: bool = False,
):

    client = OpenAI(api_key=_env("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    workflow = wants_workflow_output(query)

    contexts = retrieve(query, top_k=top_k)

    agg = aggregate_contexts(contexts)

    messages = build_prompt(query, contexts, agg, workflow)

    resp = client.responses.create(
        model=model,
        input=messages,
        text={"format": {"type": "json_object"}},
        temperature=0,
    )

    raw = resp.output_text.strip()
    cleaned = _extract_json_object(raw)

    try:
        out = json.loads(cleaned)
    except Exception as e:
        return {
            "error": "Invalid JSON",
            "raw": raw[:3000],
            "exception": str(e),
        }

    if include_debug:
        out["debug"] = {
            "workflow": workflow,
            "num_contexts": len(contexts),
            "control_families": agg,
        }

    return out
