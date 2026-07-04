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
    top_k: int = 12,
    min_score: float = 0.70,
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
def build_prompt(query, contexts, agg, workflow, detail_level: str = "standard"):

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
You are a Cybersecurity Compliance Expert specializing in Energy Sector regulatory frameworks (NERC, NIST, CISA, FERC, DOE).

Your response should be:
- **Comprehensive**: Explain the "why" behind requirements, not just the "what"
- **Structured**: Organize findings by control family, risk level, or compliance domain
- **Evidenced**: Cite every claim with [chunk_id] and relevant metadata (organization, page_number)
- **Actionable**: Provide specific, implementable guidance grounded in provided context

Guidelines:
1. Always include context about which organization (NERC/NIST/etc.) governs each requirement
2. Explain interdependencies between related controls when visible in context
3. When multiple sources discuss the same requirement, synthesize them with citations
4. Use the control_families aggregation to identify patterns and priorities
5. Never introduce external knowledge—only reference provided chunks
6. For ambiguities, explicitly state what the context does not cover
"""

    if workflow:
        schema = """
Return JSON with detailed reasoning:
{
  "answer_summary": "Executive summary (2-3 sentences)",
  "regulatory_context": "Which organizations and frameworks apply",
  "key_requirements": [
    {
      "requirement": "specific requirement text",
      "source": "[chunk_id]",
      "implementation": "how to implement this",
      "risk_if_not_met": "compliance/operational impact"
    }
  ],
  "policy_recommendations": ["specific, actionable recommendations"],
  "draft_policy_language": ["proposed policy text"],
  "interdependencies": "how requirements relate to each other"
}
"""
    else:
        schema = """
Return JSON:
{
  "answer_summary": "Executive summary (2-3 sentences)",
  "regulatory_context": "Which organizations and frameworks apply",
  "key_points": [
    {
      "point": "key finding or requirement",
      "source": "[chunk_id]",
      "organization": "governing organization",
      "relevance": "why this matters"
    }
  ],
  "sources": []
}
"""

    detail_instructions = ""
    if detail_level == "comprehensive":
        detail_instructions = "\n\nProvide an in-depth response that explains regulatory context, implementation considerations, and cross-control relationships."
    elif detail_level == "brief":
        detail_instructions = "\n\nProvide a concise response focused on the core answer."

    user = f"""
QUESTION:
{query}

CONTROL SIGNALS:
{json.dumps(agg, indent=2)}

CONTEXT:
{chr(10).join(ctx)}

{schema}{detail_instructions}
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
    top_k: int = 12,
    min_recurring_reviews: int = 2,
    include_debug: bool = False,
    detail_level: str = "standard",
):
    """
    Generate a grounded, detailed response to a compliance query.
    
    Args:
        query: The user's compliance question
        top_k: Number of context chunks to retrieve (default 12 for comprehensive detail)
        min_recurring_reviews: Minimum recurring reviews threshold
        include_debug: Include debug metadata in response
        detail_level: Response detail level - "brief", "standard", or "comprehensive"
    
    Returns:
        dict: Grounded response with citations and structured output
    """

    client = OpenAI(api_key=_env("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    workflow = wants_workflow_output(query)

    contexts = retrieve(query, top_k=top_k)

    agg = aggregate_contexts(contexts)

    messages = build_prompt(query, contexts, agg, workflow, detail_level=detail_level)

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw = resp.choices[0].message.content.strip()
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
            "detail_level": detail_level,
            "num_contexts": len(contexts),
            "control_families": agg,
        }

    return out
