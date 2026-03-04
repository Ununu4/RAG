# rag_qa.py
from __future__ import annotations
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests
from chromadb import PersistentClient
from query_improved import semantic_query

from backends import get_backend

from faithfulness import compute_faithfulness as _compute_faithfulness_score
from monitoring import (
    CostAwareStrategy,
    JsonFileStrategy,
    LoggingStrategy,
    PipelineMetrics,
    clear_strategies,
    get_logger,
    notify,
    register_strategy,
    setup_logging,
    timed,
)

# ---------------------------------------------------------------------------
# Lender eligibility registry — sourced directly from guidelines files.
# Used for: (1) deterministic pre-filtering before Chroma queries,
#           (2) structured source headers in the prompt.
#
# Fields:
#   min_revenue       : minimum monthly revenue in dollars (None = no hard min)
#   max_position      : maximum position accepted (e.g. 3 = up to 3rd; 99 = no limit)
#   restricted_states : list of uppercase 2-letter state codes that are restricted/declined
#   prohibited_keywords: lowercase industry keyword fragments that trigger exclusion
#   display_name      : human-readable lender name for source headers
# ---------------------------------------------------------------------------
LENDER_ELIGIBILITY: Dict[str, Dict] = {
    "lender-501-advance": {
        "display_name": "501 Advance",
        "min_revenue": 20000,
        "max_position": 99,
        "restricted_states": ["CA", "HI", "PR"],
        "prohibited_keywords": ["commission based", "auto sales", "financial firms", "transportation"],
    },
    "lender-advantage-capital-funding": {
        "display_name": "Advantage Capital Funding",
        "min_revenue": None,
        "max_position": 4,
        "restricted_states": [],
        "prohibited_keywords": ["adult entertainment", "attorneys", "auto dealer", "trucking", "non-profit", "real estate broker"],
    },
    "lender-alternative-funding-group": {
        "display_name": "Alternative Funding Group",
        "min_revenue": 30000,
        "max_position": 4,
        "restricted_states": ["CA", "NY"],
        "prohibited_keywords": ["stock broker", "crypto broker", "cash checking", "bail bond"],
    },
    "lender-apex-funding-source": {
        "display_name": "Apex Funding Source",
        "min_revenue": 100000,
        "max_position": 6,
        "restricted_states": ["CA", "NY", "VA", "UT"],
        "prohibited_keywords": [],
    },
    "lender-arsenal-funding": {
        "display_name": "Arsenal Funding",
        "min_revenue": None,
        "max_position": 6,
        "restricted_states": [],
        "prohibited_keywords": ["adult entertainment", "bail bond", "car dealer", "gambling", "legal service", "travel agenc"],
    },
    "lender-aspire-funding-platform": {
        "display_name": "Aspire Funding Platform",
        "min_revenue": None,
        "max_position": 6,
        "restricted_states": ["NY"],
        "prohibited_keywords": ["check cashing", "financial service", "online gambling"],
    },
    "lender-aurum-funding": {
        "display_name": "Aurum Funding",
        "min_revenue": 30000,
        "max_position": 10,
        "restricted_states": [],
        "prohibited_keywords": ["staffing", "check cashing", "auto sales", "non-profit", "real estate"],
    },
    "lender-avanza": {
        "display_name": "Avanza",
        "min_revenue": 50000,
        "max_position": 99,
        "restricted_states": ["CA"],
        "prohibited_keywords": ["attorney", "trucking", "auto sales", "vape"],
    },
    "lender-backd": {
        "display_name": "BackD",
        "min_revenue": 100000,
        "max_position": 99,
        "restricted_states": [],
        "prohibited_keywords": ["pharmacy", "pharmacies", "financial service", "car dealership", "real estate",
                                "legal service", "non-profit", "adult entertainment", "cannabis", "trucking", "solar", "wholesale"],
    },
    "lender-bellwether": {
        "display_name": "Bellwether",
        "min_revenue": 40000,
        "max_position": 99,
        "restricted_states": [],
        "prohibited_keywords": ["law", "auto dealer", "real estate", "non-profit"],
    },
    "lender-bitty-advance": {
        "display_name": "Bitty Advance",
        "min_revenue": 5000,
        "max_position": 99,
        "restricted_states": [],
        "prohibited_keywords": ["bail bond", "crypto", "debt collection", "gambling", "health supplement",
                                "money service", "non-profit", "psychic", "religious", "sexual"],
    },
    "lender-biz-2-credit": {
        "display_name": "Biz-2-Credit",
        "min_revenue": 40000,
        "max_position": 99,
        "restricted_states": [],
        "prohibited_keywords": ["used car dealer", "cannabis", "adult entertainment", "real estate investor",
                                "car service", "construction", "sole prop"],
    },
    "lender-bizfund": {
        "display_name": "BizFund",
        "min_revenue": 20000,
        "max_position": 3,
        "restricted_states": [],
        "prohibited_keywords": ["car dealership", "adult", "entertainment", "collection agenc", "gambling",
                                "bail bond", "pawn shop", "nail salon", "real estate"],
    },
    "lender-blade": {
        "display_name": "Blade",
        "min_revenue": 50000,
        "max_position": 5,
        "restricted_states": [],
        "prohibited_keywords": ["auto dealer", "real estate broker", "cannabis", "financial institution"],
    },
    "lender-can-capital": {
        "display_name": "Can Capital",
        "min_revenue": None,
        "max_position": 1,
        "restricted_states": [],
        "prohibited_keywords": ["cannabis", "real estate", "transportation", "trucking", "non-profit",
                                "auto dealer", "adult entertainment", "gambling", "financial service", "legal service"],
    },
    "lender-cashable": {
        "display_name": "Cashable",
        "min_revenue": 25000,
        "max_position": 10,
        "restricted_states": ["HI", "AK", "PR"],
        "prohibited_keywords": ["financial service", "non-profit", "trucking"],
    },
    "lender-cfg-merchant-solutions": {
        "display_name": "CFG Merchant Solutions",
        "min_revenue": 10000,
        "max_position": 4,
        "restricted_states": [],
        "prohibited_keywords": [],
    },
    "lender-channel": {
        "display_name": "Channel",
        "min_revenue": None,
        "max_position": 2,
        "restricted_states": [],
        "prohibited_keywords": [],
    },
    "lender-clearfund": {
        "display_name": "Clearfund",
        "min_revenue": 100000,
        "max_position": 5,
        "restricted_states": [],
        "prohibited_keywords": [],
    },
}


def _prefilter_collections(
    collections: List[str],
    user_criteria: Dict,
) -> List[str]:
    """
    Deterministically exclude lender collections that hard-fail on user criteria.
    Returns a filtered list of collection names that could plausibly match.
    """
    revenue = user_criteria.get("revenue_monthly")
    positions = user_criteria.get("positions")
    state = (user_criteria.get("state") or "").upper().strip()
    industry = (user_criteria.get("industry") or "").lower().strip()

    next_position = None
    if positions is not None:
        try:
            next_position = int(positions) + 1
        except (TypeError, ValueError):
            pass

    passed: List[str] = []
    for coll in collections:
        elig = LENDER_ELIGIBILITY.get(coll)
        if elig is None:
            passed.append(coll)
            continue

        # Revenue check: skip if user revenue is set and below lender minimum
        if revenue is not None and elig.get("min_revenue") is not None:
            try:
                if int(revenue) < int(elig["min_revenue"]):
                    continue
            except (TypeError, ValueError):
                pass

        # Position check: skip if user's next position exceeds lender maximum
        if next_position is not None and elig.get("max_position") is not None:
            try:
                if int(next_position) > int(elig["max_position"]):
                    continue
            except (TypeError, ValueError):
                pass

        # State check: skip if user's state is in the restricted list
        if state and state in elig.get("restricted_states", []):
            continue

        # Industry check: skip if any prohibited keyword matches the user's industry
        if industry:
            prohibited = elig.get("prohibited_keywords", [])
            if any(kw in industry for kw in prohibited):
                continue

        passed.append(coll)

    return passed if passed else collections


def _invoke_llm(messages: List[Dict], num_ctx: int = 12288, num_predict: int = 512) -> str:
    """Invoke LLM via configured backend (Ollama or Bedrock)."""
    backend = get_backend()
    return backend.invoke(messages, num_ctx=num_ctx, num_predict=num_predict)


# ---------------------------------------------------------------------------
# Regex-first fast query parser — eliminates LLM call for most structured queries
# ---------------------------------------------------------------------------

_STATE_MAP: Dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
}
_STATE_ABBRS: set = set(_STATE_MAP.values())

_INDUSTRY_PATTERNS = [
    (r"\bpharmac(?:y|ies|ist)\b", "pharmacy"),
    (r"\brestaurant\b|\bdiner\b|\bfood\s+service\b", "restaurant"),
    (r"\btrucking\b|\bfreight\b|\btruck(?:er|ers)?\b", "trucking"),
    (r"\bauto\s+(?:dealer|dealership|sales)\b|\bcar\s+dealer\b|\bused\s+car\b", "auto dealership"),
    (r"\breal\s+estate\b", "real estate"),
    (r"\binsurance\b", "insurance"),
    (r"\bveterinar(?:y|ian)\b|\bvet\s+clinic\b|\banimal\s+hospital\b", "veterinary"),
    (r"\bconstruction\b|\bcontractor\b", "construction"),
    (r"\b(?:legal|law)\s+(?:firm|service|office)\b|\battorney\b|\blawyer\b", "legal services"),
    (r"\bmedical\b|\bhealthcare\b|\bhealth\s+care\b|\bclinic\b|\bdental\b", "medical"),
    (r"\bgrocery\b|\bsupermarket\b", "grocery"),
    (r"\bsalon\b|\bbarber\b|\bnail\s+salon\b", "salon"),
    (r"\bcannabis\b|\bmarijuana\b|\bdispensary\b", "cannabis"),
    (r"\bnon[\s-]profit\b|\bnonprofit\b", "non-profit"),
    (r"\bstaffing\b|\btemp\s+agency\b", "staffing"),
    (r"\bsolar\b", "solar"),
    (r"\btransportation\b", "transportation"),
    (r"\bgas\s+station\b|\bfuel\b", "gas station"),
    (r"\bbar\b|\bnightclub\b|\bclub\b", "bar/nightclub"),
    (r"\beautomotive\b|\bauto\s+repair\b|\bauto\s+body\b", "auto repair"),
    (r"\bhospital\b|\bnursing\s+home\b|\bsenior\s+care\b", "healthcare facility"),
]

_WHICH_LENDER_PHRASES = [
    "who can fund", "which lender", "who could fund", "who would fund",
    "can fund him", "can fund her", "can fund me", "can fund them",
    "fund him", "fund her", "fund me", "fund them",
    "who can help", "looking for funding", "looking to get", "looking for a lender",
    "best lender", "recommend a lender", "find a lender", "who funds",
    "which funders", "who to go to", "what lender", "any lender",
    "need funding", "need a lender", "need financing",
]

_REQUIREMENTS_PHRASES = [
    "what do i need", "what is needed", "documentation", "what to send",
    "how to submit", "what documents", "what do you need", "required documents",
    "what should i prepare", "what should i send",
]

_ELIGIBILITY_PHRASES = [
    "eligible", "qualify", "can i apply", "do i qualify", "am i approved",
    "will they approve", "can they fund", "does it qualify",
]

_RESTRICTIONS_PHRASES = [
    "restrict", "prohibited", "not allowed", "declined", "won't fund",
    "cannot fund", "does not fund",
]

# Build lender lookup from LENDER_ELIGIBILITY at module load
def _build_lender_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for slug, info in LENDER_ELIGIBILITY.items():
        name = info.get("display_name", "")
        if name:
            lookup[name.lower()] = name
        short = slug.replace("lender-", "").replace("-", " ")
        lookup[short] = name or short.title()
    return lookup


def _parse_query_regex(query: str) -> Dict:
    """
    Fast sub-millisecond query parsing via regex patterns.
    Returns the same shape as _understand_query.
    Returns None for fields that cannot be reliably determined (triggers LLM fallback).
    """
    ql = query.lower().strip()
    out: Dict = {
        "industry": None, "lender": None, "intent": "other",
        "revenue_monthly": None, "positions": None, "state": None,
        "_regex_confidence": 0,  # internal: how many fields extracted
    }

    # Intent
    if any(p in ql for p in _WHICH_LENDER_PHRASES):
        out["intent"] = "which_lender"
    elif any(p in ql for p in _REQUIREMENTS_PHRASES):
        out["intent"] = "requirements"
    elif any(p in ql for p in _ELIGIBILITY_PHRASES):
        out["intent"] = "eligibility"
    elif any(p in ql for p in _RESTRICTIONS_PHRASES):
        out["intent"] = "restrictions"

    # Revenue: "$80K", "80k", "80,000", "$80,000", "80k per month", "80K monthly"
    rev = None
    m = re.search(r"\$?\s*(\d+(?:\.\d+)?)\s*[kK]\b", query)
    if m:
        try:
            rev = int(float(m.group(1)) * 1000)
        except ValueError:
            pass
    if rev is None:
        m = re.search(r"\$(\d{1,3}(?:,\d{3})+)", query)
        if m:
            try:
                rev = int(m.group(1).replace(",", ""))
            except ValueError:
                pass
    if rev is None:
        m = re.search(r"(\d{4,7})\s*(?:per\s+month|monthly|/month|/mo)", query, re.IGNORECASE)
        if m:
            try:
                rev = int(m.group(1))
            except ValueError:
                pass
    if rev and rev > 0:
        out["revenue_monthly"] = rev
        out["_regex_confidence"] += 1

    # Positions: "2 positions", "two positions", "2 existing positions"
    pos = None
    word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                   "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    m = re.search(r"\b(\d+)\s+(?:current\s+|existing\s+)?position", ql)
    if m:
        try:
            pos = int(m.group(1))
        except ValueError:
            pass
    if pos is None:
        for word, num in word_to_num.items():
            if re.search(rf"\b{word}\s+(?:current\s+|existing\s+)?position", ql):
                pos = num
                break
    if pos is not None:
        out["positions"] = pos
        out["_regex_confidence"] += 1

    # State: full name first, then 2-letter abbreviation with preposition
    for name, abbr in _STATE_MAP.items():
        if re.search(r"\b" + re.escape(name) + r"\b", ql):
            out["state"] = abbr
            out["_regex_confidence"] += 1
            break
    if not out["state"]:
        m = re.search(r"\b(?:in|from|based\s+in|located\s+in)\s+([A-Z]{2})\b", query)
        if m and m.group(1) in _STATE_ABBRS:
            out["state"] = m.group(1)
            out["_regex_confidence"] += 1

    # Industry
    for pattern, name in _INDUSTRY_PATTERNS:
        if re.search(pattern, ql):
            out["industry"] = name
            out["_regex_confidence"] += 1
            break

    # Lender name lookup
    lender_lookup = _build_lender_lookup()
    for token, display in sorted(lender_lookup.items(), key=lambda x: -len(x[0])):
        if len(token) >= 4 and token in ql:
            out["lender"] = display
            out["_regex_confidence"] += 1
            break

    # Intent upgrade: if intent still "other" but we have structured criteria → which_lender
    if out["intent"] == "other" and (out["revenue_monthly"] or out["positions"] or out["state"]):
        kw = ql
        if any(p in kw for p in ("looking for", "need ", "who can", "which lender", "fund me", "can fund", "funding")):
            out["intent"] = "which_lender"

    return out


def _normalize_revenue(val, query: str) -> Optional[int]:
    """Normalize revenue_monthly: 90 -> 90000 if query has 90K, 50 -> 50000 if 50K, etc."""
    if val is None:
        return None
    try:
        n = int(float(val)) if val is not None else None
    except (TypeError, ValueError):
        return None
    if n is None or n <= 0:
        return None
    if n >= 1000:
        return n
    q = (query or "").lower()
    m = re.search(rf"\b{n}\s*[kK]\b|\b{n}\s*,\s*000\b", q)
    if m:
        return n * 1000
    return n


def _understand_query(query: str, metrics=None) -> Dict:
    """
    Extract industry, lender, intent, and deal criteria for prompt context.
    Uses regex-first fast parsing (<5ms) and falls back to LLM only for ambiguous queries.
    """
    t0 = time.perf_counter()
    fast = _parse_query_regex(query)
    confidence = fast.pop("_regex_confidence", 0)

    # Use regex result if intent is clear OR we extracted enough structured fields
    # Skip LLM for: explicit which_lender/requirements/eligibility/restrictions phrasing,
    # or if we got ≥2 structured fields (revenue, positions, state, industry).
    use_llm = fast["intent"] == "other" and confidence < 2

    if use_llm:
        backend = get_backend()
        msg = [{"role": "user", "content": f"""Extract from this lender FAQ query. Return ONLY valid JSON.
{{"industry": "industry if mentioned else null", "lender": "lender name if mentioned else null", "intent": "which_lender|eligibility|requirements|restrictions|comparison|other", "revenue_monthly": "number if mentioned else null", "positions": "number if mentioned else null", "state": "2-letter US state code if mentioned else null"}}

Use "which_lender" when user asks who can fund, which lender, or describes business + funding need. Extract only what is stated.
Query: {query}"""}]
        out = {k: fast[k] for k in ("industry", "lender", "intent", "revenue_monthly", "positions", "state")}
        try:
            raw = backend.invoke(msg, num_ctx=4096, num_predict=150)
            m = re.search(r"\{[^{}]*\}", raw)
            if m:
                obj = json.loads(m.group(0))
                # Merge: LLM fills gaps, regex values take precedence where already found
                for k in ("industry", "lender", "revenue_monthly", "positions", "state"):
                    if out[k] is None:
                        out[k] = obj.get(k) or None
                if out["intent"] == "other":
                    out["intent"] = obj.get("intent") or "other"
                out["revenue_monthly"] = _normalize_revenue(out.get("revenue_monthly"), query)
        except Exception:
            pass
    else:
        out = {k: fast[k] for k in ("industry", "lender", "intent", "revenue_monthly", "positions", "state")}
        out["revenue_monthly"] = _normalize_revenue(out.get("revenue_monthly"), query)

    # Final fallback: funding intent with criteria but missed intent keywords
    if out["intent"] == "other" and (out["industry"] or out["revenue_monthly"] or out["positions"] or out["state"]):
        ql = query.lower()
        if any(p in ql for p in ("looking for", "need ", "who can", "which lender", "fund me", "can fund")):
            out["intent"] = "which_lender"

    understand_ms = (time.perf_counter() - t0) * 1000
    if metrics is not None:
        metrics.understand_ms = round(understand_ms, 2)
        metrics.understand_used_llm = use_llm

    return out


def _format_sources(results: Dict, max_chars_per_doc: int = 2000) -> str:
    lines = []
    for i, (rid, doc, meta) in enumerate(zip(results["ids"][0], results["documents"][0], results["metadatas"][0]), 1):
        lender_slug = meta.get("lender_name", "?")
        section = meta.get("section", "")

        # Build a quick-scan eligibility header from the registry when available
        elig = LENDER_ELIGIBILITY.get(lender_slug)
        if elig:
            display = elig["display_name"]
            min_rev = f"${elig['min_revenue']:,}/mo" if elig.get("min_revenue") else "no min"
            max_pos = f"up to {elig['max_position']}th pos" if elig.get("max_position") and elig["max_position"] < 99 else "1st+ pos"
            restricted = (", ".join(elig["restricted_states"]) or "none")
            header = f"[S{i}] {display} | Min Rev: {min_rev} | Positions: {max_pos} | Restricted states: {restricted}"
        else:
            header = f"[S{i}] id={rid} | lender={lender_slug} | section={section}"

        preview = doc if len(doc) <= max_chars_per_doc else doc[:max_chars_per_doc] + "..."
        lines.append(header + "\n" + preview)
    return "\n\n".join(lines)

def _build_messages(
    query: str,
    sources_block: str,
    json_schema_hint: str,
    intent_context: Optional[Dict] = None,
    background: Optional[str] = None,
) -> List[Dict]:
    system = (
        "You are a trusted financial advisor. Answer ONLY from the Sources below. "
        "Cite every claim with [S1], [S2]. Never infer or add information not in sources. "
        "Use EXACT numbers from the source—if it says $75,000/mo write $75,000/mo, not $75k or $100k. Do not round or approximate. "
        "Be smooth and confident. Use lender names (e.g. Aurum Funding), never slugs. Output valid JSON only."
    )
    focus = ""
    if intent_context:
        industry = intent_context.get("industry")
        intent = intent_context.get("intent", "other")
        if intent == "which_lender":
            criteria_parts = []
            if industry:
                criteria_parts.append(f"industry: {industry}")
            if intent_context.get("revenue_monthly"):
                rev = intent_context["revenue_monthly"]
                try:
                    criteria_parts.append(f"revenue: ${int(rev):,}/month")
                except (TypeError, ValueError):
                    criteria_parts.append(f"revenue: ${rev}/month")
            if intent_context.get("positions"):
                criteria_parts.append(f"positions: {intent_context['positions']}")
            if intent_context.get("state"):
                criteria_parts.append(f"state: {intent_context['state']}")
            criteria_str = "; ".join(criteria_parts) if criteria_parts else "see query"

            position_logic = ""
            pos_val = intent_context.get("positions")
            if pos_val is not None:
                try:
                    n = int(pos_val)
                    next_pos = n + 1
                    position_logic = (
                        f"\nPOSITION LOGIC: User has {n} existing positions = {next_pos}th position for new funding. "
                        f"EXCLUDE lenders that: (a) auto-decline {next_pos}+ positions, or (b) only accept 1st through {n}th. "
                        "Check the source for 'auto decline', 'positions', '1st-3rd' etc. Only recommend if user's position fits.\n"
                    )
                except (TypeError, ValueError):
                    pass

            revenue_logic = ""
            rev_val = intent_context.get("revenue_monthly")
            if rev_val is not None:
                try:
                    r = int(rev_val)
                    revenue_logic = (
                        f"\nREVENUE LOGIC: User has ${r:,}/month revenue. "
                        f"EXCLUDE lenders whose minimum monthly revenue requirement exceeds ${r:,}. "
                        "Check the source for 'minimum monthly revenue', 'min revenue', 'average monthly revenue' etc. "
                        "If the source says minimum $100,000/month and the user has $40,000, EXCLUDE that lender.\n"
                    )
                except (TypeError, ValueError):
                    pass

            focus = (
                f"\nContext: LENDER RECOMMENDATION. User criteria: {criteria_str}. "
                f"{position_logic}"
                f"{revenue_logic}"
                "Return only the 3–5 BEST matches that clearly fit ALL criteria. Apply judgment—exclude any lender that fails position OR revenue requirements. "
                "For each lender: COPY the exact position and revenue requirements from the source—do not paraphrase or infer. "
                "Use lender names (e.g. Cashable, Aspire), not slugs. "
                "CRITICAL: If source says 'Positions: 1st-3rd' write 1st-3rd; if '$25,000' write $25,000. Never substitute different numbers. "
                "Intro: Start with user's situation. Format: Brief intro. Then 3–5 recommendations, each one sentence with [S#].\n"
            )
        elif industry:
            focus = f"\nContext: {industry} deals. Be concise; cite sources.\n"
        elif intent != "other":
            focus = f"\nContext: {intent}. Be concise; cite sources.\n"
    summary_block = ""
    if background and background.strip():
        summary_block = f"\nKey points (distilled):\n{background.strip()}\n\n"
    user = f"""Query: {query}
{focus}
{summary_block}Sources:
{sources_block}

Respond with JSON: {json_schema_hint}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user.strip()},
    ]

def _distill_sources(results: Dict, max_bullets: int = 60) -> str:
    """Extract salient bullet lines from retrieved documents for dense guidance."""
    bullets: List[str] = []
    docs = results.get("documents", [[]])[0] if results else []
    for doc in docs:
        for line in (doc or "").splitlines():
            s = line.strip()
            if (s.startswith("*") or s.startswith("-")) and len(s) > 3:
                bullets.append(s.lstrip("*- ").strip())
    if len(bullets) > max_bullets:
        bullets = bullets[:max_bullets]
    return "\n".join(f"- {b}" for b in bullets)

def _extract_citations(text: str) -> List[str]:
    seen = set()
    cites: List[str] = []
    for m in re.finditer(r"\[S(\d+)\]", text or ""):
        sid = f"S{m.group(1)}"
        if sid not in seen:
            seen.add(sid)
            cites.append(sid)
    return cites

def _build_source_index(results: Dict) -> List[Dict]:
    index: List[Dict] = []
    ids = results.get("ids", [[]])[0] if results else []
    mets = results.get("metadatas", [[]])[0] if results else []
    for i, (rid, meta) in enumerate(zip(ids, mets), 1):
        lender = meta.get("lender_name", "?") if isinstance(meta, dict) else "?"
        section = meta.get("section", "") if isinstance(meta, dict) else ""
        index.append({"sid": f"S{i}", "id": rid, "lender": lender, "section": section})
    return index

def _normalize_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def _get_lender_tokens(chroma_path: str) -> set[str]:
    client = PersistentClient(path=chroma_path)
    tokens: set[str] = set()
    for c in client.list_collections():
        name = getattr(c, "name", "")
        if not name.startswith("lender-"):
            continue
        slug = name.replace("lender-", "")
        tokens.add(_normalize_token(slug))
        try:
            col = client.get_collection(name)
            meta = getattr(col, "metadata", {}) or {}
            lender_name = _normalize_token(meta.get("lender_name", ""))
            if lender_name:
                tokens.add(lender_name)
        except Exception:
            pass
    return {t for t in tokens if len(t) >= 3}

def _filter_cross_lender_mentions(results: Dict, expected_slug: str, chroma_path: str) -> Dict:
    all_tokens = _get_lender_tokens(chroma_path)
    exp = _normalize_token(expected_slug)
    other = {t for t in all_tokens if t and t != exp and exp not in t}

    ids = results.get("ids", [[]])[0] if results else []
    docs = results.get("documents", [[]])[0] if results else []
    metas = results.get("metadatas", [[]])[0] if results else []

    keep_ids, keep_docs, keep_metas = [], [], []
    for rid, d, m in zip(ids, docs, metas):
        txt = f" {_normalize_token(d)} "
        if any(f" {tok} " in txt for tok in other):
            continue
        keep_ids.append(rid); keep_docs.append(d); keep_metas.append(m)

    if keep_ids:
        return {"ids": [keep_ids], "documents": [keep_docs], "metadatas": [keep_metas]}
    return results

def _filter_results_by_lender(results: Dict, expected_slug: str) -> Dict:
    def _to_slug(s: str) -> str:
        return re.sub(r"[^\w\-]+", "-", (s or "").lower()).strip("-")
    ids = results.get("ids", [[]])[0] if results else []
    docs = results.get("documents", [[]])[0] if results else []
    metas = results.get("metadatas", [[]])[0] if results else []
    keep_ids, keep_docs, keep_metas = [], [], []
    for rid, d, m in zip(ids, docs, metas):
        lender_name = m.get("lender_name") if isinstance(m, dict) else ""
        if _to_slug(lender_name) == expected_slug:
            keep_ids.append(rid)
            keep_docs.append(d)
            keep_metas.append(m)
    if keep_ids:
        return {"ids": [keep_ids], "documents": [keep_docs], "metadatas": [keep_metas]}
    return results

def _dict_literal_to_prose(text: str) -> str:
    """Convert Python dict literals to readable bullets (e.g. {'lender':'X'} -> - X: ...)."""
    lines: List[str] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln in ("{", "}"):
            continue
        m = re.search(r"['\"]?lender['\"]?\s*:\s*['\"]([^'\"]+)['\"]", ln, re.I)
        if m:
            lender = m.group(1)
            rest = re.sub(r"['\"]?lender['\"]?\s*:\s*[^,}]+", "", ln)
            rest = re.sub(r"['\"][^'\"]+['\"]\s*:\s*", " ", rest).replace("'", "").replace("{", "").replace("}", "").strip()
            lines.append(f"- {lender}: {rest}" if rest else f"- {lender}")
        elif ln.startswith("{") and "lender" in ln.lower():
            m2 = re.search(r"lender['\"]?\s*:\s*['\"]([^'\"]+)['\"]", ln, re.I)
            if m2:
                lines.append(f"- {m2.group(1)}: (see sources)")
        else:
            lines.append(ln)
    return "\n".join(lines) if lines else text


def _polish_answer(answer: str) -> str:
    a = "\n".join(str(x) for x in answer).strip() if isinstance(answer, list) else (answer or "").strip()
    if not a:
        return a
    if re.search(r"\{['\"]?lender['\"]?\s*:", a):
        a = _dict_literal_to_prose(a)
    cleaned: List[str] = []
    seen: set = set()
    for ln in a.splitlines():
        s = ln.rstrip()
        # drop stray JSON echoes
        if s.startswith(('"answer"', "'answer'", '{', '}')):
            continue
        if s in seen:
            continue
        seen.add(s)
        # normalize bullets
        if s.startswith(("• ", "– ", "* ")):
            s = "- " + s[2:].strip()
        cleaned.append(s)
    # condense blank lines
    out: List[str] = []
    prev_blank = False
    for s in cleaned:
        if not s.strip():
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        out.append(s)
    return "\n".join(out).strip()

def _split_list_str(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    parts = re.split(r",|;| and ", s, flags=re.I)
    items = [re.sub(r"\s+", " ", p).strip(" .") for p in parts]
    return [i for i in items if i]

def _normalize_to_schema(obj: Dict) -> Dict:
    norm: Dict = {
        "answer": (obj.get("answer") or ""),
        "details": obj.get("details") if isinstance(obj.get("details"), list) else [],
        "requirements": {
            "fico_min": None,
            "revenue_min": None,
            "tib_min": None,
            "restricted_states": [],
            "restricted_industries": [],
            "required_documents": [],
        },
        "programs": [],
        "exclusions": [],
        "notes": [],
        "recommendations": obj.get("recommendations") if isinstance(obj.get("recommendations"), list) else [],
        "sources": obj.get("sources") if isinstance(obj.get("sources"), list) else [],
    }

    # Map model-specific keys to canonical schema
    hri = obj.get("high_risk_industries") or {}
    if isinstance(hri, dict):
        for k, v in hri.items():
            lst = _split_list_str(v)
            if not lst:
                continue
            kl = (k or "").lower()
            if "prohibited" in kl or "not_for_profit" in kl or "not_for_profits" in kl:
                norm["exclusions"].extend(lst)
            else:
                norm["requirements"]["restricted_industries"].extend(lst)

    # Submission criteria -> required docs
    ksc = obj.get("key_submission_criteria")
    if isinstance(ksc, str) and "tax return" in ksc.lower():
        norm["requirements"]["required_documents"].append("Tax returns (depends on amount)")

    # Popular industries -> notes
    pop = obj.get("popular_industries")
    if isinstance(pop, str):
        norm["notes"].append(f"Popular industries: {pop.strip()}")

    # Deduplicate lists
    for k in ("restricted_states", "restricted_industries", "required_documents"):
        items = norm["requirements"][k]
        seen: set = set()
        cleaned: List[str] = []
        for it in items:
            it2 = (it or "").strip()
            key = it2.lower()
            if it2 and key not in seen:
                seen.add(key)
                cleaned.append(it2)
        norm["requirements"][k] = cleaned

    for k in ("programs", "exclusions", "notes", "recommendations"):
        seq = norm.get(k) or []
        seen: set = set()
        cleaned: List[str] = []
        for it in seq:
            it2 = it.strip() if isinstance(it, str) else it
            key = it2.lower() if isinstance(it2, str) else None
            if isinstance(it2, str) and it2 and key not in seen:
                seen.add(key)
                cleaned.append(it2)
        norm[k] = cleaned

    # Ensure an answer summary exists
    if not norm["answer"]:
        ri = norm["requirements"]["restricted_industries"]
        if ri:
            sample = ", ".join(ri[:6]) + ("…" if len(ri) > 6 else "")
            norm["answer"] = f"Restricted industries identified: {sample}."
        else:
            norm["answer"] = "Relevant restrictions and requirements identified from sources."

    return norm

def _render_answer_text(obj: Dict) -> str:
    """Render a clean, human-readable answer from structured JSON fields."""
    lines: List[str] = []
    ans = obj.get("answer") or obj.get("answer_text")
    if isinstance(ans, str) and ans.strip():
        lines.append(ans.strip())

    req = obj.get("requirements", {}) or {}
    ri = req.get("restricted_industries") or []
    if isinstance(ri, list) and ri:
        lines.append("\nRestricted industries:")
        for item in ri:
            lines.append(f"- {item}")

    rs = req.get("restricted_states") or []
    if isinstance(rs, list) and rs:
        lines.append("\nRestricted states:")
        for st in rs:
            lines.append(f"- {st}")

    docs = req.get("required_documents") or []
    if isinstance(docs, list) and docs:
        lines.append("\nRequired documents:")
        for d in docs:
            lines.append(f"- {d}")

    # Include programs/exclusions if present
    programs = obj.get("programs") or []
    if isinstance(programs, list) and programs:
        lines.append("\nPrograms:")
        for p in programs:
            lines.append(f"- {p}")

    exclusions = obj.get("exclusions") or []
    if isinstance(exclusions, list) and exclusions:
        lines.append("\nExclusions:")
        for e in exclusions:
            lines.append(f"- {e}")

    notes = obj.get("notes") or []
    if isinstance(notes, list) and notes:
        lines.append("\nNotes:")
        for n in notes:
            lines.append(f"- {n}")

    recs = obj.get("recommendations") or []
    if isinstance(recs, list) and recs:
        lines.append("\nRecommendations:")
        for r in recs:
            lines.append(f"- {r}")

    return "\n".join(lines).strip()

def _slugify(text: str) -> str:
    return re.sub(r"[^\w\-]+", "-", text.lower()).strip("-")


def _resolve_collection(short_name: Optional[str], chroma_path: str) -> Optional[str]:
    """Resolve short names like 'Bitty' or 'bitty' to full collection name 'lender-bitty-advance'."""
    if not short_name or not short_name.strip():
        return short_name
    s = short_name.strip().lower().replace("_", "-")
    s = re.sub(r"[^\w\-]+", "-", s).strip("-")
    if not s:
        return short_name
    if s.startswith("lender-"):
        return s
    client = PersistentClient(path=chroma_path)
    for c in client.list_collections():
        name = getattr(c, "name", "")
        if not name.startswith("lender-"):
            continue
        slug = name.replace("lender-", "")
        if s == slug or s in slug or slug.startswith(s):
            return name
    return short_name  # pass through, will error with clear message


def _detect_collection_for_query(query: str, chroma_path: str, default_collection: str) -> str:
    client = PersistentClient(path=chroma_path)
    cols = client.list_collections()

    # Tokenize query slug and remove generic lender terms
    q_slug = _slugify(query)
    q_tokens = [t for t in q_slug.split("-") if len(t) >= 3]
    stop = {
        "funding", "group", "capital", "financial", "finance", "platform",
        "solutions", "advance", "advances", "business", "credit", "loans", "loan",
        "merchant", "based", "looking", "making", "revenue", "positions",
    }
    q_tokens = [t for t in q_tokens if t not in stop]
    q_set = set(q_tokens)

    best_score = 0
    best_name = default_collection

    for c in cols:
        name = getattr(c, "name", "")
        if not name.startswith("lender-"):
            continue
        slug = name.replace("lender-", "")
        s_tokens = [t for t in slug.split("-") if t and t not in stop]
        s_set = set(s_tokens)

        # Primary score: token overlap
        overlap = len(q_set & s_set)

        # Secondary: partial containment (e.g., "aspire" matches "aspire-funding-platform")
        if overlap == 0 and any(qt in st for qt in q_set for st in s_set):
            overlap = 1

        if overlap > best_score:
            best_score = overlap
            best_name = name

    return best_name if best_score > 0 else default_collection


def _approx_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(0, len((text or "").strip()) // 4)


def _get_lender_collections(chroma_path: str) -> List[str]:
    """List all lender collection names."""
    client = PersistentClient(path=chroma_path)
    return [c.name for c in client.list_collections() if getattr(c, "name", "").startswith("lender-")]


def _multi_collection_search(
    query_text: str,
    chroma_path: str,
    n_per_collection: int = 3,
    n_total: int = 14,
    max_per_lender: int = 2,
    user_criteria: Optional[Dict] = None,
    metrics_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """Search across all lender collections, merge by relevance, enforce lender diversity."""
    import time as _time

    t0 = _time.perf_counter()
    collections = _get_lender_collections(chroma_path)
    if not collections:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]]}

    # Deterministic pre-filter: skip lenders that hard-fail user criteria
    if user_criteria:
        before = len(collections)
        collections = _prefilter_collections(collections, user_criteria)
        skipped = before - len(collections)
        if skipped:
            logger = logging.getLogger(__name__)
            logger.info(f"[prefilter] Skipped {skipped} of {before} lender collections based on user criteria")

    all_ids: List[str] = []
    all_docs: List[str] = []
    all_metas: List[Dict] = []
    all_distances: List[float] = []

    for coll_name in collections:
        try:
            res = semantic_query(
                query_text=query_text,
                collection_name=coll_name,
                chroma_path=chroma_path,
                n_results=n_per_collection,
                mmr=False,
                rerank=False,
                expand_neighbors=0,
            )
            ids = res["ids"][0]
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            dists = res.get("distances", [[1.0] * len(ids)])[0]
            for i, (rid, doc, meta) in enumerate(zip(ids, docs, metas)):
                all_ids.append(rid)
                all_docs.append(doc)
                all_metas.append(meta)
                all_distances.append(dists[i] if i < len(dists) else 1.0)
        except Exception:
            continue

    # Sort by distance (lower = more relevant), then apply lender diversity cap
    if all_distances:
        order = sorted(range(len(all_distances)), key=lambda i: all_distances[i])
        lender_counts: Dict[str, int] = {}
        keep: List[int] = []
        for i in order:
            if len(keep) >= n_total:
                break
            meta = all_metas[i]
            lender = (meta.get("lender_name") or "?") if isinstance(meta, dict) else "?"
            cnt = lender_counts.get(lender, 0)
            if cnt < max_per_lender:
                keep.append(i)
                lender_counts[lender] = cnt + 1
        all_ids = [all_ids[i] for i in keep]
        all_docs = [all_docs[i] for i in keep]
        all_metas = [all_metas[i] for i in keep]

    retrieval_ms = (_time.perf_counter() - t0) * 1000
    if metrics_callback:
        metrics_callback({
            "retrieval_candidates": len(collections) * n_per_collection,
            "retrieval_after_mmr": len(all_ids),
            "retrieval_after_rerank": len(all_ids),
            "retrieval_after_expand": len(all_ids),
            "retrieval_ms": retrieval_ms,
            "embedding_calls": len(collections) * 2,
            "cross_encoder_calls": 0,
        })

    return {"ids": [all_ids], "documents": [all_docs], "metadatas": [all_metas]}


def answer_query(
    query: str,
    collection: Optional[str],
    chroma_path: str,
    n_results: int = 6,
    expand_neighbors: int = 1,
    use_rerank: bool = False,
    max_chars_per_doc: int = 2000,
    include_collection_context: bool = True,
    collection_max_docs: int = 150,
    collection_chars: int = 6000,
    num_ctx: int = 12288,
    num_predict: int = 512,
    compute_faithfulness: bool = True,
    metrics: Optional[PipelineMetrics] = None,
) -> Dict:
    m = metrics or PipelineMetrics()
    m.query = query
    m.run_id = str(uuid.uuid4())[:8]

    # 0) Query understanding (industry, lender, intent) for prompt context + retrieval
    intent_context = _understand_query(query, metrics=m)

    # Expand search query with extracted criteria (improves retrieval relevance)
    search_query = query
    extras = []
    if intent_context.get("industry"):
        extras.append(f"{intent_context['industry']} industry eligibility")
    if intent_context.get("state"):
        extras.append(f"{intent_context['state']} state restrictions")
    if intent_context.get("revenue_monthly"):
        extras.append("revenue minimum monthly requirements")
    if intent_context.get("positions") is not None:
        extras.append("positions auto decline eligibility")
    if extras:
        search_query = f"{query} {' '.join(extras)}"

    # 1) Retrieve top evidence
    use_multi_lender = (
        intent_context.get("intent") == "which_lender"
        and collection is None
    )

    def on_retrieval(stats: Dict) -> None:
        m.retrieval_candidates = stats.get("retrieval_candidates", 0)
        m.retrieval_after_mmr = stats.get("retrieval_after_mmr", 0)
        m.retrieval_after_rerank = stats.get("retrieval_after_rerank", 0)
        m.retrieval_after_expand = stats.get("retrieval_after_expand", 0)
        m.retrieval_ms = stats.get("retrieval_ms", 0)
        m.embedding_calls = stats.get("embedding_calls", 0)
        m.cross_encoder_calls = stats.get("cross_encoder_calls", 0)

    if use_multi_lender:
        m.collection = "multi"
        results = _multi_collection_search(
            query_text=search_query,
            chroma_path=chroma_path,
            n_per_collection=2,
            n_total=min(n_results * 2, 10),
            max_per_lender=1,
            user_criteria=intent_context,
            metrics_callback=on_retrieval,
        )
        # No lender filter—we want docs from multiple lenders
    else:
        resolved = _resolve_collection(collection, chroma_path) if collection else None
        chosen_collection = resolved or _detect_collection_for_query(
            search_query, chroma_path, default_collection="lender-alternative-funding-group"
        )
        m.collection = chosen_collection
        results = semantic_query(
            query_text=search_query,
            collection_name=chosen_collection,
            chroma_path=chroma_path,
            n_results=n_results,
            mmr=True,
            rerank=use_rerank,
            expand_neighbors=expand_neighbors,
            metrics_callback=on_retrieval,
        )
        expected_slug = chosen_collection.replace("lender-", "")
        results = _filter_results_by_lender(results, expected_slug)
        results = _filter_cross_lender_mentions(results, expected_slug, chroma_path)
    m.retrieval_after_filter = len(results.get("ids", [[]])[0]) if results else 0
    notify("on_retrieval_end", m)
    # If nothing survives filtering, fail fast with a grounded message
    if not results.get("ids") or not results["ids"][0]:
        m.error = "no_results_after_filter"
        notify("on_pipeline_end", m)
        err_slug = "any lender" if use_multi_lender else expected_slug
        return {
            "json": {"used_sources": 0},
            "answer_text": f"No lender-specific results found for '{err_slug}'. Please verify the lender name or try a different query.",
            "metrics": m,
        }

    # 2) Compact sources for prompt
    sources_block = _format_sources(results, max_chars_per_doc=max_chars_per_doc)

    # Distilled bullets for which_lender multi-lender: gives LLM a compact overview before full sources
    background = None
    if use_multi_lender and intent_context.get("intent") == "which_lender":
        background = _distill_sources(results, max_bullets=25)

    # 3) Ask LLM for structured JSON. Two-field schema for which_lender forces complete lender list.
    use_two_field = use_multi_lender and intent_context.get("intent") == "which_lender"
    if use_two_field:
        json_schema_hint = (
            '{"intro": "Direct intro.", "lenders": ["Lender [S#]: copy exact positions + revenue from source, then fit note."], '
            '"used_sources": 0}'
        )
    else:
        json_schema_hint = '{"answer": "chat-style reply with [S#] citations.", "used_sources": 0}'
    messages = _build_messages(query, sources_block, json_schema_hint, intent_context=intent_context, background=background)
    prompt_text = " ".join(str(m.get("content", "")) for m in messages)
    m.prompt_tokens_approx = _approx_tokens(prompt_text)
    notify("on_llm_start", m)

    try:
        t0 = time.perf_counter()
        raw = _invoke_llm(messages, num_ctx=num_ctx, num_predict=num_predict)
        m.llm_ms = (time.perf_counter() - t0) * 1000
        m.completion_tokens_approx = _approx_tokens(raw)
    except requests.HTTPError as e:
        # Retry on server error with trimmed prompt/context
        status = getattr(e.response, "status_code", None)
        if status == 500:
            try:
                trimmed_doc_chars = max(800, int(max_chars_per_doc * 0.6))
                trimmed_coll_chars = max(3000, int(collection_chars * 0.6))
                trimmed_ctx = min(num_ctx, 8192)

                # Rebuild prompt smaller
                sources_block = _format_sources(results, max_chars_per_doc=trimmed_doc_chars)
                background = None
                if include_collection_context and not use_multi_lender:
                    try:
                        client_bg = PersistentClient(path=chroma_path)
                        col_bg = client_bg.get_collection(chosen_collection)
                        got = col_bg.get(limit=collection_max_docs)
                        docs_bg = got.get("documents", [])
                        metas_bg = got.get("metadatas", [])
                        acc: List[str] = []
                        total = 0
                        for d, meta in zip(docs_bg, metas_bg):
                            section = meta.get("section") if isinstance(meta, dict) else None
                            prefix = f"[{section}] " if section else ""
                            snippet = (d or "")[:300]
                            piece = prefix + snippet
                            if total + len(piece) > trimmed_coll_chars:
                                break
                            acc.append(piece)
                            total += len(piece)
                        background = "\n".join(acc)
                    except Exception:
                        background = None

                messages = _build_messages(query, sources_block, json_schema_hint, intent_context=intent_context, background=background)
                m.llm_retries += 1
                t0 = time.perf_counter()
                raw = _invoke_llm(messages, num_ctx=trimmed_ctx, num_predict=num_predict)
                m.llm_ms += (time.perf_counter() - t0) * 1000
                m.completion_tokens_approx = _approx_tokens(raw)
            except Exception:
                raise
        else:
            raise

    notify("on_llm_end", m)

    # 4) Parse JSON robustly (extract block, unwrap nested JSON)
    try:
        obj = json.loads(raw)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                obj = json.loads(match.group(0))
            except Exception:
                obj = {"answer": raw, "used_sources": 0}
        else:
            obj = {"answer": raw, "used_sources": 0}

    # Unwrap answer: support two-field (intro + lenders) or single-field (answer)
    raw_ans = obj.get("answer")
    intro = (obj.get("intro") or "").strip() if isinstance(obj.get("intro"), str) else ""
    lenders = obj.get("lenders") if isinstance(obj.get("lenders"), list) else []

    if intro or lenders:
        parts = []
        if intro:
            parts.append(intro)
        if lenders:
            parts.append("\n".join(str(x).strip() for x in lenders if x))
        answer_text = "\n\n".join(parts).strip()
    elif isinstance(raw_ans, list):
        answer_text = "\n".join(str(x) for x in raw_ans).strip()
    else:
        answer_text = (raw_ans or "").strip() if isinstance(raw_ans, str) else ""
    if answer_text and answer_text.startswith("{") and '"answer"' in answer_text[:200]:
        try:
            inner = json.loads(answer_text)
            if isinstance(inner, dict) and "answer" in inner:
                a = inner.get("answer")
                answer_text = "\n".join(str(x) for x in a).strip() if isinstance(a, list) else (a or "").strip()
        except Exception:
            pass

    if not answer_text:
        answer_text = "No summary produced from sources."

    # 5) Determine sources count (prefer model's used_sources; fallback to retrieved count)
    used_sources = obj.get("used_sources")
    if not isinstance(used_sources, int) or used_sources <= 0:
        total = len(results.get("ids", [[]])[0]) if results else 0
        used_sources = min(total, n_results)
        obj["used_sources"] = used_sources

    m.sources_used = used_sources
    m.answer_length = len(answer_text)

    # Faithfulness: NLI entailment (answer grounded in sources)
    if compute_faithfulness and results and answer_text:
        try:
            score, unsupported, elapsed = _compute_faithfulness_score(answer_text, results)
            m.faithfulness = score
            m.faithfulness_ms = elapsed
            m.unsupported_sentences = unsupported if unsupported else None
        except Exception as e:
            err_msg = str(e)
            get_logger("faithfulness").warning("Faithfulness computation failed: %s", err_msg)
            m.faithfulness = None
            m.faithfulness_ms = 0.0
            m.faithfulness_error = err_msg[:200]
            m.unsupported_sentences = None

    notify("on_pipeline_end", m)

    return {"json": obj, "answer_text": answer_text, "metrics": m}

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parent.parent
    _DEFAULT_CHROMA = str(_ROOT / "chroma_db")

    p = argparse.ArgumentParser(description="RAG -> Ollama structured answer")
    p.add_argument("query")
    p.add_argument("--collection", default=None)
    p.add_argument("--chroma", default=_DEFAULT_CHROMA, help="Chroma DB path")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--expand", type=int, default=1)
    p.add_argument("--rerank", action="store_true", default=False)
    p.add_argument("--doc-chars", type=int, default=2000, help="Max chars per source doc in prompt")
    p.add_argument("--num-ctx", type=int, default=12288, help="Ollama context window")
    p.add_argument("--num-predict", type=int, default=512, help="Max tokens to generate")
    p.add_argument("--with-collection", action="store_true", default=True, help="Include lender-wide background context")
    p.add_argument("--coll-maxdocs", type=int, default=150, help="Max docs sampled from collection for background")
    p.add_argument("--coll-chars", type=int, default=6000, help="Max total chars of background context")
    p.add_argument("--metrics", action="store_true", help="Enable logging + JSONL metrics output")
    p.add_argument("--log-dir", type=Path, default=_ROOT / "logs", help="Log directory")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--tier", choices=["minimal", "balanced", "full"], default="minimal",
                    help="Cost/performance tier: minimal (fast, default), balanced, full (best quality)")
    p.add_argument("--no-faithfulness", action="store_true", help="Disable NLI faithfulness scoring")
    args = p.parse_args()

    if args.tier == "minimal":
        args.n = 3
        args.expand = 0
        args.rerank = False
        args.doc_chars = 1200
        args.num_ctx = 8192
        args.num_predict = 384
    elif args.tier == "full":
        args.n = 8
        args.expand = 2
        args.rerank = True
        args.doc_chars = 2500
        args.num_ctx = 16384
        args.num_predict = 768

    if args.metrics:
        clear_strategies()
        setup_logging(
            level=getattr(logging, args.log_level),
            log_dir=args.log_dir,
            log_file="rag_pipeline.log",
            console=True,
        )
        register_strategy(LoggingStrategy())
        register_strategy(JsonFileStrategy(args.log_dir / "rag_metrics.jsonl"))
        register_strategy(CostAwareStrategy())

    out = answer_query(
        args.query,
        args.collection,
        args.chroma,
        n_results=args.n,
        expand_neighbors=args.expand,
        use_rerank=args.rerank,
        max_chars_per_doc=args.doc_chars,
        include_collection_context=args.with_collection,
        collection_max_docs=args.coll_maxdocs,
        collection_chars=args.coll_chars,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        compute_faithfulness=not args.no_faithfulness,
    )
    # Print count and answer
    used = out.get("json", {}).get("used_sources")
    try:
        used_int = int(used) if used is not None else 0
    except Exception:
        used_int = 0
    print(f"Sources used: {used_int}")
    if args.metrics and "metrics" in out:
        m = out["metrics"]
        faith = f" faithfulness={m.faithfulness:.2f}" if m.faithfulness is not None else ""
        print(f"Run: {m.run_id} | retrieval_ms={m.retrieval_ms:.0f} llm_ms={m.llm_ms:.0f} tokens~{m.prompt_tokens_approx}+{m.completion_tokens_approx}{faith}")
    print()
    ans_text = (out.get("answer_text", "") or "").strip()
    ans_text = _polish_answer(ans_text)
    print(ans_text)