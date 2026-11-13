"""Grounded answer workflow for the synthetic screening Pap claim demo."""

import re
from contextlib import nullcontext
from datetime import datetime

from .claims import valid_claim_id
from .fixtures import load_fixture
from .model import PROMPT_VERSION, FixtureModel
from .retrieval import search_bm25

PREFIX = "cms-100-04-18-"
RULES = {
    "service": ("30.5.A-q0091", "Q0091 describes collecting, preparing, and conveying a screening Pap specimen to the laboratory."),
    "interval": ("30.2-interval", "For routine screening, 23 full months must pass after the end of the month of the last covered smear."),
    "remit": ("30.9-frequency", "The 119/M83 pair is consistent with a line-level frequency denial when high-risk factors are absent."),
    "diagnoses": ("30.6.A-diagnoses", "Item 21 of CMS-1500, or its electronic equivalent, records an applicable diagnosis that reflects the documented condition."),
    "pointer": ("30.6.A-pointer", "Item 24E, or its electronic equivalent, points to an applicable diagnosis for the service line. CMS describes a Common Working File rejection when the required pointer is missing."),
}
OUTSIDE = re.compile(r"\b(hpv|g0476|hospital|institutional|medicare advantage|annual high-risk|local contractor)\b")
WRITE = re.compile(r"^\s*(change|update|submit|delete|write|modify|resubmit)\b")
LOOKUP = re.compile(r"^\s*look up\s+(?:the service month for\s+)?(\S+)", re.IGNORECASE)
MONTH = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b", re.IGNORECASE)


def topics(question: str, claim: dict | None) -> list[str]:
    rules = []
    if re.search(r"\b(119|m83|high-risk status|benefit limit)\b", question):
        rules.append("remit")
    if re.search(r"\b(interval|how long|too early|start counting|waiting rule|timing requirement)\b", question):
        rules.append("interval")
    if claim and "remit" in rules and re.search(r"\b(why|month|last covered)\b", question):
        rules.append("interval")
    if re.search(r"\b(item 21|diagnosis field|low-risk list|low-risk diagnoses|z\d{2}\.\d+)\b", question):
        rules.append("diagnoses")
    if re.search(r"\b(24e|pointer|points? to|link the|connect the)\b", question):
        rules.append("pointer")
    if (re.search(r"\b(q0091|taking the sample)\b", question)
            and ("remit" not in rules or re.search(r"what is q0091|what does q0091", question))):
        rules.append("service")
    return list(dict.fromkeys(rules))


def _month_index(value: str, pattern: str) -> int:
    date = datetime.strptime(value, pattern)
    return 12 * date.year + date.month


class Assistant:
    def __init__(self, *, model=None, trace=None, search=None, limit: int = 5):
        self.corpus = load_fixture("cms_ch18.json")
        self.model = model if model is not None else FixtureModel()
        self.trace = trace if trace is not None else lambda *_: nullcontext()
        self.search = search or search_bm25
        self.limit = limit

    async def ask(self, question: str, *, allowed: bool = False, test_context=None) -> dict:
        with self.trace("assistant", "CHAIN") as root:
            result = await self._answer(question, allowed, test_context or {})
            if root:
                root.set_attributes({"corpus": self.corpus["version"], "prompt_version": PROMPT_VERSION,
                                     "model": self.model.name})
                root.set_outputs({
                    "status": result["status"],
                    "citation_ids": [c["id"] for c in result["citations"]],
                    "lookup_status": result["tool"].get("status"),
                })
                result["trace_id"] = root.trace_id
            return result

    async def _answer(self, question: str, allowed: bool, context: dict) -> dict:
        tool = {"dispatched": False, "fixture_reads": 0}
        result = {"tool": tool, "citations": [], "statements": [], "usage": {
            "calls": 0, "input_tokens": 0, "output_tokens": 0,
        }}

        def finish(status, answer):
            return {**result, "status": status, "answer": answer}

        query = question.casefold()
        claim = None
        match = LOOKUP.match(question)
        if WRITE.match(query):
            return finish("denied", "This workflow supports only authorized reads of synthetic claims.")
        if OUTSIDE.search(query):
            return finish("abstain", "That question is outside the office Q0091 routine-frequency workflow.")
        if match:
            claim_id = match[1].rstrip(".,;!?")
            with self.trace("claim_lookup", "TOOL") as span:
                if not allowed:
                    tool["status"] = "denied"
                elif not valid_claim_id(claim_id) or re.search(r"\bsyn-", query[match.end():]):
                    tool["status"] = "invalid_id"
                else:
                    from .mcp_claims import lookup_claim

                    try:
                        response = await lookup_claim(claim_id, server_allowed=True)
                    except (RuntimeError, OSError, ExceptionGroup):
                        response = {"status": "unavailable", "fixture_reads": 0}
                    tool.update(dispatched=True, claim_id=claim_id, status=response["status"],
                                fixture_reads=response["fixture_reads"])
                    claim = response.get("claim")
                if span:
                    span.set_outputs({key: value for key, value in tool.items() if key != "claim_id"})
            if tool["status"] != "ok":
                return finish(tool["status"], "The synthetic claim could not be retrieved.")
        elif re.search(r"\bsyn-", query):
            return finish("abstain", "Explicitly request lookup of one synthetic claim ID.")

        wanted = topics(query, claim)
        if not wanted:
            if claim and "service month" in query:
                return finish("lookup_result", f"Synthetic service month: {claim['service_month']}.")
            return finish("abstain", "The pinned CMS passages do not support an answer to this question.")

        excluded = set(context.get("exclude_passage_ids", []))
        canonical = {
            p["id"]: p for p in self.corpus["passages"] if p["id"] not in excluded
        }
        with self.trace("retrieval", "RETRIEVER") as span:
            if span:
                span.set_inputs({"candidate_count": len(canonical), "limit": self.limit,
                                 "method": self.search.__name__})
            hits = self.search(question, list(canonical.values()), limit=self.limit)
            if context.get("insert_untrusted_text"):
                # Simulate a poisoned retrieval result, including a forged canonical ID.
                forged_id = PREFIX + RULES[wanted[0]][0]
                hits.insert(0, {"id": forged_id, "text": context["insert_untrusted_text"]})
            evidence = {}
            rejected = 0
            for hit in hits:
                source = canonical.get(hit["id"])
                if source and all(hit.get(key) == value for key, value in source.items()):
                    evidence[hit["id"]] = source
                else:
                    rejected += 1
            required = [PREFIX + RULES[topic][0] for topic in wanted]
            if span:
                span.set_outputs({"ranked_ids": list(evidence), "rejected_passages": rejected,
                                  "missing_ids": [key for key in required if key not in evidence]})
            result["rejected_passages"] = rejected

        if any(key not in evidence for key in required):
            return finish("abstain", "A required passage is missing from the retrieved CMS evidence.")

        statements = [{"text": RULES[topic][1], "evidence_ids": [key]}
                      for topic, key in zip(wanted, required)]
        if "diagnoses" in wanted:
            codes = re.findall(r"\bz\d{2}\.\d+\b", query)
            low_risk = {"z01.411", "z01.419", "z12.4", "z12.72", "z12.79", "z12.89"}
            if codes and not set(codes) <= low_risk:
                return finish("abstain", "The question requests a diagnosis outside the routine screening list.")
            for code in dict.fromkeys(codes):
                statements.append({"text": f"CMS lists {code.upper()} as a low-risk diagnosis.",
                                   "evidence_ids": [PREFIX + RULES["diagnoses"][0]]})

        qualifier = ""
        if "interval" in wanted and claim:
            prior = claim["previous_covered_screening_month"]
            if not prior:
                qualifier = "The prior covered screening month is missing. I cannot determine whether this claim meets the interval."
            elif claim["frequency_category"] != "routine" or claim["hcpcs"] != "Q0091":
                qualifier = "The claim does not establish an office Q0091 routine-frequency scenario."
            else:
                months = _month_index(claim["service_month"], "%Y-%m") - _month_index(prior, "%Y-%m")
                if months <= 0:
                    qualifier = "The supplied screening chronology does not support a claim-specific timing conclusion."
                else:
                    timing = "meets" if months - 1 >= 23 else "falls short of"
                    statements.append({
                        "text": f"The synthetic claim shows {prior} and {claim['service_month']}. "
                                f"{months - 1} full months elapsed, which {timing} the routine interval.",
                        "evidence_ids": [PREFIX + RULES["interval"][0]],
                    })
        elif "interval" in wanted:
            dates = MONTH.findall(question)
            if len(dates) == 2:
                months = _month_index(dates[1], "%B %Y") - _month_index(dates[0], "%B %Y")
                if months <= 0:
                    qualifier = "The supplied screening chronology does not support a claim-specific timing conclusion."
                else:
                    timing = "meets" if months - 1 >= 23 else "falls short of"
                    statements.append({
                        "text": f"On the supplied dates, {months - 1} full months elapsed, which {timing} the routine interval.",
                        "evidence_ids": [PREFIX + RULES["interval"][0]],
                    })

        with self.trace("answer_model", "LLM") as span:
            order, usage = self.model.order(question, statements)
            if span:
                span.set_inputs({"statement_count": len(statements)})
                span.set_attributes({"model": self.model.name, **usage, "mlflow.chat.tokenUsage": {
                    "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
                    "total_tokens": usage["input_tokens"] + usage["output_tokens"],
                }})
                span.set_outputs({"statement_order": order})
        result["usage"] = usage
        result["statements"] = [statements[i] for i in order]
        result["citations"] = [
            {name: evidence[key][name] for name in ("id", "section", "pdf_page", "source_url")}
            for key in required
        ]
        rendered = " ".join(statement["text"] for statement in result["statements"])
        return finish("partial" if qualifier else "answer", " ".join(filter(None, [rendered, qualifier])))
