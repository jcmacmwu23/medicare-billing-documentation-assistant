"""Rebuild the indexed excerpts from the pinned CMS PDF using pypdf."""

import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sources/cms-100-04-ch18.pdf"
TARGET = ROOT / "src/billing_assistant/fixtures/cms_ch18.json"
SHA256 = "e67c515cbe53f6f327931b7da9b37fee7c4fa55cb43cdbabc2edd3a405a065f6"
URL = "https://www.cms.gov/regulations-and-guidance/guidance/manuals/downloads/clm104c18.pdf"

# Page numbers are one-based PDF positions. Start and end markers are literal
# text from the document after joining wrapped lines.
EXCERPTS = [
    ("30.2", "interval", 75, "3329", "Pap Smears On and After July 1, 2001",
     "If the beneficiary does not qualify", "30.2.1"),
    ("30.5.A", "q0091", 77, "3460", "Screening Pap Smears: HCPCS Codes for Billing",
     "Q0091 - Screening Papanicolaou", "• P3001"),
    ("30.6.A", "diagnoses", 80, "11445", "Screening Pap Smears: Applicable Diagnoses for Billing A/B MAC (B)",
     "There are a number of appropriate diagnosis codes", "Then either the low-risk"),
    ("30.6.A", "pointer", 80, "11445", "Screening Pap Smears: Applicable Diagnoses for Billing A/B MAC (B)",
     "Then either the low-risk", "Periodically, A/B MACs"),
    ("30.9", "frequency", 83, "11021", "Screening Pap Smear Remittance Advice Codes",
     "Pap Smear Screening: If high risk", "HPV Screening:"),
]


def extract(page, start, end):
    # Replace soft wrapping within paragraphs, preserving paragraph breaks.
    text = page.extract_text()
    text = re.sub(r"\n[ \t]*\n", "\n\n", text)
    text = "\n\n".join(" ".join(paragraph.split()) for paragraph in text.split("\n\n"))
    first = text.index(start)
    last = text.index(end, first + len(start))
    return text[first:last].strip().replace("ICD-9- CM", "ICD-9-CM")


def main():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SHA256:
        raise SystemExit("CMS PDF checksum differs from the pinned source.")
    pdf = PdfReader(SOURCE)
    passages = [
        {
            "id": f"cms-100-04-18-{section}-{slug}",
            "section": section,
            "section_revision": revision,
            "heading": heading,
            "pdf_page": page,
            "source_url": f"{URL}#page={page}",
            "text": extract(pdf.pages[page - 1], start, end),
        }
        for section, slug, page, revision, heading, start, end in EXCERPTS
    ]
    corpus = {
        "version": "cms-100-04-ch18-r13709-v1",
        "status": "official_pdf_excerpts",
        "publication": "CMS Pub. 100-04",
        "chapter": 18,
        "chapter_revision_observed": "13709",
        "retrieved_on": "2026-11-13",
        "source_file": "sources/cms-100-04-ch18.pdf",
        "source_url": URL,
        "source_sha256": SHA256,
        "passages": passages,
    }
    TARGET.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
