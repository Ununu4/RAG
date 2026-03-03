# rag_build_lenders_optimized.py
# Purpose: Extract, chunk, and embed lender text files with precision and persistence
from __future__ import annotations
import os, re, argparse
from pathlib import Path
from typing import List, Tuple, Dict
from chonkie import FileFetcher, TextChef, SentenceChunker, EmbeddingsRefinery
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# Default to guidelines/ relative to project root (parent of pre_processing/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = str(_PROJECT_ROOT / "guidelines")
DEFAULT_CHROMA = str(_PROJECT_ROOT / "chroma_db")
# Use a QA-tuned model for better retrieval semantics
EMBED_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

# Chunking strategy (tokens; ~4 chars/token heuristic for char-based thresholds)
CHUNK_SIZE = 320
CHUNK_OVERLAP = 64
SMALL_SECTION_CHARS = 600   # ~150 tokens - keep whole
INDUSTRY_LIST_CHARS = 1600  # ~400 tokens - split long comma lists
INDUSTRY_BATCH_SIZE = 20

# ---------- Extraction ----------
# Canonical section order for guidelines schema
CANONICAL_SECTIONS = ("funding_guidelines", "eligibility_criteria", "deal_structure", "key_findings")


def _normalize_lender_slug(s: str, file_stem: str) -> str:
    """Derive clean lender slug; strip _guidelines suffix when from filename."""
    base = (s or file_stem).strip()
    base = re.sub(r"_guidelines$", "", base, flags=re.I)
    return base


def parse_lender_file(text: str, file_path: Path) -> tuple[dict, str]:
    """Extract metadata and content body from lender txt file.
    Supports:
    - Guidelines format: # LENDER: <slug> followed by ## section blocks
    - Legacy format: ### LENDER_ID/NAME/SOURCE_FILE, dashed line, then body
    """
    lender_id = None
    lender_name = None
    source_file = None
    body_text = text.strip()

    lines = text.splitlines()
    first_line = (lines[0] or "").strip()

    # Guidelines format: # LENDER: <slug>
    lender_match = re.match(r"^#\s*LENDER:\s*(.+)$", first_line, re.I)
    if lender_match:
        slug = lender_match.group(1).strip()
        lender_id = lender_name = _normalize_lender_slug(slug, file_path.stem)
        source_file = file_path.name
        # Body: everything after first line (skip # LENDER header)
        body_text = "\n".join(lines[1:]).strip()
        metadata = {
            "lender_id": lender_id,
            "lender_name": lender_name,
            "source_file": source_file,
        }
        return metadata, body_text

    # Legacy format: ### LENDER_ID:, ### LENDER_NAME:, ### SOURCE_FILE:, -----
    for line in lines:
        line = line.strip()
        if line.startswith("### LENDER_ID:"):
            lender_id = re.sub(r"^### LENDER_ID:\s*", "", line)
        elif line.startswith("### LENDER_NAME:"):
            lender_name = re.sub(r"^### LENDER_NAME:\s*", "", line)
        elif line.startswith("### SOURCE_FILE:"):
            source_file = re.sub(r"^### SOURCE_FILE:\s*", "", line)
        elif line.startswith("-----"):
            break

    body_match = re.split(r"-{10,}", text, maxsplit=1)
    body_text = body_match[1].strip() if len(body_match) > 1 else text.strip()

    stem = _normalize_lender_slug(None, file_path.stem)
    metadata = {
        "lender_id": lender_id or stem,
        "lender_name": lender_name or stem,
        "source_file": source_file or file_path.name,
    }
    return metadata, body_text


# ---------- Chunking Helpers ----------
_SUBHEADING_RE = re.compile(r"^\s*[-*]\s+\*\*([^*]+)\*\*:\s*", re.MULTILINE)


def _split_by_subheadings(text: str) -> List[Tuple[str, str]]:
    """Split section content by **SubHeading:** groups. Returns (subheading, content) tuples."""
    groups: List[Tuple[str, str]] = []
    current_heading = ""
    current_lines: List[str] = []

    for line in text.splitlines():
        m = _SUBHEADING_RE.match(line)
        if m:
            if current_lines:
                groups.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = m.group(1).strip()
            rest = line[m.end() :].strip()
            current_lines = [rest] if rest else []
        else:
            current_lines.append(line)

    if current_lines:
        groups.append((current_heading, "\n".join(current_lines).strip()))
    return groups if groups else [("", text.strip())]


def _maybe_split_industry_list(content: str, category_prefix: str) -> List[str]:
    """If content is a long comma-separated list, split into batches for retrieval."""
    if len(content) <= INDUSTRY_LIST_CHARS:
        return [content]
    parts = [p.strip() for p in re.split(r",|;", content) if p.strip()]
    if len(parts) < 10:
        return [content]
    batches: List[str] = []
    for i in range(0, len(parts), INDUSTRY_BATCH_SIZE):
        batch = parts[i : i + INDUSTRY_BATCH_SIZE]
        batch_text = ", ".join(batch)
        if category_prefix:
            batch_text = f"{category_prefix} (items {i + 1}-{i + len(batch)}): {batch_text}"
        batches.append(batch_text)
    return batches


class _SimpleChunk:
    """Minimal chunk object for refinery (needs .text, .metadata; gets .embedding)."""

    def __init__(self, text: str, metadata: Dict):
        self.text = text
        self.metadata = metadata


# ---------- Utility ----------
def slugify(name: str) -> str:
    """Normalize to hyphenated slug for collection names (e.g. alternative-funding-group)."""
    s = name.lower().replace("_", "-")
    s = re.sub(r"[^\w\-]+", "-", s).strip("-")
    return s[:64]


# ---------- Main Process ----------
def process_directory(txt_dir: str, chroma_path: str):
    os.makedirs(chroma_path, exist_ok=True)
    fetcher = FileFetcher()
    files = fetcher.fetch(dir=txt_dir, ext=[".txt"])
    chef = TextChef()
    docs = chef.process_batch([str(p) for p in files])

    # Hierarchical chunking: subheading-aware, optimized for guidelines structure
    chunker = SentenceChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    refinery = EmbeddingsRefinery(embedding_model=EMBED_MODEL)
    client = PersistentClient(path=chroma_path)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    def is_heading(line: str) -> bool:
        l = line.strip()
        if not l:
            return False
        # Guidelines schema: ## funding_guidelines, ## eligibility_criteria, etc.
        if re.match(r"^##\s+\S+", l):
            return True
        # Bold-style headings like **Eligibility:** or plain 'Eligibility:'
        if re.match(r"^\*\*?.+\*\*?:\s*$", l):
            return True
        if l.endswith(":") and len(l) <= 80:
            return True
        # Named sections common in lender docs
        named = [
            "Key Requirements", "Requirements", "Eligibility", "Restrictions",
            "Prohibitions", "Underwriting", "Funding Process", "Pre-Qualification",
            "Submission Requirements", "Underwriting Criteria", "Renewal", "Documents",
            "Rates", "Terms", "Minimums", "Maximums", "Fees"
        ]
        return any(n.lower() in l.lower() for n in named)

    def split_sections(text: str) -> List[Tuple[str, str]]:
        """Split body text into (section_title, section_text) pairs."""
        sections: List[Tuple[str, str]] = []
        current_title = "Intro"
        current_lines: List[str] = []
        for raw in text.splitlines():
            line = raw.rstrip()
            if is_heading(line):
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines).strip()))
                    current_lines = []
                # Strip ## markdown, **bold**, trailing colon
                title = re.sub(r"^##\s*", "", line)
                title = re.sub(r"^\*\*|\*\*$", "", title).strip()
                title = title.rstrip(":")
                current_title = title or current_title
            else:
                current_lines.append(line)
        if current_lines:
            sections.append((current_title, "\n".join(current_lines).strip()))
        # If no headings detected, return single Intro section
        return sections or [("Intro", text.strip())]

    def detect_tags(text: str) -> List[str]:
        t = text.lower()
        tags: List[str] = []
        keywords = [
            ("eligibility", ["eligibility", "qualify", "criteria", "must", "requirement", "minimum"]),
            ("fico", ["fico", "credit score"]),
            ("revenue", ["monthly revenue", "revenue", "sales", "avg daily balance", "adb"]),
            ("tib", ["tib", "time in business", "months in business", "years in business"]),
            ("nsf", ["nsf", "negative days", "overdraft"]),
            ("docs", ["documents", "statements", "application", "tax return", "financials"]),
            ("restrictions", ["restriction", "restricted", "prohibited", "not available"]),
            ("renewal", ["renewal", "add-on", "50% paid", "retain"]) ,
            ("rates", ["rate", "buy rate", "sell rate", "fees"]),
            ("position", ["1st position", "2nd position", "3rd position", "positions"]),
        ]
        for tag, terms in keywords:
            if any(term in t for term in terms):
                tags.append(tag)
        return tags

    def extract_numeric_hints(text: str) -> Dict[str, str]:
        """Extract simple numeric signals like min_fico, min_revenue, min_tib."""
        hints: Dict[str, str] = {}
        t = text
        # FICO / credit score
        m = re.search(r"(?:fico|credit\s+score)[^\d]{0,10}(\d{3})", t, re.I)
        if m:
            hints["min_fico"] = m.group(1)
        # Revenue like $30k or 30000
        m = re.search(r"\$\s?(\d+[\d,]*\s?[kKmM]?)\s*(?:monthly\s+)?(?:revenue|sales)", t, re.I)
        if m:
            hints["min_revenue"] = m.group(1)
        # TIB months/years
        m = re.search(r"(\d+)\s*(?:months?|mos?)\s*(?:in\s+business|tib)", t, re.I)
        if m:
            hints["min_tib_months"] = m.group(1)
        else:
            m = re.search(r"(\d+)\s*(?:years?|yrs?)\s*(?:in\s+business|tib)", t, re.I)
            if m:
                try:
                    hints["min_tib_months"] = str(int(m.group(1)) * 12)
                except Exception:
                    pass
        return hints

    for doc, file_path in zip(docs, files):
        metadata, body_text = parse_lender_file(doc.content, file_path)
        lender_slug = slugify(metadata["lender_name"])
        metadata["lender_name"] = lender_slug  # hyphenated slug for retrieval filtering
        collection_name = f"lender-{lender_slug}"

        sections = split_sections(body_text)
        augmented_chunks = []
        ids: List[str] = []
        metadatas: List[Dict] = []
        documents: List[str] = []
        embeddings: List[List[float]] = []

        # For simple deduplication
        seen_hashes: set[str] = set()

        for s_idx, (title, content) in enumerate(sections):
            if not content.strip():
                continue
            subheading_groups = _split_by_subheadings(content)
            section_chunk_list: List[object] = []

            for subheading, sub_content in subheading_groups:
                if not sub_content.strip():
                    continue
                section_prefix = f"{title}: " if title and title.lower() != "intro" else ""
                sub_prefix = f"{subheading}: " if subheading else ""
                full_prefix = (section_prefix + sub_prefix).strip()

                # Small section: keep whole
                if len(sub_content) < SMALL_SECTION_CHARS:
                    text_with_prefix = (full_prefix + " " + sub_content).strip() if full_prefix else sub_content
                    section_chunk_list.append(_SimpleChunk(text_with_prefix, {}))
                    continue

                # Long comma-separated list (e.g. industry lists): batch split
                industry_parts = _maybe_split_industry_list(sub_content, full_prefix if full_prefix else "")
                if len(industry_parts) > 1:
                    for part in industry_parts:
                        section_chunk_list.append(_SimpleChunk(part, {}))
                    continue

                # Default: sentence-aware chunking
                raw_chunks = chunker.chunk_batch([sub_content])[0]
                for ch in raw_chunks:
                    ch.text = (full_prefix + " " + ch.text).strip() if full_prefix else ch.text.strip()
                    section_chunk_list.append(ch)

            # Assign metadata, deduplicate, and build augmented_chunks
            for c_idx, ch in enumerate(section_chunk_list):
                base_text = ch.text.strip() if hasattr(ch, "text") else ""
                # Skip trivial bullets or numbering-only lines
                if re.fullmatch(r"\d+[.)]?", base_text) or base_text in {"-", "•"}:
                    continue
                # All chunks have section/subheading prefix applied at creation
                text_with_prefix = ch.text.strip()

                # Deduplicate near-duplicates by hash of normalized text
                norm = re.sub(r"\s+", " ", text_with_prefix.lower()).strip()
                h = hash(norm)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                # Canonical section index for guidelines schema ordering
                try:
                    canonical_idx = CANONICAL_SECTIONS.index(title)
                except ValueError:
                    canonical_idx = s_idx
                sec_meta = {
                    **metadata,
                    "section": title or "",
                    "section_index": s_idx,
                    "canonical_section_index": canonical_idx,
                    "chunk_index": c_idx,
                    "tags": ",".join(detect_tags(title + "\n" + base_text)),
                }
                sec_meta.update(extract_numeric_hints(base_text))

                setattr(ch, "metadata", sec_meta)
                setattr(ch, "text", text_with_prefix)
                augmented_chunks.append(ch)

        if not augmented_chunks:
            continue

        refined = refinery.refine(augmented_chunks)

        # Build stable IDs with section and chunk indices
        for ch in refined:
            s_idx = ch.metadata.get("section_index", 0)
            c_idx = ch.metadata.get("chunk_index", 0)
            ids.append(f"{collection_name}-{s_idx}-{c_idx}")
            documents.append(ch.text)
            embeddings.append(ch.embedding)
            metadatas.append(ch.metadata)

        # Link neighbors (parent-child style) within same section
        by_section: Dict[int, List[int]] = {}
        for i, m in enumerate(metadatas):
            by_section.setdefault(m.get("section_index", 0), []).append(i)
        for idxs in by_section.values():
            idxs.sort()
            for j, i in enumerate(idxs):
                prev_i = idxs[j-1] if j > 0 else None
                next_i = idxs[j+1] if j < len(idxs)-1 else None
                if prev_i is not None:
                    metadatas[i]["prev_id"] = ids[prev_i]
                if next_i is not None:
                    metadatas[i]["next_id"] = ids[next_i]

        collection = client.get_or_create_collection(
            collection_name,
            embedding_function=embedding_fn,
            metadata={"lender_id": metadata["lender_id"], "lender_name": metadata["lender_name"]}
        )
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"[OK] {len(ids)} chunks -> {collection_name}")

    print("\n[OK] All lenders processed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build lender embeddings with precise extraction")
    parser.add_argument("--dir", default=DEFAULT_DIR)
    parser.add_argument("--chroma", default=DEFAULT_CHROMA)
    args = parser.parse_args()
    process_directory(args.dir, args.chroma)

