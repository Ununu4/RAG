# rag_build_lenders_optimized.py
# Purpose: Extract, chunk, and embed lender text files with precision and persistence
from __future__ import annotations
import os, re, argparse
from pathlib import Path
from typing import List, Tuple, Dict
from chonkie import FileFetcher, TextChef, SentenceChunker, EmbeddingsRefinery
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

DEFAULT_DIR = r"C:\Users\ottog\desktop\extracted_pdfs"
DEFAULT_CHROMA = r"C:\Users\ottog\desktop\Chromaa"
# Use a QA-tuned model for better retrieval semantics
EMBED_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

# ---------- Extraction ----------
def parse_lender_file(text: str, file_path: Path) -> tuple[dict, str]:
    """Extract metadata and content body from lender txt file."""
    lender_id = None
    lender_name = None
    source_file = None

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("### LENDER_ID:"):
            lender_id = re.sub(r"^### LENDER_ID:\s*", "", line)
        elif line.startswith("### LENDER_NAME:"):
            lender_name = re.sub(r"^### LENDER_NAME:\s*", "", line)
        elif line.startswith("### SOURCE_FILE:"):
            source_file = re.sub(r"^### SOURCE_FILE:\s*", "", line)
        elif line.startswith("-----"):
            break

    # everything after the dashed line is body content
    body_match = re.split(r"-{10,}", text, maxsplit=1)
    body_text = body_match[1].strip() if len(body_match) > 1 else text.strip()

    metadata = {
        "lender_id": lender_id or file_path.stem,
        "lender_name": lender_name or file_path.stem,
        "source_file": source_file or file_path.name
    }
    return metadata, body_text


# ---------- Utility ----------
def slugify(name: str) -> str:
    return re.sub(r"[^\w\-]+", "-", name.lower()).strip("-")[:64]


# ---------- Main Process ----------
def process_directory(txt_dir: str, chroma_path: str):
    os.makedirs(chroma_path, exist_ok=True)
    fetcher = FileFetcher()
    files = fetcher.fetch(dir=txt_dir, ext=[".txt"])
    chef = TextChef()
    docs = chef.process_batch([str(p) for p in files])

    # Smaller, sentence-aware chunks for better semantic precision
    chunker = SentenceChunker(chunk_size=180, chunk_overlap=40)
    refinery = EmbeddingsRefinery(embedding_model=EMBED_MODEL)
    client = PersistentClient(path=chroma_path)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    def is_heading(line: str) -> bool:
        l = line.strip()
        if not l:
            return False
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
                title = re.sub(r"^\*\*|\*\*$", "", line).strip()
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
        lender_name = slugify(metadata["lender_name"])
        collection_name = f"lender-{lender_name}"

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
            sec_chunks = chunker.chunk_batch([content])[0]
            # Assign metadata and prefix with section title to strengthen semantics
            for c_idx, ch in enumerate(sec_chunks):
                base_text = ch.text.strip()
                # Skip trivial bullets or numbering-only lines
                if re.fullmatch(r"\d+[.)]?", base_text) or base_text in {"-", "•"}:
                    continue
                prefix = f"{title}: " if title and title.lower() != "intro" else ""
                text_with_prefix = (prefix + base_text).strip()

                # Deduplicate near-duplicates by hash of normalized text
                norm = re.sub(r"\s+", " ", text_with_prefix.lower()).strip()
                h = hash(norm)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                sec_meta = {**metadata,
                            "section": title or "",
                            "section_index": s_idx,
                            "chunk_index": c_idx,
                            # Store tags as a comma-separated string to satisfy Chroma metadata type constraints
                            "tags": ",".join(detect_tags(title + "\n" + base_text))}
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
        print(f"✅ {len(ids)} chunks → {collection_name}")

    print("\n✅ All lenders processed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build lender embeddings with precise extraction")
    parser.add_argument("--dir", default=DEFAULT_DIR)
    parser.add_argument("--chroma", default=DEFAULT_CHROMA)
    args = parser.parse_args()
    process_directory(args.dir, args.chroma)

