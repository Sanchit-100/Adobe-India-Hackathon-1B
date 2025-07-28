"""
universal_pipeline.py
  1) Extract sections from PDFs via headings.json
  2) Build & cache BM25 (rank_bm25) and FAISS (vector) indices
  3) Query: merge Top-K BM25 + Top-K FAISS, re-rank + MMR, return Top-5
"""

import os, json, hashlib, pickle
import fitz                                # PyMuPDF
import numpy as np
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
# path to pre-downloaded embedding model
EMBED_MODEL_DIR = Path(__file__).resolve().parent / "embed_model"

# Add CLI and extractor integration
import argparse
import extract_headings
import glob

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Remove static paths: INPUT_DIR, HEAD_JSON, OUTPUT_DIR
CACHE_DIR   = "cache"
DATA_JSON   = os.path.join(CACHE_DIR, "data.json")
BM25_CACHE  = os.path.join(CACHE_DIR, "bm25_{digest}.pkl")
FAISS_CACHE = os.path.join(CACHE_DIR, "faiss_{digest}.index")
IDMAP_CACHE = os.path.join(CACHE_DIR, "idmap_{digest}.pkl")
TOP_BM25    = 50   # how many BM25 candidates
TOP_FAISS   = 50   # how many vector candidates
FINAL_TOPK  = 5    # how many to return
LEX_WEIGHT  = 0.6  # how to merge BM25 vs vector
MODEL_NAME  = "all-MiniLM-L6-v2"

# ─── 1. EXTRACTOR ─────────────────────────────────────────────────────────────
def extract_records(input_dir, headings_json, out_json):
    os.makedirs(CACHE_DIR, exist_ok=True)
    # Load headings file
    with open(headings_json, encoding="utf-8") as f:
        outlines = json.load(f)
    records = []
    for fname, meta in outlines.items():
        path = os.path.join(input_dir, fname)
        if not os.path.exists(path): continue
        doc = fitz.open(path)
        for node in meta["outline"]:
            page = node.get("page", 1) - 1
            title = node.get("text", "").strip()
            text = doc.load_page(page).get_text("text")
            records.append({
                "id": f"{fname}:{page+1}",
                "doc": fname,
                "page": page+1,
                "title": title,
                "text": text
            })
        doc.close()
    # cache data.json
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f)
    return records

# ─── 2. INDEXER ───────────────────────────────────────────────────────────────
def build_indices(records):
    # compute digest of data.json to key caches
    digest = hashlib.md5(json.dumps(records, sort_keys=True).encode()).hexdigest()
    bm25_file  = BM25_CACHE.format(digest=digest)
    faiss_file = FAISS_CACHE.format(digest=digest)
    idmap_file = IDMAP_CACHE.format(digest=digest)

    if os.path.exists(bm25_file) and os.path.exists(faiss_file) and os.path.exists(idmap_file):
        bm25   = pickle.load(open(bm25_file, "rb"))
        index  = faiss.read_index(faiss_file)
        id_map = pickle.load(open(idmap_file, "rb"))
    else:
        # BM25 on titles
        tokenized = [rec["title"].split() for rec in records]
        bm25 = BM25Okapi(tokenized)

        # FAISS on title embeddings using local model
        model = SentenceTransformer(str(EMBED_MODEL_DIR))
        titles = [rec["title"] for rec in records]
        emb = model.encode(titles, convert_to_numpy=True)
        dim = emb.shape[1]
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(dim)
        index.add(emb)

        id_map = [rec["id"] for rec in records]

        # write caches
        pickle.dump(bm25, open(bm25_file, "wb"))
        faiss.write_index(index, faiss_file)
        pickle.dump(id_map, open(idmap_file, "wb"))

    return bm25, index, id_map

# ─── 3. QUERYER ────────────────────────────────────────────────────────────────
def query_pipeline(query, records, bm25, index, id_map):
    # 3.1 lexical scores
    q_tokens = query.split()
    lex_scores = bm25.get_scores(q_tokens)
    top_lex = np.argsort(lex_scores)[::-1][:TOP_BM25]

    # 3.2 vector scores (fixed)
    # use local embedding model
    model = SentenceTransformer(str(EMBED_MODEL_DIR))
    # encode → float32 → reshape to (1, d)
    q_emb = model.encode(query, convert_to_numpy=True).astype('float32')
    q_emb = np.expand_dims(q_emb, axis=0)  
    # normalize in-place
    faiss.normalize_L2(q_emb)
    # now search with shape (1, d)
    D, I = index.search(q_emb, TOP_FAISS)
    top_vec = I[0]

    # 3.3 merge candidates
    cand = list(dict.fromkeys(list(top_lex) + list(top_vec)))
    merged_scores = []
    # normalize scores
    lex_range = np.ptp(lex_scores)
    lex_norm  = (lex_scores - lex_scores.min()) / (lex_range if lex_range != 0 else 1)

    vec_scores = D[0]
    vec_range  = np.ptp(vec_scores)
    vec_norm   = (vec_scores - vec_scores.min()) / (vec_range if vec_range != 0 else 1)

    for idx in cand:
        l = lex_norm[idx]
        if idx in top_vec:
            v = vec_norm[list(top_vec).index(idx)]
        else:
            v = 0.0
        merged_scores.append((LEX_WEIGHT * l + (1-LEX_WEIGHT) * v, idx))

    # 3.4 pick top FINAL_TOPK + simple MMR (one-per-doc)
    merged_scores.sort(reverse=True)
    selected, used_docs = [], set()
    for score, idx in merged_scores:
        doc = records[idx]["doc"]
        if doc in used_docs: continue
        selected.append(idx)
        used_docs.add(doc)
        if len(selected) == FINAL_TOPK: break

    # build output entries
    out = []
    for rank, idx in enumerate(selected, start=1):
        rec = records[idx]
        out.append({
            "document": rec["doc"],
            "section_title": rec["title"],
            "page_number": rec["page"],
            "importance_rank": rank
        })
    return out

# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multi-collection PDF analysis")
    parser.add_argument("input_json", help="Path to challenge input JSON")
    parser.add_argument("pdf_dir", help="Directory containing PDF documents")
    parser.add_argument("output_json", help="Path to write challenge output JSON")
    parser.add_argument("--model", help="Optional model path for heading extractor", default=None)
    args = parser.parse_args()

    # Load challenge configuration
    with open(args.input_json, encoding="utf-8") as f:
        cfg = json.load(f)
    persona = cfg["persona"]["role"]
    job = cfg["job_to_be_done"]["task"]

    # Prepare cache and headings paths
    pdf_dir = args.pdf_dir
    cache_dir = CACHE_DIR
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    headings_dir = Path(cache_dir) / "headings"
    headings_dir.mkdir(parents=True, exist_ok=True)
    combined_headings = Path(cache_dir) / "headings.json"

    # Extract headings for each PDF
    extract_headings.extract_all_headings(pdf_dir, headings_dir, args.model)

    # Combine per-PDF heading JSONs into single mapping
    mapping = {}
    for js in glob.glob(str(headings_dir / "*.json")):
        name = Path(js).stem + ".pdf"
        mapping[name] = json.load(open(js, encoding="utf-8"))
    with open(combined_headings, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)

    # 1) Extract records
    records = extract_records(pdf_dir, str(combined_headings), DATA_JSON)
    # 2) Index
    bm25, index, id_map = build_indices(records)
    # 3) Query
    query = f"{persona}. {job}"
    sections = query_pipeline(query, records, bm25, index, id_map)

    # Summaries: first 3 sentences from page text
    import nltk
    nltk.download("punkt", quiet=True)
    subs = []
    for s in sections:
        rec = next(r for r in records if r["doc"]==s["document"] and r["page"]==s["page_number"])
        text = rec["text"].replace("\n"," ")
        summ = " ".join(nltk.sent_tokenize(text)[:3])
        subs.append({
            "document": s["document"],
            "refined_text": summ,
            "page_number": s["page_number"]
        })

    # final JSON
    out = {
        "metadata": {
            "input_documents": sorted({r["doc"] for r in records}),
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": __import__("datetime").datetime.now().isoformat()
        },
        "extracted_sections": sections,
        "subsection_analysis": subs
    }
    # Write output
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    print("✅ Done →", args.output_json)