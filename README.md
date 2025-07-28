
Persona-Driven Document Intelligence
====================================

_A Hybrid Search Pipeline for Any PDF Collection_

---

Inspiration 🌟
--------------

Imagine drowning in a pile of PDFs—research papers, textbooks, reports—while racing a deadline. Our open-source pipeline transforms that chaos into clarity, surfacing the exact sections you need based on who you are (the persona) and what you need to accomplish.

---

Try it out
----------

```bash
git clone https://github.com/your-org/challenge-1b.git
cd challenge-1b
python universal_pipeline.py
````

---

## What it Does 💡

#### Problem Statement

Given 3–10 diverse PDFs, a **Persona** (role + expertise) and a **Job-to-be-Done** (task), users waste hours hunting for relevant sections. A generic, fast, CPU-only solution is required.

#### Solution

This open-source system:

1. **Extracts** section headings + raw page text.
2. **Indexes** titles via BM25 (keyword) and FAISS (semantic) using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
3. **Queries** by merging Top-K from both indices, re-ranking with weighted fusion (0.6 BM25 + 0.4 FAISS) and simple MMR (one section per document).
4. **Summarizes** each selected section into three sentences.
5. **Outputs** structured JSON matching the challenge spec.

---

## Detailed Description 📝

#### 1. Extraction

* Load `headings.json` mapping each PDF to its section titles and pages.
* Use PyMuPDF to extract full-page text for every listed heading.
* Build records:

  ```json
  { "id":"file.pdf:5", "doc":"file.pdf", "page":5, "title":"Section", "text":"…" }
  ```
* Cache to `cache/data.json` for faster repeated runs.

#### 2. Indexing

* Compute MD5 digest of `records` → cache key.
* **Cache hit**: load BM25 pickle, FAISS index, and ID map.
* **Cache miss**:

  1. **BM25**: tokenize each title, build a `BM25Okapi` index.
  2. **FAISS**:

     * Encode titles with `all-MiniLM-L6-v2` → NumPy embeddings.
     * L2-normalize vectors.
     * Create a FlatIP FAISS index and add embeddings.
     * Store an array mapping index positions → record IDs.
* Persist caches for sub-60 s startup on unchanged data.

#### 3. Querying

* **Lexical**: split persona+job query into tokens, compute BM25 scores, select Top-50.
* **Semantic**: encode query → normalize → FAISS `search` → Top-50 indices + similarity scores.
* **Merge & normalize**:

  * Union BM25 + FAISS hits (preserving order).
  * Peak-to-peak normalize both sets into \[0,1].
  * Compute combined score: `0.6*lex + 0.4*vec`.
* **Diversify**: pick Top-5 while skipping any document already chosen using Maximal Marginal Relevance(MMR) strategy.
* **Format**: produce entries like

  ```json
  { "document":"file.pdf", "section_title":"…", "page_number":5, "importance_rank":1 }
  ```

#### 4. Summarization & Output

* Use NLTK’s Punkt tokenizer to split each chosen record’s text into sentences and take the first three.
* Assemble JSON and write to `result.json`.

---

## Output Format

```json
{
  "metadata": {
    "input_documents": ["file1.pdf", "..."],
    "persona": "Researcher",
    "job_to_be_done": "…",
    "processing_timestamp": "2025-07-28T…"
  },
  "extracted_sections": [
    { "document":"file1.pdf", "section_title":"…", "page_number":2, "importance_rank":1 },
    …
  ],
  "subsection_analysis": [
    { "document":"file1.pdf", "page_number":2, "refined_text":"First sentence. Second sentence. Third sentence." },
    …
  ]
}
```

---

## What Sets It Apart 🌠

* **Fully open-source**: uses only free tools and the `all-MiniLM-L6-v2` transformer.
* **Hybrid retrieval**: combines exact BM25 with semantic FAISS embeddings.
* **Diverse coverage**: enforces one section per document to avoid redundancy.
* **Sub-60 s, CPU-only**: caches heavy steps for fast re-runs on commodity hardware.

```
```
