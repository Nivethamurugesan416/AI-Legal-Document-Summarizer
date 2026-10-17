# AI Legal Document Summarizer

A **reconstructed research prototype** based on Nivetha Murugesan's earlier legal-document summarization project. This is a fresh implementation informed by the documented project design, **not a recovered copy of the original source code**.

## Features
- Upload PDF, DOCX, or TXT documents
- Extract text and optionally OCR image-only PDF pages
- Summarize with LexRank (extractive)
- Optionally run T5-small or DistilBART CNN transformer inference
- Preview extracted text and download summaries

## Architecture

`Document → Text extraction (PyMuPDF/python-docx/OCR) → NLP summarizer (LexRank/T5/DistilBART) → Streamlit UI → Summary`

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab
streamlit run app.py
```

For transformer models (downloads model weights on first use):

```bash
pip install -r requirements-transformers.txt
```

For scanned PDFs:

```bash
pip install -r requirements-ocr.txt
```

Also install the **Tesseract OCR system executable** using your OS package manager. OCR is optional and disabled by default.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Research context

The original project explored extractive and abstractive summarization, long-document handling, and ROUGE/BLEU evaluation. This reconstructed demo provides **inference only**; it does not include the original fine-tuning pipeline, evaluation dataset, benchmark results, or AWS deployment infrastructure. Those can be added as independently validated follow-up modules.

## Limitations and responsible use

- **Not legal advice.** Verify every citation, clause, deadline, obligation, and number against the source document.
- Transformer summaries may hallucinate or omit important details.
- Very long documents are capped at 12 overlapping chunks in transformer mode; not all text may be covered.
- T5-small is a general pretrained model and may produce weak summaries without task-specific fine-tuning.
- Use synthetic or public-domain documents for demonstrations; do not commit client or sensitive records.

## Author

Nivetha Murugesan — AI, cloud engineering, and applied NLP research.
