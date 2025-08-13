# InfluenceOS (100% Free, Assignment-Ready)

A lightweight **Streamlit** app that performs:
- Webpage/article scraping (URL or raw text)
- Clean text extraction
- Summarization (FLAN-T5 via Hugging Face, free)
- Sentiment analysis (DistilBERT SST-2, free)
- Outreach message draft generation
- CSV export of results

> Everything runs locally and uses only free/open-source packages.

## Quick Start

```bash
# 1) Create & activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run
streamlit run app.py
```

Then open the local URL that Streamlit prints (usually http://localhost:8501).

## Notes
- First run will download small open-source models from Hugging Face (free).
- If you are offline, toggle **No-LLM Mode** to use simple rule-based summaries.
- You can submit this folder as-is (zip).

## Why Streamlit?
- Zero-cost to run locally.
- Fast to build.
- Clean UI for demos/assignments.

## Optional: Export
Click **Download CSV** to export your analysis.
