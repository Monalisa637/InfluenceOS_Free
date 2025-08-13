import io
import pandas as pd
import streamlit as st

from utils import fetch_and_clean, rule_based_summary, safe_truncate, build_outreach_prompt

# Try to import transformers models
USE_TRANSFORMERS = True
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
except Exception as e:
    USE_TRANSFORMERS = False

st.set_page_config(page_title="InfluenceOS (Free)", page_icon="🧠", layout="wide")

st.title("InfluenceOS — 100% Free, Local & Assignment-Ready 🧠")
st.caption("Scrape → Summarize → Analyze Sentiment → Draft Outreach. No paid APIs.")

with st.sidebar:
    st.header("Settings")
    no_llm = st.toggle("No-LLM Mode (rule-based only)", value=False)
    tone = st.selectbox("Outreach Tone", ["friendly", "professional", "casual"])
    max_len = st.slider("Summary Max Tokens (LLM)", 32, 256, 128, step=8)
    st.markdown("---")
    st.write("First run downloads small models (free). If issues, toggle No-LLM Mode.")

tab1, tab2 = st.tabs(["Analyze Content", "Generate Outreach"])

@st.cache_resource(show_spinner=False)
def get_summarizer():
    # Use FLAN-T5 small for speed
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    model_name = "google/flan-t5-small"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=mdl, tokenizer=tok)

@st.cache_resource(show_spinner=False)
def get_sentiment():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=mdl, tokenizer=tok)

with tab1:
    colL, colR = st.columns(2)
    with colL:
        st.subheader("1) Input")
        mode = st.radio("Choose input type", ["URL", "Raw Text"], horizontal=True)
        raw_text = ""
        if mode == "URL":
            url = st.text_input("Paste a URL to analyze", placeholder="https://example.com/blog")
            if st.button("Fetch & Extract"):
                if not url.strip():
                    st.warning("Please paste a URL.")
                else:
                    with st.spinner("Fetching & extracting..."):
                        raw_text = fetch_and_clean(url)
                        st.session_state["raw_text"] = raw_text
            raw_text = st.session_state.get("raw_text", "")
        else:
            raw_text = st.text_area("Paste text to analyze", height=200, placeholder="Paste article, post, or notes here...")
    with colR:
        st.subheader("2) Summary & Sentiment")
        if st.button("Run Analysis"):
            if not raw_text.strip():
                st.warning("Please provide text (via URL or Raw Text).")
            else:
                if no_llm or not USE_TRANSFORMERS:
                    summary = rule_based_summary(raw_text, max_sentences=3)
                    sent = "N/A (No-LLM mode)"
                else:
                    try:
                        summarizer = get_summarizer()
                        prompt = f"Summarize in 5 short bullet points:\n{raw_text[:3000]}"
                        out = summarizer(prompt, max_new_tokens=max_len, num_return_sequences=1)
                        summary = out[0]["generated_text"]
                        sentiment = get_sentiment()
                        sent = sentiment(raw_text[:2000])[0]
                    except Exception as e:
                        st.error(f"Model error: {e}")
                        summary = rule_based_summary(raw_text, max_sentences=3)
                        sent = "N/A (fallback)"
                st.session_state["summary"] = summary
                st.session_state["sentiment"] = sent
        if "summary" in st.session_state:
            st.markdown("**Summary**")
            st.write(st.session_state["summary"])
            st.markdown("**Sentiment**")
            st.write(st.session_state["sentiment"])

with tab2:
    st.subheader("3) Outreach Draft")
    topic = st.text_input("Topic or product")
    insights = st.text_area("Key insights (paste your summary or key points)",
                            value=st.session_state.get("summary", ""),
                            height=150)
    if st.button("Generate Outreach"):
        if no_llm or not USE_TRANSFORMERS:
            # Simple template in No-LLM mode
            msg = (
                f"Hi there,\n\nI came across your work related to {topic}. "
                f"Here are a few points that stood out:\n{insights}\n\n"
                "Would you be open to a quick chat to explore a small collaboration? "
                "Happy to share more details.\n\nThanks!"
            )
        else:
            try:
                summarizer = get_summarizer()
                prompt = build_outreach_prompt(topic, insights, tone=tone)
                out = summarizer(prompt, max_new_tokens=120, num_return_sequences=1)
                msg = out[0]["generated_text"]
            except Exception as e:
                st.error(f"Model error: {e}")
                msg = (
                    f"(Fallback) Hi! Loved your work on {topic}. "
                    "Based on the insights above, I'd love to connect and explore a quick collaboration. "
                    "If you're open, could we schedule a short call this week? Thanks!"
                )
        st.session_state["outreach"] = msg
    if "outreach" in st.session_state:
        st.markdown("**Outreach Message**")
        st.code(st.session_state["outreach"], language="markdown")
        # Export section
        st.markdown("---")
        st.subheader("Export")
        data = {
            "topic": [topic],
            "summary": [st.session_state.get("summary", "")],
            "sentiment": [st.session_state.get("sentiment", "")],
            "outreach": [st.session_state.get("outreach", "")],
        }
        df = pd.DataFrame(data)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="influenceos_export.csv", mime="text/csv")
