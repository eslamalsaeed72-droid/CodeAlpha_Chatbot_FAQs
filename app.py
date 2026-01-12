# app.py – English FAQ Chatbot (Streamlit + Sentence-BERT)

import streamlit as st
import pandas as pd
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from nltk.corpus import stopwords
import re
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


# ==============================
# NLTK setup (safe for local/Cloud)
# ==============================

def setup_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


setup_nltk()
STOPWORDS_EN = set(stopwords.words("english"))


# ==============================
# English Text Analyzer
# ==============================

class EnglishTextAnalyzer:
    """Analyze English queries: intent, sentiment, keywords."""

    def __init__(self):
        self.stopwords_en = STOPWORDS_EN

    def analyze_intent(self, text: str):
        t = text.lower()

        patterns = {
            "track_order": ["track", "tracking", "where is my order", "order status"],
            "return_policy": ["return", "refund", "exchange", "money back"],
            "shipping": ["shipping", "delivery", "ship", "arrive"],
            "account": ["account", "login", "password", "sign in"],
            "payment": ["pay", "payment", "card", "credit", "billing"],
            "complaint": ["angry", "upset", "bad", "problem", "issue", "late"],
        }

        best_intent = "general"
        best_score = 0.0
        for intent, kws in patterns.items():
            score = sum(1 for k in kws if k in t)
            if score > best_score:
                best_score = score
                best_intent = intent

        confidence = 0.9 if best_score >= 2 else 0.6 if best_score == 1 else 0.4
        return {"type": best_intent, "confidence": confidence}

    def analyze_sentiment(self, text: str):
        t = text.lower()
        neg = ["angry", "upset", "bad", "terrible", "hate", "problem", "issue", "late"]
        pos = ["great", "good", "amazing", "thanks", "thank you", "love"]

        neg_score = sum(1 for k in neg if k in t)
        pos_score = sum(1 for k in pos if k in t)

        if neg_score > pos_score:
            return {"type": "negative", "polarity": -0.6}
        if pos_score > neg_score:
            return {"type": "positive", "polarity": 0.6}
        return {"type": "neutral", "polarity": 0.0}

    def extract_keywords(self, text: str, max_keywords: int = 8):
        clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        tokens = clean.split()
        tokens = [t for t in tokens if t not in self.stopwords_en and len(t) > 2]

        freq = defaultdict(int)
        for t in tokens:
            freq[t] += 1

        sorted_kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_kw[:max_keywords]]

    def analyze(self, text: str):
        return {
            "intent": self.analyze_intent(text),
            "sentiment": self.analyze_sentiment(text),
            "keywords": self.extract_keywords(text),
        }


# ==============================
# Response Generator
# ==============================

class EnglishResponseGenerator:
    """Generate responses with simple tone control."""

    def __init__(self, analyzer: EnglishTextAnalyzer):
        self.analyzer = analyzer

    def _build_prefix(self, analysis):
        intent = analysis["intent"]["type"]
        sentiment = analysis["sentiment"]["type"]

        if intent == "complaint" or sentiment == "negative":
            return "We are really sorry for the inconvenience. "
        if intent == "track_order":
            return "Here is how you can check your order status: "
        if intent == "return_policy":
            return "Here is an overview of our return policy: "
        return "Here is the information you requested: "

    def determine_tone(self, analysis):
        sentiment = analysis["sentiment"]["type"]
        intent = analysis["intent"]["type"]

        if intent == "complaint" or sentiment == "negative":
            return "empathetic"
        if intent in ["track_order", "return_policy", "shipping"]:
            return "informative"
        return "neutral"

    def generate(self, base_answer: str, analysis):
        prefix = self._build_prefix(analysis)
        full = prefix + base_answer
        tone = self.determine_tone(analysis)
        return full, tone


# ==============================
# Load English FAQ data
# ==============================

@st.cache_data(show_spinner=True)
def load_english_faqs(max_rows: int = 20000):
    """Load real English FAQ subset or fallback."""
    try:
        ds = load_dataset(
            "PaDaS-Lab/webfaq-retrieval", "eng", split=f"train[:{max_rows}]"
        )  # [web:8]
        faqs = []

        for row in ds:
            query = str(row.get("query", "")).strip()
            pos = row.get("positive_passages", [])
            if not query or not pos:
                continue
            passage = pos[0]
            if isinstance(passage, dict):
                ans = str(passage.get("passage_text", "")).strip()
            else:
                ans = str(passage).strip()

            if len(query) < 8 or len(ans) < 20:
                continue

            faqs.append(
                {
                    "question_en": query,
                    "answer_en": ans,
                    "category": row.get("category", "General"),
                    "source": "WebFAQ-eng",
                }
            )

        if len(faqs) == 0:
            raise RuntimeError("Empty WebFAQ subset")

        return faqs
    except Exception:
        # Small fallback set for local testing
        return [
            {
                "question_en": "How can I track my order?",
                "answer_en": "Log in to your account, open the 'Your Orders' page, and click on the order to see detailed tracking information.",
                "category": "Shipping",
                "source": "fallback",
            },
            {
                "question_en": "What is your return policy?",
                "answer_en": "You can return most items within 30 days of delivery as long as they are in their original condition.",
                "category": "Returns",
                "source": "fallback",
            },
            {
                "question_en": "How long does shipping take?",
                "answer_en": "Standard shipping usually takes between 5 and 7 business days.",
                "category": "Shipping",
                "source": "fallback",
            },
            {
                "question_en": "How do I reset my password?",
                "answer_en": "Click on 'Forgot password' on the login page and follow the instructions sent to your email.",
                "category": "Account",
                "source": "fallback",
            },
            {
                "question_en": "What payment methods do you accept?",
                "answer_en": "We accept major credit cards, debit cards, and PayPal.",
                "category": "Payment",
                "source": "fallback",
            },
        ]


# ==============================
# BERT-based English FAQ matcher
# ==============================

class EnglishBERTFAQMatcher:
    """Semantic FAQ matcher using Sentence-BERT (English only)."""

    def __init__(
        self, faqs, analyzer: EnglishTextAnalyzer, generator: EnglishResponseGenerator
    ):
        self.faqs = faqs
        self.analyzer = analyzer
        self.generator = generator

        # all-MiniLM-L6-v2: fast English semantic model [web:52]
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._build_index()

    def _build_index(self):
        questions = [f["question_en"] for f in self.faqs]
        self.embeddings = self.model.encode(
            questions, convert_to_tensor=True, show_progress_bar=False
        )

    def _find_best(self, user_question: str):
        query_emb = self.model.encode(user_question, convert_to_tensor=True)
        sims = util.pytorch_cos_sim(query_emb, self.embeddings)[0]
        best_idx = int(torch.argmax(sims).item())
        best_score = float(sims[best_idx])
        confidence = max(0.0, min(best_score, 1.0))
        return best_idx, confidence

    def get_intelligent_answer(self, user_question: str):
        analysis = self.analyzer.analyze(user_question)
        best_idx, confidence = self._find_best(user_question)
        faq = self.faqs[best_idx]

        base_answer = faq.get("answer_en", "")
        enhanced_answer, tone = self.generator.generate(base_answer, analysis)

        return {
            "answer": enhanced_answer,
            "confidence": confidence,
            "category": faq.get("category", "General"),
            "analysis": analysis,
            "tone": tone,
            "matched_question": faq.get("question_en", ""),
            "source": faq.get("source", "unknown"),
        }


# ==============================
# Build app objects (once)
# ==============================

@st.cache_resource(show_spinner=True)
def build_system():
    faqs = load_english_faqs()
    analyzer = EnglishTextAnalyzer()
    generator = EnglishResponseGenerator(analyzer)
    matcher = EnglishBERTFAQMatcher(faqs, analyzer, generator)
    return matcher


matcher = build_system()


# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="English FAQ Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 English FAQ Chatbot")
st.markdown(
    "Semantic FAQ assistant powered by **Sentence-BERT (all-MiniLM-L6-v2)**."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Ask a question (English only):", placeholder="How can I track my order?"
    )
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    question = user_input.strip()
    result = matcher.get_intelligent_answer(question)
    st.session_state.chat_history.append(
        {
            "user": question,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "tone": result["tone"],
            "intent": result["analysis"]["intent"]["type"],
        }
    )

st.markdown("---")

for msg in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {msg['user']}")
    st.markdown(f"**Bot:** {msg['answer']}")
    cols = st.columns(3)
    cols[0].metric("Confidence", f"{msg['confidence']:.1%}")
    cols[1].metric("Tone", msg["tone"].title())
    cols[2].metric("Intent", msg["intent"])

st.markdown("---")
st.caption("English FAQ Chatbot · Sentence-BERT all-MiniLM-L6-v2")
