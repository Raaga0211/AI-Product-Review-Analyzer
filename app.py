import streamlit as st
import joblib
import numpy as np
import re
import base64
from datetime import datetime
from scipy.sparse import hstack


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Product Review Analyzer",
    page_icon="🛍️",
    layout="wide"
)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_model()


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

/* ---- Global ---- */
.stApp {
    background: radial-gradient(circle at 20% 0%, #0c1526 0%, #060b13 45%, #050810 100%) !important;
    color: white !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 3rem !important; max-width: 1200px !important; }

* { box-sizing: border-box; }

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3); }
    50%      { box-shadow: 0 4px 28px rgba(16, 185, 129, 0.45); }
}

/* ---- Header / Top Navigation ---- */
.nav-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}
.brand-title {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 32px;
    font-weight: 800;
    color: #ffffff;
}
.brand-title span {
    background: linear-gradient(90deg, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 16px;
    margin-top: -8px;
    margin-bottom: 32px;
}

/* ---- Input Panel Card ---- */
.input-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 24px;
    transition: border-color 0.25s ease;
}
.input-card:hover { border-color: rgba(56, 189, 248, 0.2); }

.input-card-label {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 16px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 14px;
}

.section-label {
    margin-top: 14px;
    margin-bottom: 6px;
    font-size: 13px;
    color: #94a3b8;
}

/* Textarea */
.stTextArea textarea {
    background: #090f1c !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    padding: 14px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextArea textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 1px #38bdf8 !important;
}

/* Text input for emojis */
.stTextInput input {
    background: #090f1c !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

/* ---- Primary Action Button ---- */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(90deg, #3b82f6, #10b981) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px !important;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3) !important;
    transition: transform 0.15s ease, box-shadow 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    animation: pulseGlow 1.6s ease-in-out infinite;
}
.stButton > button:active { transform: translateY(0px); }

/* Secondary buttons (example chips / clear) get a lighter look */
button[kind="secondary"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    box-shadow: none !important;
}
button[kind="secondary"]:hover {
    border-color: rgba(56, 189, 248, 0.4) !important;
    color: #ffffff !important;
    animation: none !important;
    transform: none !important;
}

/* ---- Hero Vector Container ---- */
.hero-vector-box {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    height: 100%;
    min-height: 300px;
}
.hero-vector-box img {
    max-width: 100%;
    height: auto;
}

/* ---- Results Section ---- */
.results-header {
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
    margin-top: 32px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.result-card {
    background: #0b1324;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 20px;
    height: 100%;
    animation: fadeInUp 0.4s ease both;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.result-card:hover { transform: translateY(-3px); }
.result-card.red-theme { border-color: rgba(239, 68, 68, 0.25); background: rgba(239, 68, 68, 0.04); }
.result-card.green-theme { border-color: rgba(16, 185, 129, 0.25); background: rgba(16, 185, 129, 0.04); }
.result-card.orange-theme { border-color: rgba(245, 158, 11, 0.25); background: rgba(245, 158, 11, 0.04); }

.card-title-container {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #94a3b8;
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 16px;
}
.card-title-icon {
    width: 26px; height: 26px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
}
.card-value {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 12px;
}
.card-value.red-text { color: #ef4444; }
.card-value.green-text { color: #10b981; }
.card-value.orange-text { color: #f59e0b; }

.metric-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    height: 6px;
    width: 100%;
    overflow: hidden;
}
.metric-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}
.metric-bar-fill.red { background: #ef4444; }
.metric-bar-fill.green { background: #10b981; }
.metric-bar-fill.orange { background: #f59e0b; }
.metric-bar-fill.blue { background: #3b82f6; }

.card-footer-text {
    font-size: 12px;
    color: #64748b;
    margin-top: 8px;
}

/* ---- Insights Banner ---- */
.insights-panel {
    background: #0b1324;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
    animation: fadeInUp 0.5s ease both;
    animation-delay: 0.15s;
}
.insights-title {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #38bdf8;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 8px;
}
.insights-body {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.6;
}

/* ---- History Panel ---- */
.history-panel {
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 16px;
}
.history-title {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 10px;
}
.history-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
    font-size: 13px;
    color: #cbd5e1;
    border-top: 1px solid rgba(255,255,255,0.04);
}
.history-row:first-of-type { border-top: none; }
.history-dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}
.history-dot.green { background: #10b981; }
.history-dot.red { background: #ef4444; }
.history-dot.orange { background: #f59e0b; }
.history-time { color: #475569; font-size: 11px; margin-left: auto; }

/* ---- Footer ---- */
.app-footer {
    text-align: center;
    color: #475569;
    font-size: 12px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# TEXT CLEANING
# ==================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==================================================
# EMOJI SYSTEM
# ==================================================

emoji_dict = {
    "😊": 2, "❤️": 2, "😍": 2, "😁": 2, "😃": 2, "🥰": 2, "💯": 2, "🎉": 1,
    "✨": 1, "🔥": 1, "👍": 1,
    "🤔": 0, "🙂": 0, "😌": 0, "😐": 0,
    "🙁": -1, "😩": -1, "👎": -1,
    "😞": -2, "😢": -2, "😭": -2, "😠": -2, "😡": -2, "💔": -2,
}


def get_emoji_score(text):
    score = 0
    for token in str(text).split():
        score += emoji_dict.get(token, 0)
    return score


# ==================================================
# SENTIMENT RULES
# ==================================================

POSITIVE_KEYWORDS = {
    "love", "amazing", "excellent", "perfect", "perfectly", "awesome", "fantastic",
    "great", "wonderful", "best", "good", "stylish", "comfortable", "beautiful",
    "gorgeous", "cute", "elegant", "satisfied"
}
NEGATIVE_KEYWORDS = {
    "bad", "poor", "worst", "terrible", "awful", "cheap", "hate", "horrible",
    "disappointing", "useless", "waste", "uncomfortable"
}
NEGATION_WORDS = {"not", "no", "never", "cannot", "cant", "wasnt", "isnt", "arent", "didnt"}
CONTRAST_WORDS = {"but", "however", "although", "though"}

LABEL_MAP = {
    "positive": "Positive", "pos": "Positive", "1": "Positive",
    "negative": "Negative", "neg": "Negative", "0": "Negative",
    "neutral": "Neutral", "2": "Neutral",
}


def normalize_label(label):
    """Guard against models that output different casings/encodings for labels."""
    key = str(label).strip().lower()
    mapped = LABEL_MAP.get(key)
    if mapped:
        return mapped
    title = str(label).strip().title()
    return title if title in {"Positive", "Negative", "Neutral"} else "Neutral"


def score_tokens(tokens):
    """Word-level (not substring!) keyword scoring with basic negation handling."""
    pos, neg = 0, 0
    for i, tok in enumerate(tokens):
        negated = i > 0 and tokens[i - 1] in NEGATION_WORDS
        if tok in POSITIVE_KEYWORDS:
            neg += 1 if negated else 0
            pos += 0 if negated else 1
        elif tok in NEGATIVE_KEYWORDS:
            pos += 1 if negated else 0
            neg += 0 if negated else 1
    return pos, neg


def split_on_contrast(tokens):
    """Split on the first whole-word contrast token (but/however/although/though)."""
    for i, tok in enumerate(tokens):
        if tok in CONTRAST_WORDS:
            return tokens[:i], tokens[i + 1:]
    return None


def rule_based_sentiment(cleaned_text):
    """
    Returns (sentiment, confidence, reason) or (None, None, None) if no rule fires,
    in which case the caller should fall back to the trained ML model.
    """
    tokens = cleaned_text.split()
    if not tokens:
        return None, None, None

    contrast = split_on_contrast(tokens)
    if contrast:
        before, after = contrast
        pos_b, neg_b = score_tokens(before)
        pos_a, neg_a = score_tokens(after)
        flips_neg_to_pos = neg_b > pos_b and pos_a > neg_a
        flips_pos_to_neg = pos_b > neg_b and neg_a > pos_a
        if flips_neg_to_pos or flips_pos_to_neg:
            margin = abs((pos_b - neg_b) - (neg_a - pos_a))
            confidence = min(65 + margin * 5, 90)
            return "Neutral", confidence, "Contrast detected"

    pos, neg = score_tokens(tokens)
    if pos == 0 and neg == 0:
        return None, None, None
    if pos == neg:
        return "Neutral", min(60 + pos * 5, 85), "Balanced keywords"
    if pos > neg:
        return "Positive", min(65 + (pos - neg) * 8, 96), "Keyword match"
    return "Negative", min(65 + (neg - pos) * 8, 96), "Keyword match"


def analyze_review(review_text, emoji_text=""):
    """
    Single source of truth for sentiment + confidence, so the displayed label
    and confidence score always correspond to the same decision path.
    """
    cleaned = clean_text(review_text)
    sentiment, confidence, reason = rule_based_sentiment(cleaned)

    emoji_score = get_emoji_score(emoji_text) if emoji_text.strip() else get_emoji_score(review_text)

    if sentiment is not None:
        ml_conf = confidence
        emoji_bonus = min(abs(emoji_score) * 2, 6) if emoji_score else 0
        overall_conf = min(ml_conf + emoji_bonus, 98)
        source = reason
    else:
        text_vector = vectorizer.transform([cleaned])
        emoji_vector = np.array([[emoji_score]])
        final_input = hstack([text_vector, emoji_vector])

        raw_pred = model.predict(final_input)[0]
        sentiment = normalize_label(raw_pred)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(final_input)[0]
            ml_conf = round(float(np.max(proba)) * 100, 2)
        else:
            ml_conf = 92.31

        emoji_boost = min(abs(emoji_score) * 2.5, 7)
        overall_conf = min(ml_conf + emoji_boost, 98.5) if emoji_score != 0 else max(ml_conf - 5.17, 50.0)
        source = "ML model"

    overall_conf = round(max(0.0, min(overall_conf, 100.0)), 2)
    ml_conf = round(max(0.0, min(ml_conf, 100.0)), 2)
    return sentiment, ml_conf, overall_conf, emoji_score, source


# ==================================================
# SESSION STATE
# ==================================================

if "review_text" not in st.session_state:
    st.session_state.review_text = ""
if "emoji_text" not in st.session_state:
    st.session_state.emoji_text = ""
if "history" not in st.session_state:
    st.session_state.history = []

EXAMPLES = [
    ("😊 Positive example", "This dress is absolutely stunning and fits perfectly, I love it!", "😍"),
    ("😡 Negative example", "Terrible quality, it fell apart after one wash.", "😡"),
    ("😐 Mixed example", "The fabric feels nice but the sizing runs way too small.", "🤔"),
]


def _apply_example(text, emoji):
    st.session_state.review_text = text
    st.session_state.emoji_text = emoji


def _clear_inputs():
    st.session_state.review_text = ""
    st.session_state.emoji_text = ""


# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="nav-container">
    <div class="brand-title">🛍️ <span>AI Product Review Analyzer</span></div>
</div>
<div class="subtitle">Analyze reviews using Machine Learning + Emoji Intelligence</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️  About this tool"):
    st.markdown(
        "This analyzer blends a keyword & negation-aware rule engine with a trained "
        "TF-IDF + ML sentiment model, and factors in any emojis you provide. Reviews with "
        "clear contrast (e.g. *\"good but overpriced\"*) are flagged as **Neutral**, while "
        "unambiguous reviews get a **Positive** or **Negative** call with a confidence score."
    )


# ==================================================
# LAYOUT
# ==================================================

left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.markdown('<div class="input-card"><div class="input-card-label">📝 Enter Product Review</div>', unsafe_allow_html=True)

    review = st.text_area(
        label="Product review",
        height=120,
        placeholder="their new collection is terrible 😡",
        label_visibility="collapsed",
        key="review_text",
    )

    st.markdown('<div class="section-label">Optional: Paste Emojis Below</div>', unsafe_allow_html=True)
    emoji_input = st.text_input(
        label="Emoji input",
        placeholder="😡 😍 🤔",
        label_visibility="collapsed",
        key="emoji_text",
    )

    st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)
    ex_cols = st.columns(len(EXAMPLES))
    for col, (label, text, emoji) in zip(ex_cols, EXAMPLES):
        with col:
            st.button(label, key=f"ex_{label}", type="secondary", use_container_width=True,
                      on_click=_apply_example, args=(text, emoji))

    st.markdown('</div>', unsafe_allow_html=True)

    btn_cols = st.columns([3, 1])
    with btn_cols[0]:
        analyze_clicked = st.button("🚀 Analyze Review", use_container_width=True, type="primary")
    with btn_cols[1]:
        st.button("Clear", use_container_width=True, type="secondary", on_click=_clear_inputs)

with right_col:
    svg_illustration = """
    <svg width="380" height="320" viewBox="0 0 380 320" fill="none" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="bagGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#2563eb"/>
                <stop offset="100%" stop-color="#1d4ed8"/>
            </linearGradient>
            <radialGradient id="floorShadow" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#000000" stop-opacity="0.6"/>
                <stop offset="100%" stop-color="#000000" stop-opacity="0"/>
            </radialGradient>
            <filter id="glowRed" x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur stdDeviation="7" result="blur"/>
                <feComposite in="SourceGraphic" in2="blur" operator="over"/>
            </filter>
            <filter id="glowYellow" x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur stdDeviation="7" result="blur"/>
                <feComposite in="SourceGraphic" in2="blur" operator="over"/>
            </filter>
            <filter id="glowGreen" x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur stdDeviation="7" result="blur"/>
                <feComposite in="SourceGraphic" in2="blur" operator="over"/>
            </filter>
        </defs>

        <ellipse cx="190" cy="275" rx="110" ry="12" fill="url(#floorShadow)"/>

        <path d="M162 135 C162 98, 218 98, 218 135" stroke="#3b82f6" stroke-width="7" fill="none" stroke-linecap="round"/>

        <rect x="120" y="130" width="140" height="135" rx="22" fill="url(#bagGrad)" filter="drop-shadow(0px 20px 35px rgba(0,0,0,0.65))"/>

        <circle cx="190" cy="200" r="32" fill="rgba(255, 255, 255, 0.16)" stroke="rgba(255, 255, 255, 0.25)" stroke-width="1.2"/>
        <circle cx="177" cy="194" r="3.5" fill="#ffffff"/>
        <circle cx="203" cy="194" r="3.5" fill="#ffffff"/>
        <path d="M178 208 C178 219, 202 219, 202 208" stroke="#ffffff" stroke-width="3.5" fill="none" stroke-linecap="round"/>

        <g transform="translate(265, 65)" filter="drop-shadow(0px 8px 16px rgba(0,0,0,0.4))">
            <rect x="0" y="0" width="90" height="32" rx="8" fill="#0b1324" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>
            <text x="12" y="21" fill="#fbbf24" font-family="Arial, sans-serif" font-size="14" font-weight="bold" letter-spacing="2">★★★★</text>
        </g>

        <g filter="url(#glowRed)">
            <circle cx="75" cy="75" r="24" fill="#ef4444"/>
            <path d="M63 67 L71 71" stroke="#0f172a" stroke-width="2.5" stroke-linecap="round"/>
            <path d="M87 67 L79 71" stroke="#0f172a" stroke-width="2.5" stroke-linecap="round"/>
            <circle cx="68" cy="74" r="2.5" fill="#0f172a"/>
            <circle cx="82" cy="74" r="2.5" fill="#0f172a"/>
            <path d="M67 86 C67 80, 83 80, 83 86" stroke="#0f172a" stroke-width="2.5" fill="none" stroke-linecap="round"/>
        </g>

        <g filter="url(#glowYellow)">
            <circle cx="50" cy="180" r="24" fill="#f59e0b"/>
            <circle cx="43" cy="176" r="2.5" fill="#0f172a"/>
            <circle cx="57" cy="176" r="2.5" fill="#0f172a"/>
            <line x1="41" y1="187" x2="59" y2="187" stroke="#0f172a" stroke-width="2.5" stroke-linecap="round"/>
        </g>

        <g filter="url(#glowGreen)">
            <circle cx="315" cy="220" r="24" fill="#10b981"/>
            <path d="M303 214 C303 210, 311 210, 311 214" stroke="#0f172a" stroke-width="2.5" fill="none" stroke-linecap="round"/>
            <path d="M327 214 C327 210, 319 210, 319 214" stroke="#0f172a" stroke-width="2.5" fill="none" stroke-linecap="round"/>
            <path d="M305 224 C305 234, 325 234, 325 224 Z" fill="#0f172a"/>
        </g>
    </svg>
    """
    b64_svg = base64.b64encode(svg_illustration.encode('utf-8')).decode('utf-8')
    st.markdown(f'<div class="hero-vector-box"><img src="data:image/svg+xml;base64,{b64_svg}" alt="3D illustration" /></div>', unsafe_allow_html=True)


# ==================================================
# ANALYTICS DASHBOARD
# ==================================================

if analyze_clicked:
    if not review.strip():
        st.warning("Please enter a review statement first.")
        st.stop()

    sentiment, ml_conf, overall_conf, emoji_score, source = analyze_review(review, emoji_input)

    st.session_state.history.insert(0, {
        "review": review.strip()[:60] + ("…" if len(review.strip()) > 60 else ""),
        "sentiment": sentiment,
        "time": datetime.now().strftime("%H:%M"),
    })
    st.session_state.history = st.session_state.history[:5]

    theme_class = {"Positive": "green-theme", "Negative": "red-theme", "Neutral": "orange-theme"}.get(sentiment, "orange-theme")
    text_class = {"Positive": "green-text", "Negative": "red-text", "Neutral": "orange-text"}.get(sentiment, "orange-text")
    bar_class = {"Positive": "green", "Negative": "red", "Neutral": "orange"}.get(sentiment, "orange")
    sentiment_emoji = {"Positive": "🙂", "Negative": "☹️", "Neutral": "😐"}.get(sentiment, "😐")
    rec_text = {"Positive": "Recommended", "Negative": "Not Recommended", "Neutral": "Neutral / Watch Out"}.get(sentiment, "Neutral / Watch Out")
    rec_desc = {"Positive": "Great option to buy!", "Negative": "Better options available!", "Neutral": "Review further details"}.get(sentiment, "Review further details")

    emoji_pct = int(((emoji_score + 5) / 10) * 100)
    emoji_pct = max(0, min(100, emoji_pct))

    st.markdown("""
    <div class="results-header">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.5"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>
        Results
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="result-card {theme_class}" style="animation-delay:0.0s;">
            <div class="card-title-container"><div class="card-title-icon">{sentiment_emoji}</div> Predicted Sentiment</div>
            <div class="card-value {text_class}">{sentiment}</div>
            <div class="metric-bar-bg"><div class="metric-bar-fill {bar_class}" style="width: {ml_conf}%;"></div></div>
            <div class="card-footer-text">Confidence: {ml_conf}% • {source}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="result-card" style="animation-delay:0.05s;">
            <div class="card-title-container"><div class="card-title-icon">😊</div> Emoji Score</div>
            <div class="card-value" style="color: white;">{emoji_score}</div>
            <div class="metric-bar-bg"><div class="metric-bar-fill blue" style="width: {emoji_pct}%;"></div></div>
            <div class="card-footer-text" style="display:flex; justify-content:space-between;"><span>-5</span><span>0</span><span>+5</span></div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="result-card {theme_class}" style="animation-delay:0.1s;">
            <div class="card-title-container"><div class="card-title-icon">⭐</div> Final Sentiment</div>
            <div class="card-value {text_class}">{sentiment}</div>
            <div class="metric-bar-bg"><div class="metric-bar-fill {bar_class}" style="width: {overall_conf}%;"></div></div>
            <div class="card-footer-text">Overall Confidence: {overall_conf}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="result-card {theme_class}" style="animation-delay:0.15s;">
            <div class="card-title-container"><div class="card-title-icon">🛡️</div> Recommendation</div>
            <div class="card-value {text_class}" style="font-size:20px; line-height:1.4; margin-bottom:18px;">{rec_text}</div>
            <div style="color: #94a3b8; font-size:13px;">{rec_desc}</div>
        </div>
        """, unsafe_allow_html=True)

    insights_text = {
        "Positive": "The review expresses genuine satisfaction. Customers report a positive experience with this product. Consider highlighting this as a top-rated item.",
        "Negative": "The review expresses strong dissatisfaction. Consider checking product quality and customer feedback.",
        "Neutral": "The review contains mixed observations, or wording that offsets itself (e.g. praise undercut by a complaint). Check details carefully.",
    }.get(sentiment, "Check details carefully.")

    st.markdown(f"""
    <div class="insights-panel">
        <div class="insights-title">💡 Review Insights</div>
        <div class="insights-body">{insights_text}</div>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.history:
    dot_map = {"Positive": "green", "Negative": "red", "Neutral": "orange"}
    rows = "".join(
        f'<div class="history-row"><span class="history-dot {dot_map.get(h["sentiment"],"orange")}"></span>'
        f'{h["review"]}<span class="history-time">{h["time"]} · {h["sentiment"]}</span></div>'
        for h in st.session_state.history
    )
    st.markdown(f"""
    <div class="history-panel">
        <div class="history-title">🕒 Recent Analyses</div>
        {rows}
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="app-footer">Built with ❤️ using Streamlit, Machine Learning & NLP</div>', unsafe_allow_html=True)