import streamlit as st
import pickle
import numpy as np
import re
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
import tensorflow as tf

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineScope · Sentiment AI",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #070711 !important;
    color: #e8e4dc !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, #1a0a2e88, transparent),
        radial-gradient(ellipse 60% 40% at 80% 80%, #0a1a2e55, transparent),
        #070711 !important;
}

[data-testid="stHeader"] { background: transparent !important; }
section[data-testid="stSidebar"] { display: none; }
.block-container { max-width: 760px !important; padding: 2rem 1.5rem 4rem !important; }
footer { display: none !important; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: #c9a96e;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: clamp(2.4rem, 6vw, 3.8rem);
    font-weight: 900;
    line-height: 1.1;
    background: linear-gradient(135deg, #f5e6c8 0%, #c9a96e 40%, #e8d5b0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.75rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #6b6b80;
    font-weight: 300;
    letter-spacing: 0.03em;
}
.hero-line {
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #c9a96e, transparent);
    margin: 1.5rem auto;
}

/* ── Text area ── */
.stTextArea textarea {
    background: #0e0e1c !important;
    border: 1px solid #1e1e35 !important;
    border-radius: 12px !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    padding: 16px !important;
    resize: vertical !important;
    transition: border-color 0.3s ease !important;
}
.stTextArea textarea:focus {
    border-color: #c9a96e !important;
    box-shadow: 0 0 0 3px #c9a96e18 !important;
    outline: none !important;
}
.stTextArea textarea::placeholder { color: #3a3a55 !important; }
.stTextArea label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #4a4a65 !important;
}

/* ── Button ── */
.stButton button {
    width: 100% !important;
    background: linear-gradient(135deg, #c9a96e, #a07840) !important;
    color: #070711 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.75rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
.stButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px #c9a96e33 !important;
}
.stButton button:active { transform: translateY(0) !important; }

/* ── Result cards ── */
.result-card {
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
    animation: slideUp 0.5s ease;
}
.result-card.positive {
    background: linear-gradient(135deg, #0a1f0f, #0d2015);
    border: 1px solid #1a4025;
}
.result-card.negative {
    background: linear-gradient(135deg, #1f0a0a, #200d0d);
    border: 1px solid #401a1a;
}
.result-card.uncertain {
    background: linear-gradient(135deg, #111120, #13131f);
    border: 1px solid #252535;
}
.result-glow {
    position: absolute;
    top: -40px; right: -40px;
    width: 120px; height: 120px;
    border-radius: 50%;
    filter: blur(40px);
    opacity: 0.3;
}
.positive .result-glow { background: #22c55e; }
.negative .result-glow { background: #ef4444; }
.uncertain .result-glow { background: #a78bfa; }

.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    opacity: 0.6;
}
.result-sentiment {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 1.2rem;
}
.positive .result-sentiment { color: #4ade80; }
.negative .result-sentiment { color: #f87171; }
.uncertain .result-sentiment { color: #c4b5fd; }

/* ── Confidence bar ── */
.conf-wrap { margin-bottom: 1.2rem; }
.conf-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: #4a4a65;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    display: flex;
    justify-content: space-between;
}
.conf-track {
    height: 5px;
    background: #ffffff0d;
    border-radius: 999px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 1s ease;
}
.positive .conf-fill { background: linear-gradient(90deg, #16a34a, #4ade80); }
.negative .conf-fill { background: linear-gradient(90deg, #b91c1c, #f87171); }
.uncertain .conf-fill { background: linear-gradient(90deg, #7c3aed, #c4b5fd); }

/* ── Word importance ── */
.words-section { margin-top: 1.2rem; }
.words-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a4a65;
    margin-bottom: 0.8rem;
}
.word-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.word-chip {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    min-width: 100px;
    color: #c9c9d8;
}
.word-bar-track {
    flex: 1;
    height: 4px;
    background: #ffffff08;
    border-radius: 999px;
    overflow: hidden;
}
.word-bar-fill-pos { height: 100%; background: #4ade8088; border-radius: 999px; }
.word-bar-fill-neg { height: 100%; background: #f8717188; border-radius: 999px; }
.word-score {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #4a4a65;
    min-width: 40px;
    text-align: right;
}

/* ── Example chips ── */
.example-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 0.5rem 0 1.2rem;
}
.example-chip {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    background: #0e0e1c;
    border: 1px solid #1e1e35;
    color: #5a5a75;
    padding: 5px 12px;
    border-radius: 20px;
    cursor: pointer;
    transition: all 0.2s;
}
.example-chip:hover {
    border-color: #c9a96e55;
    color: #c9a96e;
}

/* ── History ── */
.history-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-bottom: 1px solid #0e0e1c;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    color: #4a4a65;
}
.history-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    padding: 3px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    flex-shrink: 0;
}
.badge-pos { background: #4ade8022; color: #4ade80; border: 1px solid #4ade8033; }
.badge-neg { background: #f8717122; color: #f87171; border: 1px solid #f8717133; }
.badge-unc { background: #c4b5fd22; color: #c4b5fd; border: 1px solid #c4b5fd33; }

/* ── Stats ── */
.stats-row {
    display: flex;
    gap: 12px;
    margin: 1.5rem 0;
}
.stat-box {
    flex: 1;
    background: #0e0e1c;
    border: 1px solid #1a1a2e;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}
.stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #c9a96e;
}
.stat-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3a3a55;
    margin-top: 3px;
}

/* ── Divider ── */
.fancy-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 2rem 0;
}
.fancy-divider-line { flex: 1; height: 1px; background: #1a1a2e; }
.fancy-divider-dot {
    width: 4px; height: 4px;
    border-radius: 50%;
    background: #c9a96e44;
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# ── Attention Layer (needed to load model) ───────────────────────────────────
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True,
            name='attention_weight'
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        score = tf.nn.tanh(tf.matmul(x, self.W))
        weights = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(x * weights, axis=1)


# ── Load model & tokenizer ───────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = load_model('sentiment_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


# ── Helpers ──────────────────────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()


def predict(model, tokenizer, review):
    cleaned = clean_text(review)
    seq = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=300)
    score = float(model.predict(seq, verbose=0)[0][0])
    if score > 0.65:
        label = 'Positive'
    elif score < 0.35:
        label = 'Negative'
    else:
        label = 'Uncertain'
    return label, score, cleaned


def word_importance(model, tokenizer, cleaned, base_score, top_n=6):
    words = cleaned.split()
    if len(words) < 2:
        return []
    scores = []
    for i in range(len(words)):
        masked = words.copy()
        masked[i] = ''
        seq = pad_sequences(tokenizer.texts_to_sequences([' '.join(masked)]), maxlen=300)
        s = float(model.predict(seq, verbose=0)[0][0])
        scores.append((words[i], base_score - s))
    scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return scores[:top_n]


# ── Session state ─────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'review_text' not in st.session_state:
    st.session_state.review_text = ''


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">🎬 &nbsp; Deep Learning · NLP · BiLSTM + Attention</div>
    <div class="hero-title">CineScope</div>
    <div class="hero-sub">Sentiment intelligence for movie reviews</div>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
try:
    model, tokenizer = load_resources()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model: {e}")


# ── Stats row ─────────────────────────────────────────────────────────────────
total = len(st.session_state.history)
pos   = sum(1 for h in st.session_state.history if h['label'] == 'Positive')
neg   = sum(1 for h in st.session_state.history if h['label'] == 'Negative')

st.markdown(f"""
<div class="stats-row">
    <div class="stat-box">
        <div class="stat-val">{total}</div>
        <div class="stat-lbl">Analyzed</div>
    </div>
    <div class="stat-box">
        <div class="stat-val" style="color:#4ade80">{pos}</div>
        <div class="stat-lbl">Positive</div>
    </div>
    <div class="stat-box">
        <div class="stat-val" style="color:#f87171">{neg}</div>
        <div class="stat-lbl">Negative</div>
    </div>
    <div class="stat-box">
        <div class="stat-val">~87%</div>
        <div class="stat-lbl">Model Acc.</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Example chips ─────────────────────────────────────────────────────────────
st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:#3a3a55;margin-bottom:0.5rem;">Try an example</div>', unsafe_allow_html=True)

examples = [
    "Absolutely breathtaking. A masterpiece of cinema.",
    "Terrible film. Complete waste of time.",
    "It was okay, nothing special.",
    "The acting was superb but the plot was thin.",
]

cols = st.columns(len(examples))
for i, (col, ex) in enumerate(zip(cols, examples)):
    with col:
        if st.button(f"#{i+1}", key=f"ex_{i}", help=ex):
            st.session_state.review_text = ex
            st.rerun()


# ── Input ─────────────────────────────────────────────────────────────────────
review = st.text_area(
    "Your Review",
    value=st.session_state.review_text,
    placeholder="Write or paste a movie review here…",
    height=130,
    key="review_input",
    label_visibility="visible",
)

analyze = st.button("Analyze Sentiment →", use_container_width=True)


# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze and review.strip() and model_loaded:
    with st.spinner(""):
        time.sleep(0.3)
        label, score, cleaned = predict(model, tokenizer, review)
        importance = word_importance(model, tokenizer, cleaned, score)

    css_class = label.lower()
    conf_pct  = score * 100 if label == 'Positive' else (1 - score) * 100
    emoji     = "😊" if label == "Positive" else ("😞" if label == "Negative" else "🤔")

    # Build word importance HTML
    words_html = ""
    if importance:
        words_html = '<div class="words-section"><div class="words-title">Word influence</div>'
        for word, imp in importance:
            pct  = min(abs(imp) * 200, 100)
            cls  = "word-bar-fill-pos" if imp > 0 else "word-bar-fill-neg"
            sign = "▲" if imp > 0 else "▼"
            words_html += f"""
            <div class="word-row">
                <span class="word-chip">{word}</span>
                <div class="word-bar-track"><div class="{cls}" style="width:{pct}%"></div></div>
                <span class="word-score">{sign} {abs(imp):.3f}</span>
            </div>"""
        words_html += "</div>"

    st.markdown(f"""
    <div class="result-card {css_class}">
        <div class="result-glow"></div>
        <div class="result-label">Prediction</div>
        <div class="result-sentiment">{emoji} &nbsp;{label}</div>
        <div class="conf-wrap">
            <div class="conf-label">
                <span>Confidence</span>
                <span>{conf_pct:.1f}%</span>
            </div>
            <div class="conf-track">
                <div class="conf-fill" style="width:{conf_pct}%"></div>
            </div>
        </div>
        {words_html}
    </div>
    """, unsafe_allow_html=True)

    # Save to history
    st.session_state.history.insert(0, {
        'text':  review[:70] + ('…' if len(review) > 70 else ''),
        'label': label,
        'conf':  conf_pct,
    })
    st.session_state.history = st.session_state.history[:10]

elif analyze and not review.strip():
    st.warning("Please enter a review first.")


# ── Divider ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fancy-divider">
    <div class="fancy-divider-line"></div>
    <div class="fancy-divider-dot"></div>
    <div class="fancy-divider-line"></div>
</div>
""", unsafe_allow_html=True)


# ── History ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:#3a3a55;margin-bottom:0.5rem;">Recent analyses</div>', unsafe_allow_html=True)

    history_html = ""
    for h in st.session_state.history:
        badge_cls = "badge-pos" if h['label'] == "Positive" else ("badge-neg" if h['label'] == "Negative" else "badge-unc")
        history_html += f"""
        <div class="history-item">
            <span class="history-badge {badge_cls}">{h['label']}</span>
            <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{h['text']}</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#3a3a55;flex-shrink:0">{h['conf']:.0f}%</span>
        </div>"""

    st.markdown(f'<div style="background:#0a0a15;border:1px solid #1a1a2e;border-radius:12px;overflow:hidden;">{history_html}</div>', unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.15em;color:#2a2a3a;text-transform:uppercase;">
    BiLSTM · Attention · IMDB 50K · TensorFlow
</div>
""", unsafe_allow_html=True)
