"""
AI UI Copywriting Lab
======================
A Streamlit application that helps UI/UX designers generate, analyze,
and improve UI copy (button text, error messages, success messages,
notifications, and onboarding messages).

Tech stack: Python, Streamlit, Pandas, NumPy, Scikit-learn, OpenAI (optional)

Author: Internship submission
"""

import os
import io
import random
import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# Optional: OpenAI API is only used if the user supplies a key.
# The app works fully offline using template-based generation if no key is given.
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ----------------------------------------------------------------------------
# PAGE CONFIG / GLOBAL STYLE
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI UI Copywriting Lab",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A little custom CSS for a cleaner, modern look.
CUSTOM_CSS = """
<style>
    /* Main background tweak */
    .main {
        background-color: #fafbfc;
    }
    /* Card-like containers */
    .copy-card {
        background-color: #ffffff;
        border: 1px solid #e6e9ef;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .score-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.95rem;
        color: white;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "ui_copy_dataset.csv")


@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    """Load the sample UI copy dataset used for ML training & templates."""
    df = pd.read_csv(path)
    return df


df_dataset = load_dataset(DATA_PATH)

COPY_TYPES = [
    "Button Text",
    "Error Message",
    "Success Message",
    "Notification",
    "Onboarding Message",
]

APP_CATEGORIES = sorted(df_dataset["app_category"].unique().tolist())


# ----------------------------------------------------------------------------
# SIMPLE TEMPLATE-BASED COPY GENERATOR (fallback / offline mode)
# ----------------------------------------------------------------------------
# These templates give varied, realistic UI copy without needing an API key.
TEMPLATES = {
    "Button Text": [
        "Get Started",
        "Buy Now",
        "Try {category} Free",
        "Continue",
        "Sign Up Today",
        "Explore {category}",
        "Add to Cart",
        "Start Now",
    ],
    "Error Message": [
        "Something went wrong. Please try again.",
        "We couldn't process your request. Check your connection and retry.",
        "Oops! That didn't work. Please try again in a moment.",
        "Your {category} session has expired. Please log in again.",
        "Invalid input. Please double-check and try again.",
    ],
    "Success Message": [
        "All set! Your changes have been saved.",
        "Success! Your {category} request was completed.",
        "Great job! You're all done.",
        "Your order has been placed successfully.",
        "Done! Everything is up to date.",
    ],
    "Notification": [
        "You have a new update in {category}.",
        "Reminder: Don't miss your scheduled activity.",
        "New activity is waiting for you.",
        "Your {category} update is ready to view.",
        "Heads up! Something needs your attention.",
    ],
    "Onboarding Message": [
        "Welcome! Let's set up your {category} experience.",
        "Hi there! We're excited to have you here.",
        "Welcome aboard! Let's get you started in three easy steps.",
        "Glad you're here! Let's personalize your {category} journey.",
        "Welcome! Discover everything {category} has to offer.",
    ],
}


def generate_copy_template(copy_type: str, category: str) -> str:
    """Generate UI copy using simple randomized templates (offline mode)."""
    options = TEMPLATES.get(copy_type, ["Sample text"])
    chosen = random.choice(options)
    return chosen.format(category=category)


def generate_copy_openai(copy_type: str, category: str, api_key: str) -> str:
    """Generate UI copy using the OpenAI API, if a key is provided."""
    client = OpenAI(api_key=api_key)
    prompt = (
        f"Write a single short, clear, user-friendly piece of UI copy for a "
        f"'{copy_type}' in a '{category}' mobile/web application. "
        f"Return only the copy text, no quotes, no explanation, max 12 words."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


# ----------------------------------------------------------------------------
# COPY QUALITY ANALYSIS
# ----------------------------------------------------------------------------
# A small list of "difficult" / jargon-y words that hurt UI copy clarity.
DIFFICULT_WORDS = {
    "utilize", "facilitate", "initiate", "commence", "anomaly",
    "unanticipated", "transaction", "exception", "regimen", "subsequently",
    "leverage", "optimal", "functionality", "implement", "terminate",
    "erroneous", "proceed", "finalize", "ascertain", "verification",
    "authentication", "configuration", "parameter", "instantiate",
}

COMMON_SIMPLE_WORDS = {
    "go", "buy", "try", "add", "save", "send", "next", "done", "ok",
    "now", "new", "start", "stop", "open", "close", "edit", "share",
    "view", "help", "back", "yes", "no",
}


def analyze_copy(text: str) -> dict:
    """
    Analyze a piece of UI copy and return a dictionary with:
      - clarity score (0-100)
      - simplicity score (0-100)
      - friendliness score (0-100)
      - overall score (0-100)
      - list of difficult words found
    Scoring is rule-based (heuristic), suitable for a beginner/intermediate
    internship-level project.
    """
    words = [w.strip(".,!?:;\"'").lower() for w in text.split() if w.strip()]
    word_count = max(len(words), 1)

    # --- Clarity: shorter sentences & fewer long words = clearer ---
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    clarity = 100 - (avg_word_len - 4) * 8  # penalize long avg word length
    clarity = np.clip(clarity, 0, 100)

    # --- Simplicity: penalize for length and difficult words ---
    difficult_found = [w for w in words if w in DIFFICULT_WORDS]
    length_penalty = max(0, word_count - 8) * 5
    difficult_penalty = len(difficult_found) * 15
    simplicity = 100 - length_penalty - difficult_penalty
    simplicity = np.clip(simplicity, 0, 100)

    # --- Friendliness: reward simple/common/friendly words, exclamation ---
    friendly_hits = sum(1 for w in words if w in COMMON_SIMPLE_WORDS)
    friendliness = 60 + friendly_hits * 8
    if "!" in text:
        friendliness += 10
    if any(g in text.lower() for g in ["welcome", "great", "thank", "let's", "you"]):
        friendliness += 10
    friendliness = np.clip(friendliness, 0, 100)

    overall = round((clarity * 0.35 + simplicity * 0.4 + friendliness * 0.25), 1)

    return {
        "clarity": round(clarity, 1),
        "simplicity": round(simplicity, 1),
        "friendliness": round(friendliness, 1),
        "overall": overall,
        "difficult_words": sorted(set(difficult_found)),
        "word_count": word_count,
    }


def suggest_improvement(text: str, analysis: dict) -> str:
    """
    Suggest a shorter / clearer alternative to the given copy.
    Removes difficult words and trims length using simple heuristics.
    """
    words = text.split()
    cleaned_words = []
    for w in words:
        bare = w.strip(".,!?:;\"'").lower()
        if bare in DIFFICULT_WORDS:
            continue  # drop hard words
        cleaned_words.append(w)

    # Trim to at most 8 words for buttons/short copy, 14 for longer messages
    max_len = 8 if analysis["word_count"] <= 10 else 14
    cleaned_words = cleaned_words[:max_len]

    suggestion = " ".join(cleaned_words).strip()
    if not suggestion:
        suggestion = "Try again"

    # Capitalize first letter
    suggestion = suggestion[0].upper() + suggestion[1:] if suggestion else suggestion
    return suggestion


# ----------------------------------------------------------------------------
# MACHINE LEARNING CLASSIFICATION MODEL
# ----------------------------------------------------------------------------
@st.cache_resource
def train_classifier(data: pd.DataFrame):
    """
    Train a simple Naive Bayes text classifier (TF-IDF + MultinomialNB)
    on the sample dataset to classify UI copy as:
      Excellent / Good / Needs Improvement
    This is intentionally simple and beginner-friendly.
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=200)
    X = vectorizer.fit_transform(data["text"])

    encoder = LabelEncoder()
    y = encoder.fit_transform(data["label"])

    model = MultinomialNB()
    model.fit(X, y)

    return model, vectorizer, encoder


ml_model, ml_vectorizer, ml_encoder = train_classifier(df_dataset)


def classify_copy_ml(text: str) -> str:
    """Use the trained ML model to classify a piece of UI copy."""
    X_new = ml_vectorizer.transform([text])
    pred = ml_model.predict(X_new)[0]
    label = ml_encoder.inverse_transform([pred])[0]
    return label


def classify_copy_score(overall_score: float) -> str:
    """
    Secondary rule-based classification driven by the heuristic score,
    used to cross-check / blend with the ML model's prediction.
    """
    if overall_score >= 80:
        return "Excellent"
    elif overall_score >= 60:
        return "Good"
    else:
        return "Needs Improvement"


def score_color(score: float) -> str:
    """Return a hex color based on score band, used for badges."""
    if score >= 80:
        return "#16a34a"  # green
    elif score >= 60:
        return "#f59e0b"  # amber
    else:
        return "#dc2626"  # red


# ----------------------------------------------------------------------------
# SESSION STATE INIT
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: type, category, text, analysis, label

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
st.sidebar.title("✨ AI UI Copywriting Lab")
st.sidebar.caption("Generate. Analyze. Improve. Ship better UI copy.")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📝 Copy Generator",
        "📊 Quality Analysis & Dashboard",
        "💡 Improvement Suggestions",
        "📁 History & Report Download",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")

use_openai = st.sidebar.checkbox("Use OpenAI API (optional)", value=False)
openai_key = ""
if use_openai:
    if not OPENAI_AVAILABLE:
        st.sidebar.warning(
            "openai package not installed. Run `pip install openai` to enable this."
        )
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    st.sidebar.caption("Your key is only used locally for this session and never stored.")

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 Tip: Generate a few copy variants, run the analysis, then check "
    "the **Improvement Suggestions** tab to polish your text."
)


# ----------------------------------------------------------------------------
# PAGE: HOME
# ----------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("✨ AI UI Copywriting Lab")
    st.markdown(
        "An AI-powered assistant that helps **UI/UX designers** generate, "
        "analyze, and improve interface copy — button labels, error messages, "
        "success messages, notifications, and onboarding text."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="copy-card"><h3>📝 Generate</h3>'
            "<p>Create UI copy for 5 categories of interface text in one click, "
            "with or without the OpenAI API.</p></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="copy-card"><h3>📊 Analyze</h3>'
            "<p>Score copy on Clarity, Simplicity, and User-Friendliness, "
            "and classify it using a trained ML model.</p></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="copy-card"><h3>💡 Improve</h3>'
            "<p>Get shorter, clearer alternatives and spot jargon or "
            "hard-to-read words instantly.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### 📦 Sample Dataset Preview")
    st.dataframe(df_dataset.head(10), use_container_width=True)

    st.markdown("### 🚀 How to use this app")
    st.markdown(
        """
        1. Go to **📝 Copy Generator** → pick a copy type & app category → generate copy.
        2. Go to **📊 Quality Analysis & Dashboard** → see scores, ML classification, and charts.
        3. Go to **💡 Improvement Suggestions** → get a polished alternative.
        4. Go to **📁 History & Report Download** → export everything as a `.txt` report.
        """
    )


# ----------------------------------------------------------------------------
# PAGE: COPY GENERATOR
# ----------------------------------------------------------------------------
elif page == "📝 Copy Generator":
    st.title("📝 UI Copy Generator")
    st.write("Select a copy type and an app category, then generate UI copy.")

    col1, col2 = st.columns(2)
    with col1:
        copy_type = st.selectbox("Copy Type", COPY_TYPES)
    with col2:
        category_choice = st.selectbox(
            "App Category", APP_CATEGORIES + ["Custom..."]
        )
        if category_choice == "Custom...":
            category = st.text_input("Enter custom app category", "Productivity")
        else:
            category = category_choice

    n_variants = st.slider("Number of variants to generate", 1, 5, 3)

    if st.button("🚀 Generate UI Copy", type="primary"):
        generated_texts = []
        for _ in range(n_variants):
            if use_openai and OPENAI_AVAILABLE and openai_key:
                try:
                    text = generate_copy_openai(copy_type, category, openai_key)
                except Exception as e:
                    st.error(f"OpenAI API error: {e}. Falling back to offline mode.")
                    text = generate_copy_template(copy_type, category)
            else:
                text = generate_copy_template(copy_type, category)
            generated_texts.append(text)

        # Deduplicate while preserving order
        generated_texts = list(dict.fromkeys(generated_texts))

        st.session_state["generated_texts"] = generated_texts
        st.session_state["generated_meta"] = {"copy_type": copy_type, "category": category}

    if "generated_texts" in st.session_state:
        st.markdown("### ✅ Generated Copy")
        meta = st.session_state["generated_meta"]
        for i, text in enumerate(st.session_state["generated_texts"]):
            st.markdown(
                f'<div class="copy-card"><b>Variant {i+1}:</b> {text}</div>',
                unsafe_allow_html=True,
            )

        selected_text = st.selectbox(
            "Select a variant to analyze further",
            st.session_state["generated_texts"],
        )

        if st.button("➡️ Send to Quality Analysis"):
            st.session_state.last_result = {
                "copy_type": meta["copy_type"],
                "category": meta["category"],
                "text": selected_text,
            }
            st.success("Sent! Open the '📊 Quality Analysis & Dashboard' tab to view results.")


# ----------------------------------------------------------------------------
# PAGE: QUALITY ANALYSIS & DASHBOARD
# ----------------------------------------------------------------------------
elif page == "📊 Quality Analysis & Dashboard":
    st.title("📊 Copy Quality Analysis & Dashboard")

    default_text = ""
    default_type = COPY_TYPES[0]
    default_category = APP_CATEGORIES[0]
    if st.session_state.last_result:
        default_text = st.session_state.last_result["text"]
        default_type = st.session_state.last_result["copy_type"]
        default_category = st.session_state.last_result["category"]

    with st.form("analysis_form"):
        col1, col2 = st.columns(2)
        with col1:
            copy_type = st.selectbox("Copy Type", COPY_TYPES, index=COPY_TYPES.index(default_type))
        with col2:
            category = st.selectbox(
                "App Category", APP_CATEGORIES,
                index=APP_CATEGORIES.index(default_category) if default_category in APP_CATEGORIES else 0,
            )
        text_input = st.text_area("UI Copy to Analyze", value=default_text, height=100)
        submitted = st.form_submit_button("🔍 Analyze Copy", type="primary")

    if submitted and text_input.strip():
        analysis = analyze_copy(text_input)
        ml_label = classify_copy_ml(text_input)
        rule_label = classify_copy_score(analysis["overall"])

        # Save to history
        record = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "copy_type": copy_type,
            "category": category,
            "text": text_input,
            "clarity": analysis["clarity"],
            "simplicity": analysis["simplicity"],
            "friendliness": analysis["friendliness"],
            "overall": analysis["overall"],
            "ml_label": ml_label,
            "rule_label": rule_label,
            "difficult_words": ", ".join(analysis["difficult_words"]) or "None",
        }
        st.session_state.history.append(record)
        st.session_state.last_result = {
            "copy_type": copy_type, "category": category, "text": text_input,
            "analysis": analysis, "ml_label": ml_label, "rule_label": rule_label,
        }

    if st.session_state.last_result and "analysis" in st.session_state.last_result:
        result = st.session_state.last_result
        analysis = result["analysis"]

        st.markdown("### 📋 Result")
        st.markdown(
            f'<div class="copy-card"><b>Copy:</b> "{result["text"]}"<br>'
            f'<span class="metric-label">{result["copy_type"]} • {result["category"]}</span></div>',
            unsafe_allow_html=True,
        )

        c = score_color(analysis["overall"])
        st.markdown(
            f'<span class="score-badge" style="background-color:{c};">'
            f'Overall Score: {analysis["overall"]}/100</span>',
            unsafe_allow_html=True,
        )

        st.markdown("#### 🧮 Score Breakdown")
        m1, m2, m3 = st.columns(3)
        m1.metric("Clarity", f'{analysis["clarity"]}/100')
        m2.metric("Simplicity", f'{analysis["simplicity"]}/100')
        m3.metric("User Friendliness", f'{analysis["friendliness"]}/100')

        st.markdown("#### 🤖 AI Classification")
        col_a, col_b = st.columns(2)
        col_a.success(f"ML Model Prediction: **{result['ml_label']}**")
        col_b.info(f"Rule-based (score) Prediction: **{result['rule_label']}**")

        if analysis["difficult_words"]:
            st.warning(f"⚠️ Difficult words detected: {', '.join(analysis['difficult_words'])}")
        else:
            st.success("✅ No difficult words detected.")

        # ---- Charts ----
        st.markdown("#### 📈 Score Visualization")
        chart_df = pd.DataFrame({
            "Metric": ["Clarity", "Simplicity", "User Friendliness"],
            "Score": [analysis["clarity"], analysis["simplicity"], analysis["friendliness"]],
        }).set_index("Metric")
        st.bar_chart(chart_df)

        if len(st.session_state.history) > 1:
            st.markdown("#### 📉 Score Trend Over Session History")
            trend_df = pd.DataFrame(st.session_state.history)[["overall"]]
            trend_df.index = [f"#{i+1}" for i in range(len(trend_df))]
            st.line_chart(trend_df)
    else:
        st.info("Enter or generate some UI copy, then click **Analyze Copy** to see results here.")


# ----------------------------------------------------------------------------
# PAGE: IMPROVEMENT SUGGESTIONS
# ----------------------------------------------------------------------------
elif page == "💡 Improvement Suggestions":
    st.title("💡 Copy Improvement Suggestions")

    default_text = st.session_state.last_result["text"] if st.session_state.last_result else ""
    text_input = st.text_area("Paste UI copy to improve", value=default_text, height=100)

    if st.button("✨ Suggest Improvement", type="primary") and text_input.strip():
        analysis = analyze_copy(text_input)
        suggestion = suggest_improvement(text_input, analysis)
        suggestion_analysis = analyze_copy(suggestion)

        st.markdown("### 🔍 Original vs Suggested")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original**")
            st.markdown(f'<div class="copy-card">{text_input}</div>', unsafe_allow_html=True)
            st.metric("Original Score", f'{analysis["overall"]}/100')
        with col2:
            st.markdown("**Suggested Alternative**")
            st.markdown(f'<div class="copy-card">{suggestion}</div>', unsafe_allow_html=True)
            st.metric(
                "Suggested Score", f'{suggestion_analysis["overall"]}/100',
                delta=round(suggestion_analysis["overall"] - analysis["overall"], 1),
            )

        if analysis["difficult_words"]:
            st.markdown("### ⚠️ Difficult Words Highlighted")
            highlighted = text_input
            for w in analysis["difficult_words"]:
                highlighted = highlighted.replace(
                    w, f"**:red[{w}]**"
                )
            st.markdown(highlighted)
        else:
            st.success("No difficult words found in the original text!")

        st.markdown("### 📝 Tips for Better UI Copy")
        st.markdown(
            """
            - Keep button text to **1–3 words** ("Buy Now", "Get Started").
            - Avoid technical jargon (e.g. *initiate*, *anomaly*, *facilitate*).
            - Use active voice and speak directly to the user ("You", "Your").
            - Keep error messages helpful, not blaming — explain **what happened**
              and **what to do next**.
            - Add warmth to success/onboarding messages with positive words
              ("Great!", "Welcome!", "You're all set!").
            """
        )
    else:
        st.info("Enter some UI copy above and click **Suggest Improvement**.")


# ----------------------------------------------------------------------------
# PAGE: HISTORY & REPORT DOWNLOAD
# ----------------------------------------------------------------------------
elif page == "📁 History & Report Download":
    st.title("📁 Session History & Report Download")

    if not st.session_state.history:
        st.info("No analyzed copy yet. Go to **Quality Analysis & Dashboard** to analyze some text first.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.markdown("### 🗂️ Analysis History")
        st.dataframe(hist_df, use_container_width=True)

        st.markdown("### 📊 Label Distribution")
        label_counts = hist_df["ml_label"].value_counts()
        st.bar_chart(label_counts)

        st.markdown("### 📈 Average Scores")
        avg_scores = hist_df[["clarity", "simplicity", "friendliness", "overall"]].mean().round(1)
        st.dataframe(avg_scores.rename("Average Score"), use_container_width=True)

        # ---- Build downloadable text report ----
        report_lines = [
            "AI UI COPYWRITING LAB - ANALYSIS REPORT",
            "=" * 50,
            f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total entries analyzed: {len(hist_df)}",
            "",
        ]
        for i, row in hist_df.iterrows():
            report_lines.append(f"--- Entry #{i+1} ---")
            report_lines.append(f"Timestamp: {row['timestamp']}")
            report_lines.append(f"Copy Type: {row['copy_type']}")
            report_lines.append(f"App Category: {row['category']}")
            report_lines.append(f"Text: {row['text']}")
            report_lines.append(
                f"Scores -> Clarity: {row['clarity']}, Simplicity: {row['simplicity']}, "
                f"Friendliness: {row['friendliness']}, Overall: {row['overall']}"
            )
            report_lines.append(f"ML Classification: {row['ml_label']}")
            report_lines.append(f"Rule-based Classification: {row['rule_label']}")
            report_lines.append(f"Difficult Words: {row['difficult_words']}")
            report_lines.append("")

        report_lines.append("=" * 50)
        report_lines.append("SUMMARY")
        report_lines.append(f"Average Overall Score: {avg_scores['overall']}")
        report_lines.append("Label Distribution:")
        for label, count in label_counts.items():
            report_lines.append(f"  - {label}: {count}")

        report_text = "\n".join(report_lines)

        st.download_button(
            label="⬇️ Download Full Report (.txt)",
            data=report_text,
            file_name=f"ui_copy_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()


# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit · Scikit-learn · Pandas · NumPy")
