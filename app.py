# app.py
# Unspoken Meaning Detector – Robust ML (TF-IDF + Logistic Regression)
# ✔ Single file (app.py only)
# ✔ No saved models / no extra files
# ✔ Stable on Streamlit Cloud (Python 3.13)
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
CSV_FILE = "unspoken_meaning_dataset_200rows_context(2).csv"

# ───────────────────────────────────────────────
# LOAD DATA + TRAIN MODEL (CACHED, SAFE)
# ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model from dataset…")
def load_and_train():
    if not Path(CSV_FILE).is_file():
        st.error(f"CSV file not found: {CSV_FILE}")
        st.stop()

    df = pd.read_csv(CSV_FILE)

    # Build rich combined text (VERY IMPORTANT)
    df["text"] = (
        "Message: " + df["message"].astype(str) +
        " | Context: " + df["context"].astype(str) +
        " | Situation: " + df["situation"].astype(str) +
        " | Role: " + df["speaker_role"].astype(str)
    )

    X = df["text"]
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ⚠️ FIXED: removed incompatible parameters
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=8000,
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)

    mappings = {
        "labels": sorted(df["label"].unique()),
        "label_to_emoji": df.groupby("label")["emoji_suggestion"].first().to_dict(),
        "label_to_meaning": df.groupby("label")["hidden_meaning"].first().to_dict(),
        "accuracy": acc
    }

    return model, mappings

# ───────────────────────────────────────────────
# PREDICTION
# ───────────────────────────────────────────────
def predict_hidden_intent(message, context, situation, role, model):
    text = f"Message: {message} | Context: {context} | Situation: {situation} | Role: {role}"

    probs = model.predict_proba([text])[0]
    labels = model.classes_

    top_idx = int(np.argmax(probs))
    label = labels[top_idx]
    confidence = float(probs[top_idx]) * 100

    prob_map = dict(zip(labels, probs))
    return label, confidence, prob_map, text

# ───────────────────────────────────────────────
# STREAMLIT UI
# ───────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Unspoken Meaning Detector", layout="wide")

    st.title("🕵️‍♂️ Unspoken Meaning Detector")
    st.caption("Detecting hidden emotional intent behind everyday messages")

    model, mappings = load_and_train()

    st.success(f"Model trained · Validation accuracy: {mappings['accuracy']:.2%}")

    message = st.text_area(
        "Paste or type the message here",
        height=100,
        placeholder="It's fine.\nSure, why not.\nWe’ll see.\nGood for you.",
    )

    st.markdown("**Context information**")

    c1, c2, c3 = st.columns(3)
    with c1:
        ctx = st.selectbox(
            "Context",
            ["friend_chat", "family_chat", "workplace", "relationship", "classroom"],
        )
    with c2:
        sit = st.selectbox(
            "Situation",
            ["casual_chat", "decision_making", "emotional", "conflict", "argument"],
        )
    with c3:
        role = st.selectbox(
            "Role",
            ["friend", "student", "parent", "partner", "manager"],
        )

    if st.button("Analyze Message", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Please enter a message.")
            return

        label, conf, prob_map, used_text = predict_hidden_intent(
            message, ctx, sit, role, model
        )

        emoji = mappings["label_to_emoji"].get(label, "😶")
        meaning = mappings["label_to_meaning"].get(label, "—")

        st.markdown("---")

        left, right = st.columns([5, 4])

        with left:
            st.subheader("Detected Intent")
            st.markdown(
                f"""
                <div style="background:#eef2ff; padding:1.4rem; border-radius:12px; text-align:center;">
                    <h1 style="margin:0; color:#1e3a8a; font-size:2.6rem;">
                        {label.replace('_',' ').title()}
                    </h1>
                    <p style="font-size:1.4rem; margin:0.6rem 0 0;">
                        {conf:.1f}% confidence
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:100px; text-align:center; margin:1.2rem 0;'>{emoji}</div>",
                unsafe_allow_html=True,
            )

        with right:
            st.subheader("What they really meant")
            st.info(meaning)
            st.progress(min(conf / 100, 1.0))

            level = "High" if conf > 75 else "Moderate" if conf > 45 else "Low"
            if level == "High":
                st.error(f"**{level}** – strong emotional signal")
            elif level == "Moderate":
                st.warning(f"**{level}** – mixed emotional undertone")
            else:
                st.success(f"**{level}** – subtle or neutral intent")

        st.markdown("---")
        st.subheader("Probability Distribution (Top 5)")

        top_items = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:5]
        lbls = [k for k, _ in top_items]
        vals = [v for _, v in top_items]

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.barh(lbls, vals)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")

        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.2f}",
                va="center",
                fontsize=9,
            )

        st.pyplot(fig)

        with st.expander("Input sent to model"):
            st.code(used_text)

if __name__ == "__main__":
    main()
