```python
# app.py
# Unspoken Meaning Detector – DistilBERT + Context
# Streamlit-safe version (NO Trainer / NO TrainingArguments / Python 3.13 compatible)
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
CSV_FILE      = "unspoken_meaning_dataset_200rows_context(2).csv"
MODEL_NAME    = "distilbert-base-uncased"
MAPPINGS_FILE = "./label_mappings.json"
DEVICE = torch.device("cpu")

# ───────────────────────────────────────────────
# LOAD MODEL + DATA (SAFE FOR STREAMLIT CLOUD)
# ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model and dataset…")
def load_model_and_data():
    if not Path(CSV_FILE).is_file():
        st.error(f"CSV file not found: {CSV_FILE}")
        st.stop()

    df = pd.read_csv(CSV_FILE)

    unique_labels = sorted(df["label"].unique())
    label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
    id2label = {str(i): lbl for lbl, i in label2id.items()}

    mappings = {
        "label_to_id": label2id,
        "id_to_label": id2label,
        "label_to_emoji": df.groupby("label")["emoji_suggestion"].first().to_dict(),
        "label_to_meaning": df.groupby("label")["hidden_meaning"].first().to_dict(),
        "all_labels": unique_labels,
    }

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(unique_labels),
        id2label={int(k): v for k, v in id2label.items()},
        label2id=label2id,
    ).to(DEVICE)
    model.eval()

    return tokenizer, model, mappings, df

# ───────────────────────────────────────────────
# PREDICTION (HEURISTIC + MODEL SAFE)
# ───────────────────────────────────────────────
def predict_hidden_intent(message, context, situation, role, tokenizer, model, mappings):
    text = f"Message: {message} Context: {context} Situation: {situation} Role: {role}"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy().squeeze()
    pred_idx = int(probs.argmax())

    label = mappings["id_to_label"][str(pred_idx)]
    confidence = float(probs[pred_idx]) * 100

    return label, confidence, probs, text

# ───────────────────────────────────────────────
# STREAMLIT UI
# ───────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Unspoken Meaning Detector", layout="wide")

    st.title("🕵️‍♂️ Unspoken Meaning Detector")
    st.caption("Detecting hidden emotional intent behind everyday messages")

    tokenizer, model, mappings, df = load_model_and_data()

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

        label, conf, probs, used_text = predict_hidden_intent(
            message, ctx, sit, role, tokenizer, model, mappings
        )

        emoji = mappings["label_to_emoji"].get(label, "😶")
        meaning = mappings["label_to_meaning"].get(label, "—")

        st.markdown("---")

        left, right = st.columns([5, 4])

        with left:
            st.subheader("Detected Intent")
            st.markdown(
                f"""
                <div style="background:#f0f4ff; padding:1.4rem; border-radius:12px; text-align:center;">
                    <h1 style="margin:0; color:#1e40af; font-size:2.8rem;">
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
                f"<div style='font-size:110px; text-align:center; margin:1.2rem 0;'>{emoji}</div>",
                unsafe_allow_html=True,
            )

        with right:
            st.subheader("What they really meant")
            st.info(meaning)
            st.progress(min(conf / 100, 1.0))

            level = "High" if conf > 80 else "Moderate" if conf > 45 else "Low"
            if level == "High":
                st.error(f"**{level}** – strong suppressed emotion likely")
            elif level == "Moderate":
                st.warning(f"**{level}** – noticeable undertone")
            else:
                st.success(f"**{level}** – mostly literal")

        st.markdown("---")
        st.subheader("Model's Probability Breakdown (Top 5)")

        prob_pairs = sorted(
            zip(mappings["all_labels"], probs), key=lambda x: x[1], reverse=True
        )[:5]
        lbls, prs = zip(*prob_pairs)

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.barh(lbls, prs)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")

        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}",
                va="center",
                fontsize=9,
            )

        st.pyplot(fig)

        with st.expander("Input sent to model"):
            st.code(used_text)

if __name__ == "__main__":
    main()
```
