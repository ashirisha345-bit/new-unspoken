# app.py
# Unspoken Meaning Detector – DistilBERT + Context – single file version
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import torch
import json
import os
import re
from pathlib import Path
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────

CSV_FILE      = "unspoken_meaning_dataset_200rows_context(2).csv"
MODEL_DIR     = "./unspoken_model"
MAPPINGS_FILE = "./label_mappings.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ───────────────────────────────────────────────
# DATA & MODEL PREPARATION (runs once + cached)
# ───────────────────────────────────────────────

@st.cache_resource(show_spinner="Preparing model (first run may take 3–10 minutes)…")
def load_or_train_model():
    if Path(MODEL_DIR).is_dir() and Path(MAPPINGS_FILE).is_file():
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
        with open(MAPPINGS_FILE, "r", encoding="utf-8") as f:
            mappings = json.load(f)
        st.success("Loaded pre-trained model.")
        return tokenizer, model, mappings

    # ── Train from scratch ───────────────────────────────
    if not Path(CSV_FILE).is_file():
        st.error(f"CSV file not found: {CSV_FILE}\nPlease place it in the same folder as this script.")
        st.stop()

    df = pd.read_csv(CSV_FILE)

    # Label setup
    unique_labels = sorted(df['label'].unique())
    label2id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    id2label = {str(idx): lbl for lbl, idx in label2id.items()}

    label2emoji   = df.groupby('label')['emoji_suggestion'].first().to_dict()
    label2meaning = df.groupby('label')['hidden_meaning'].first().to_dict()

    mappings = {
        "label_to_id": label2id,
        "id_to_label": id2label,
        "label_to_emoji": label2emoji,
        "label_to_meaning": label2meaning,
        "all_labels": unique_labels
    }

    # Prepare input format
    def build_input(row):
        return f"Message: {row['message']} Context: {row['context']} Situation: {row['situation']} Role: {row['speaker_role']}"

    df['text'] = df.apply(build_input, axis=1)
    df['labels'] = df['label'].map(label2id)

    # Split
    train_df, eval_df = train_test_split(
        df, test_size=0.15, stratify=df['labels'], random_state=42
    )

    train_ds = Dataset.from_pandas(train_df[['text', 'labels']])
    eval_ds  = Dataset.from_pandas(eval_df[['text', 'labels']])

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    train_ds = train_ds.map(tokenize_fn, batched=True)
    eval_ds  = eval_ds.map(tokenize_fn, batched=True)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir="./training_tmp",
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_steps=10,
    )

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        acc = accuracy_score(p.label_ids, preds)
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    with st.spinner("Fine-tuning DistilBERT on your dataset…"):
        trainer.train()

    # Save
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    with open(MAPPINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)

    st.success("Training complete. Model saved for future runs.")
    return tokenizer, model, mappings


# ───────────────────────────────────────────────
# PREDICTION FUNCTION
# ───────────────────────────────────────────────

def predict_hidden_intent(message, context, situation, role, tokenizer, model, mappings):
    text = f"Message: {message} Context: {context} Situation: {situation} Role: {role}"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
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

    tokenizer, model, mappings = load_or_train_model()

    # ── INPUT ────────────────────────────────────────────
    message = st.text_area(
        "Paste or type the message here",
        height=100,
        placeholder="Examples:\nIt's fine.\nSure, why not.\nWe’ll see.\nGood for you.",
        key="msg"
    )

    st.markdown("**Context information** (makes prediction more accurate)")

    c1, c2, c3 = st.columns(3)
    with c1:
        ctx = st.selectbox("Context", [
            "friend_chat", "family_chat", "workplace", "relationship", "classroom"
        ])
    with c2:
        sit = st.selectbox("Situation", [
            "casual_chat", "decision_making", "emotional", "conflict", "argument"
        ])
    with c3:
        role = st.selectbox("Role", [
            "friend", "student", "parent", "partner", "manager"
        ])

    if st.button("Analyze Message", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Please enter a message.")
            return

        with st.spinner("Detecting hidden meaning…"):
            label, conf, probs, used_text = predict_hidden_intent(
                message, ctx, sit, role, tokenizer, model, mappings
            )

        emoji = mappings["label_to_emoji"].get(label, "😶")
        meaning = mappings["label_to_meaning"].get(label, "—")

        # ── RESULT LAYOUT ────────────────────────────────────
        st.markdown("---")

        left, right = st.columns([5, 4])

        with left:
            st.subheader("Detected Intent")
            st.markdown(f"""
            <div style="background:#f0f4ff; padding:1.4rem; border-radius:12px; text-align:center;">
                <h1 style="margin:0; color:#1e40af; font-size:2.8rem;">{label.replace('_',' ').title()}</h1>
                <p style="font-size:1.4rem; margin:0.6rem 0 0;">{conf:.1f}% confidence</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"<div style='font-size:110px; text-align:center; margin:1.2rem 0;'>{emoji}</div>", unsafe_allow_html=True)

        with right:
            st.subheader("What they really meant")
            st.info(meaning)

            st.markdown("**Emotional strength**")
            st.progress(min(conf / 100, 1.0))

            level = "High" if conf > 80 else "Moderate" if conf > 45 else "Low"
            if level == "High":
                st.error(f"**{level}** – strong suppressed emotion likely")
            elif level == "Moderate":
                st.warning(f"**{level}** – noticeable undertone")
            else:
                st.success(f"**{level}** – mostly literal")

        # Insight
        st.markdown("---")
        with st.expander("🧠 Quick Insight", expanded=True):
            st.write(
                f"In a **{sit}** situation from a **{role}** perspective, "
                f"**{label}** tone often signals **{meaning.lower()}** "
                "while trying to keep things surface-polite or avoid escalation."
            )

        # Distribution chart
        st.markdown("---")
        st.subheader("Model's Probability Breakdown (Top 5)")

        prob_pairs = sorted(zip(mappings["all_labels"], probs), key=lambda x: x[1], reverse=True)[:5]
        lbls, prs = zip(*prob_pairs)

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.barh(lbls, prs, color="#60a5fa")
        ax.barh(lbls[0], prs[0], color="#f59e0b")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{w:.3f}", va="center", fontsize=9)
        st.pyplot(fig)

        with st.expander("Input sent to model"):
            st.code(used_text, language=None)

if __name__ == "__main__":
    main()
