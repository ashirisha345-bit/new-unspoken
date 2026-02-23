# app.py
# Unspoken Meaning Detector – DistilBERT + Context – single file version
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import torch
import json
import os
import shutil
from pathlib import Path
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
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

DEVICE = torch.device("cpu")

# ───────────────────────────────────────────────
# DATA & MODEL PREPARATION
# ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Preparing model (first run may take 5–15 minutes)…")
def load_or_train_model():
    if Path(MODEL_DIR).is_dir() and Path(MAPPINGS_FILE).is_file():
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
        model.to(DEVICE)
        model.eval()
        with open(MAPPINGS_FILE, "r", encoding="utf-8") as f:
            mappings = json.load(f)
        return tokenizer, model, mappings

    if not Path(CSV_FILE).is_file():
        st.error(f"CSV file not found: {CSV_FILE}")
        st.stop()

    df = pd.read_csv(CSV_FILE)

    unique_labels = sorted(df["label"].unique())
    label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    mappings = {
        "label_to_id": label2id,
        "id_to_label": {str(k): v for k, v in id2label.items()},
        "label_to_emoji": df.groupby("label")["emoji_suggestion"].first().to_dict(),
        "label_to_meaning": df.groupby("label")["hidden_meaning"].first().to_dict(),
        "all_labels": unique_labels,
    }

    def build_input(row):
        return (
            f"Message: {row['message']} "
            f"Context: {row['context']} "
            f"Situation: {row['situation']} "
            f"Role: {row['speaker_role']}"
        )

    df["text"] = df.apply(build_input, axis=1)
    df["labels"] = df["label"].map(label2id)

    train_df, eval_df = train_test_split(
        df, test_size=0.15, stratify=df["labels"], random_state=42
    )

    train_ds = Dataset.from_pandas(train_df[["text", "labels"]], preserve_index=False)
    eval_ds  = Dataset.from_pandas(eval_df[["text", "labels"]], preserve_index=False)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    eval_ds  = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
    ).to(DEVICE)

    args = TrainingArguments(
        output_dir="./training_tmp",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=False,
        bf16=False,
        report_to="none",
        logging_steps=30,
        disable_tqdm=False,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_steps=50,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    with st.spinner("Fine-tuning DistilBERT (CPU mode)..."):
        trainer.train()

    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    with open(MAPPINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)

    shutil.rmtree("./training_tmp", ignore_errors=True)

    model.eval()
    return tokenizer, model, mappings


# ───────────────────────────────────────────────
# PREDICTION
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

    tokenizer, model, mappings = load_or_train_model()

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

        st.markdown("---")
        st.subheader("Model's Probability Breakdown (Top 5)")

        prob_pairs = sorted(
            zip(mappings["all_labels"], probs), key=lambda x: x[1], reverse=True
        )[:5]
        lbls, prs = zip(*prob_pairs)

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.barh(lbls, prs)
        ax.set_xlim(0, 1)
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
