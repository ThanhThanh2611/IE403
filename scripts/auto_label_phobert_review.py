import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def embed_texts(model, tokenizer, texts, batch_size=64, max_len=128, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size].tolist()
            enc = tokenizer(batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=max_len,
                            return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).type(out.last_hidden_state.dtype)
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embeddings.append(pooled.cpu())

    embeddings = torch.cat(embeddings, dim=0).numpy()
    return embeddings


def train_classifier(seed_path, phobert_model):
    df = pd.read_csv(seed_path)
    if "Sentence" not in df.columns or "Emotion" not in df.columns:
        raise RuntimeError("Seed file must contain columns: Sentence, Emotion")

    df = df.dropna(subset=["Sentence", "Emotion"]).reset_index(drop=True)
    df["Emotion"] = df["Emotion"].astype(str).str.strip().str.capitalize().replace({"Neutral": "Other"})

    tokenizer = AutoTokenizer.from_pretrained(phobert_model)
    model = AutoModel.from_pretrained(phobert_model)

    print(f"[+] Embedding {len(df)} labeled examples with {phobert_model}")
    X = embed_texts(model, tokenizer, df["Sentence"], max_len=128, batch_size=64)

    le = LabelEncoder()
    y = le.fit_transform(df["Emotion"])

    if len(le.classes_) < 2:
        raise RuntimeError("Need at least 2 emotion classes in seed labels")

    min_count = df["Emotion"].value_counts().min()
    if min_count < 2:
        print("[!] Some classes have fewer than 2 samples. Skip stratified split and train on full data.")
        X_train, y_train = X, y
        X_val, y_val = X, y
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                          random_state=42,
                                                          stratify=y)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("Validation classification report:\n", classification_report(y_val, y_pred, target_names=le.classes_))

    return tokenizer, model, clf, le


def infer_comments(tokenizer, model, clf, le, youtube_path, tiktok_path, output_path, confidence_threshold=0.6):
    yt = pd.read_csv(youtube_path)
    tt = pd.read_csv(tiktok_path)

    if "text" not in yt.columns:
        raise RuntimeError("YouTube file must contain column 'text'")
    if "comment" not in tt.columns:
        raise RuntimeError("TikTok file must contain column 'comment'")

    df_yt = yt[["text"]].rename(columns={"text": "Sentence"})
    df_tt = tt[["comment"]].rename(columns={"comment": "Sentence"})

    df_all = pd.concat([df_yt, df_tt], ignore_index=True).astype(str)
    df_all["Sentence"] = df_all["Sentence"].fillna("")

    print(f"[+] Embedding {len(df_all)} total comments")
    X_all = embed_texts(model, tokenizer, df_all["Sentence"], max_len=128, batch_size=64)

    proba = clf.predict_proba(X_all)
    y_pred = clf.predict(X_all)
    max_conf = proba.max(axis=1)

    df_all["Emotion"] = le.inverse_transform(y_pred)
    df_all["Confidence"] = max_conf
    df_all["id"] = range(1, len(df_all) + 1)

    out = df_all[["id", "Emotion", "Sentence", "Confidence"]]
    out.to_csv(output_path, index=False)

    print(f"[+] Saved auto-labeled output: {output_path}")

    df_low = out[out.Confidence < confidence_threshold].sort_values("Confidence")
    low_path = output_path.replace(".csv", "_low_confidence.csv")
    df_low.to_csv(low_path, index=False)
    print(f"[+] Saved low-confidence rows (<{confidence_threshold}): {low_path}")

    return out, df_low


def prepare_final(out_path):
    df = pd.read_csv(out_path)
    if "id" not in df.columns or "Emotion" not in df.columns or "Sentence" not in df.columns:
        raise RuntimeError("Output file must contain id, Emotion, Sentence")

    df["Emotion"] = df["Emotion"].astype(str).str.strip().str.capitalize().replace({"Neutral":"Other"})
    df["id"] = range(1, len(df)+1)
    final_path = out_path.replace(".csv", "_train_nor_811_like.csv")
    df[["id","Emotion","Sentence"]].to_csv(final_path, index=False)
    print(f"[+] Final train-like file: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto PhoBERT labeling pipeline and review")
    parser.add_argument("--seed", default="seed_labeled.csv", help="Seed labeled csv file")
    parser.add_argument("--youtube", default="youtube_vietnamese_comments.csv", help="YouTube comments csv")
    parser.add_argument("--tiktok", default="tiktok_comments_dataset.csv", help="TikTok comments csv")
    parser.add_argument("--output", default="auto_labeled_phobert.csv", help="Auto-labeled output csv")
    parser.add_argument("--confidence", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--phobert", default="vinai/phobert-base", help="PhoBERT model name")
    args = parser.parse_args()

    tokenizer, model, clf, le = train_classifier(args.seed, args.phobert)
    infer_comments(tokenizer, model, clf, le, args.youtube, args.tiktok, args.output, args.confidence)
    prepare_final(args.output)
