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

    embeds = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size].tolist()
            enc = tokenizer(chunk,
                            padding=True,
                            truncation=True,
                            max_length=max_len,
                            return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            # Mean pooling over token embeddings (exclude padding via attention mask)
            mask = enc["attention_mask"].unsqueeze(-1)
            embeddings = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            embeds.append(embeddings.cpu())

    embeddings = torch.cat(embeds, dim=0)
    return embeddings.numpy()


def train_and_predict(seed_csv,
                      output_file,
                      youtube_csv="youtube_vietnamese_comments.csv",
                      tiktok_csv="tiktok_comments_dataset.csv",
                      val_size=0.2,
                      random_state=2026):

    print("[1] Load seed labeled data")
    df = pd.read_csv(seed_csv)
    assert "Sentence" in df.columns and "Emotion" in df.columns, "Seed file cần có cột Sentence, Emotion"
    df = df.dropna(subset=["Sentence", "Emotion"]).reset_index(drop=True)

    print("[2] Load PhoBERT")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = AutoModel.from_pretrained("vinai/phobert-base")

    print("[3] Tạo embedding cho dữ liệu seed")
    X = embed_texts(model, tokenizer, df["Sentence"], max_len=128, batch_size=64)

    le = LabelEncoder()
    y = le.fit_transform(df["Emotion"].str.strip().str.capitalize().replace({"Neutral": "Other"}))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size,
                                                      random_state=random_state,
                                                      stratify=y)

    print("[4] Train classifier (Logistic Regression)")
    clf = LogisticRegression(max_iter=5000, multi_class="ovr", class_weight="balanced")
    clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)
    print("Val classification report:\n", classification_report(y_val, y_val_pred, target_names=le.classes_))

    print("[5] Dự đoán lên tất cả dữ liệu mới từ YouTube/TikTok")
    yt = pd.read_csv(youtube_csv)
    tt = pd.read_csv(tiktok_csv)

    df_all = pd.concat([
        yt["text"].rename("Sentence"),
        tt["comment"].rename("Sentence")
    ], ignore_index=True).to_frame()

    df_all["Sentence"] = df_all["Sentence"].astype(str)
    X_new = embed_texts(model, tokenizer, df_all["Sentence"], max_len=128, batch_size=64)
    y_new = clf.predict(X_new)
    prob = clf.predict_proba(X_new).max(axis=1)

    df_all["Emotion"] = le.inverse_transform(y_new)
    df_all["Confidence"] = prob
    df_all["id"] = range(1, len(df_all) + 1)

    out = df_all[["id", "Emotion", "Sentence", "Confidence"]]
    out.to_csv(output_file, index=False)

    print(f"[6] Lưu kết quả auto-labeled: {output_file}")
    print("Hoàn tất.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto label comments using PhoBERT + classifier")
    parser.add_argument("--seed", default="seed_labeled.csv", help="CSV seed manual labels")
    parser.add_argument("--output", default="auto_labeled_phobert.csv", help="Output CSV path")
    parser.add_argument("--youtube", default="youtube_vietnamese_comments.csv", help="YouTube comments CSV")
    parser.add_argument("--tiktok", default="tiktok_comments_dataset.csv", help="TikTok comments CSV")
    args = parser.parse_args()

    train_and_predict(args.seed, args.output, args.youtube, args.tiktok)
