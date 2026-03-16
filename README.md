# Social Media Emotion Dataset (YouTube + TikTok)

## Tổng quan

Bộ dữ liệu này được dựng cho bài toán phân loại cảm xúc (Emotion Classification) dựa trên comment tiếng Việt ở YouTube và TikTok.

- Input: `Sentence` (comment text)
- Output: `Emotion` (1 trong 7 nhãn)

7 nhãn:
- `Anger`
- `Disgust`
- `Fear`
- `Sadness`
- `Enjoyment`
- `Surprise`
- `Other`

## File xuất ra

- `dataset_full_final.csv` (full dataset)
- `train_final.csv` (80%)
- `val_final.csv` (10%)
- `test_final.csv` (10%)

Thêm hỗ trợ:
- `auto_labeled_phobert.csv` (kết quả auto-label với confidence)
- `auto_labeled_phobert_low_confidence.csv` (dòng cần review)
- `auto_labeled_phobert_train_nor_811_like.csv` (format giống mẫu `train_nor_811`)

## Dữ liệu nguồn

- `youtube_vietnamese_comments.csv` (Cột `text` chứa comment)
- `tiktok_comments_dataset.csv` (Cột `comment` chứa comment)

## Pipeline (script)

- `auto_label_phobert.py`: pipeline cơ bản PhoBERT embedding + logistic
- `auto_label_phobert_review.py`: pipeline đầy đủ (train, infer, confidence, map label, output)

## Cách chạy nhanh

1. Cài library:

```bash
pip install torch transformers scikit-learn pandas
```

2. Chuẩn bị seed label (ví dụ `seed_labeled.csv`), dạng:

```
id,Emotion,Sentence
1,Enjoyment,Hài quá, cười đau cả bụng
... 
```

3. Chạy:

```bash
python auto_label_phobert_review.py --seed seed_labeled.csv --output auto_labeled_phobert.csv
```

4. Chuyển nhãn sang tiếng Anh và split giàu:

```bash
# script đã thực hiện trong repo theo hướng mapping và split 80/10/10
```

## Ghi chú

- Mức độ tin cậy (confidence) có trong `auto_labeled_phobert.csv`.
- Phần tiền xử lý/EDA/mô hình không nằm trong file này, có thể thực hiện tiếp với dữ liệu đã tạo.
