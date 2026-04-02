import pandas as pd

df = pd.read_csv("youtube_vietnamese_comments.csv", encoding="utf-8")

df.to_csv("youtube_vietnamese_comments_fixed.csv",
          index=False,
          encoding="utf-8-sig")

print("Fixed file created")