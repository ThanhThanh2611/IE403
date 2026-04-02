from googleapiclient.discovery import build
import pandas as pd
from langdetect import detect
import time

# ==============================
# 1. API KEY
# ==============================

API_KEY = "AIzaSyB7MhTS8K52PLlI30OTMNvetlxUA8I6k5U"

youtube = build("youtube", "v3", developerKey=API_KEY)

# ==============================
# 2. KEYWORDS TIẾNG VIỆT
# ==============================

keywords = [
    "review điện thoại",
    "tin tức Việt Nam",
    "vlog Việt Nam",
    "ẩm thực Việt Nam",
    "du lịch Việt Nam",
    "game Việt Nam",
    "show truyền hình Việt Nam",
    "drama Việt Nam",
    "hài Việt Nam",
    "phim Việt Nam"
]

# ==============================
# 3. TÌM VIDEO ID
# ==============================

video_ids = []

for keyword in keywords:

    request = youtube.search().list(
        q=keyword,
        part="snippet",
        type="video",
        maxResults=10,
        regionCode="VN"
    )

    response = request.execute()

    for item in response["items"]:
        video_id = item["id"]["videoId"]
        video_ids.append(video_id)

print("Total videos:", len(video_ids))

# ==============================
# 4. CRAWL COMMENT
# ==============================

comments_data = []

for video_id in video_ids:

    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        )

        response = request.execute()

        for item in response["items"]:

            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]

            try:
                if detect(comment) == "vi":

                    comments_data.append({
                        "video_id": video_id,
                        "platform": "youtube",
                        "text": comment
                    })

            except:
                continue

        print("Collected from video:", video_id)

        time.sleep(1)

    except:
        print("Skip video:", video_id)
        continue

# ==============================
# 5. LƯU DATASET
# ==============================

df = pd.DataFrame(comments_data)

df.to_csv("youtube_vietnamese_comments.csv", index=False)

print("Done! Total comments:", len(df))