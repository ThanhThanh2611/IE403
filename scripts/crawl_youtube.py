from googleapiclient.discovery import build
import pandas as pd

API_KEY = "YOUR_API_KEY"

youtube = build("youtube", "v3", developerKey=API_KEY)

video_id = "VIDEO_ID"

comments = []

request = youtube.commentThreads().list(
    part="snippet",
    videoId=video_id,
    maxResults=100
)

response = request.execute()

for item in response["items"]:
    comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
    comments.append(comment)

df = pd.DataFrame(comments, columns=["comment"])

df.to_csv("youtube_comments.csv", index=False)

print("Done! Comments saved.")