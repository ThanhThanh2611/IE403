import requests
import pandas as pd
import time
from langdetect import detect

headers = {
    "User-Agent": "Mozilla/5.0"
}

subreddit = "VietNam"

posts = []
comments_data = []

# lấy danh sách posts
url = f"https://www.reddit.com/r/{subreddit}/new.json?limit=100"

res = requests.get(url, headers=headers)
data = res.json()

for p in data["data"]["children"]:

    post_id = p["data"]["id"]
    posts.append(post_id)

print("Collected posts:", len(posts))


def is_vietnamese(text):
    try:
        return detect(text) == "vi"
    except:
        return False


# crawl comment từng post
for post_id in posts:

    comment_url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"

    try:

        res = requests.get(comment_url, headers=headers)
        json_data = res.json()

        comments = json_data[1]["data"]["children"]

        for c in comments:

            if c["kind"] != "t1":
                continue

            text = c["data"]["body"]

            if len(text) > 5 and is_vietnamese(text):

                comments_data.append({
                    "platform": "reddit",
                    "text": text
                })

                print(text[:80])

    except:
        pass

    time.sleep(1)


df = pd.DataFrame(comments_data)

df.to_csv(
    "reddit_vietnamese_comments.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Total Vietnamese comments:", len(df))