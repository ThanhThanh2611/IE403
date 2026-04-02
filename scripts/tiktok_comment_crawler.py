import asyncio
import pandas as pd
from langdetect import detect, LangDetectException
from TikTokApi import TikTokApi

KEYWORDS = ["ai", "du lịch", "âm nhạc", "công nghệ"]
VIDEO_LIMIT = 20  # mỗi từ khóa
COMMENT_LIMIT = 100

dataset = []

async def main():

    async with TikTokApi() as api:

        await api.create_sessions(
            num_sessions=1,
            headless=False
        )

        for keyword in KEYWORDS:
            print(f"\n=== Searching for keyword: {keyword} ===")
            videos = api.search.search_type(keyword, "item", count=VIDEO_LIMIT)

            count = 0

            async for video in videos:
                if count >= VIDEO_LIMIT:
                    break

                print("Video:", video.id)

                try:
                    comments = video.comments(count=COMMENT_LIMIT)
                    c_count = 0

                    async for comment in comments:
                        if c_count >= COMMENT_LIMIT:
                            break

                        text = comment.text

                        # chỉ lưu comment tiếng việt
                        try:
                            lang = detect(text)
                        except LangDetectException:
                            continue

                        if lang != "vi":
                            continue

                        dataset.append({
                            "platform": "tiktok",
                            "keyword": keyword,
                            "video_id": video.id,
                            "comment": text,
                        })

                        print(f"[VI] {text}")
                        c_count += 1

                    count += 1

                except Exception as e:
                    print(f"Error when fetching comments for {video.id}: {e}")
                    continue

            if count >= VIDEO_LIMIT:
                break

            print("Video:", video.id)

            try:

                comments = video.comments(count=COMMENT_LIMIT)

                c_count = 0

                async for comment in comments:

                    if c_count >= COMMENT_LIMIT:
                        break

                    text = comment.text

                    dataset.append({
                        "platform":"tiktok",
                        "video_id":video.id,
                        "comment":text
                    })

                    print(text)

                    c_count += 1

                count += 1

            except:
                pass


asyncio.run(main())

df = pd.DataFrame(dataset)

df.to_csv(
    "tiktok_comments_dataset.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Total comments:", len(df))