import time
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager


EMAIL = "trinhthanhtama114ieltsbmt@gmail.com"
PASSWORD = "Trinhthanhtam12*1"

# Pages cần crawl
PAGES = [
    "https://www.facebook.com/spiderumcom",
    "https://www.facebook.com/beatvn",
    "https://www.facebook.com/vnexpress"
]

POST_LIMIT = 50
SCROLL_COMMENT = 15


# Chrome options (tránh bị Facebook phát hiện bot)
options = Options()
options.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

wait = WebDriverWait(driver,20)

driver.get("https://www.facebook.com/login")


# LOGIN
email_box = wait.until(
    EC.presence_of_element_located((By.NAME,"email"))
)

pass_box = wait.until(
    EC.presence_of_element_located((By.NAME,"pass"))
)

email_box.send_keys(EMAIL)
pass_box.send_keys(PASSWORD)

driver.find_element(By.NAME,"login").click()

print("Logging in...")

time.sleep(10)


dataset = []


for page in PAGES:

    print("Opening page:",page)

    driver.get(page)

    time.sleep(5)


    posts = driver.find_elements(By.XPATH,'//a[contains(@href,"/posts/")]')

    post_links = []

    for p in posts:

        link = p.get_attribute("href")

        if link and link not in post_links:

            post_links.append(link)

        if len(post_links) >= POST_LIMIT:

            break


    print("Found posts:",len(post_links))


    for post in post_links:

        print("Opening post:",post)

        driver.get(post)

        time.sleep(5)


        # Scroll để load comment
        for i in range(SCROLL_COMMENT):

            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )

            time.sleep(2)


        comments = driver.find_elements(
            By.XPATH,
            '//div[@data-ad-preview="message"]'
        )


        for c in comments:

            text = c.text

            if len(text) > 3:

                dataset.append({
                    "platform":"facebook",
                    "post":post,
                    "comment":text
                })

                print(text)


df = pd.DataFrame(dataset)

df.to_csv(
    "facebook_comments_dataset.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Total comments:",len(df))

driver.quit()