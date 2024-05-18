```python
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import time

def is_scraping_allowed(url):
    """
    指定されたURLに対してスクレイピングが許可されているかを確認する関数。
    
    ステージ 1: robots.txt の確認
    ステージ 2: 利用規約およびプライバシーポリシーの確認

    Parameters
    ----------
    url : str
        チェック対象のURL。

    Returns
    -------
    bool
        スクレイピングが許可されている場合はTrue、そうでない場合はFalseを返す。
    """
    # ステージ 1: robots.txt の確認
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    
    try:
        # robots.txt にアクセスして内容を取得
        response = requests.get(robots_url)
        if response.status_code == 200:
            print(f"Accessing robots.txt at {robots_url}")
            lines = response.text.splitlines()
            user_agent_allows = False
            for line in lines:
                line = line.lower().strip()  # 小文字に変換してトリム
                # User-agent: * のセクションを確認
                if line.startswith("user-agent: *"):
                    user_agent_allows = True
                # Disallow のパスをチェック
                if user_agent_allows and line.startswith("disallow:"):
                    disallow_path = line.split(":")[1].strip()
                    if url.startswith(f"{parsed_url.scheme}://{parsed_url.netloc}{disallow_path}"):
                        print(f"Scraping disallowed for path: {disallow_path}")
                        return False
            print("Scraping allowed according to robots.txt")
            return True
        else:
            # ステージ 2: robots.txt が存在しない場合、利用規約とプライバシーポリシーを確認
            print(f"No robots.txt found at {robots_url}, checking additional pages")
            tos_url = urljoin(f"{parsed_url.scheme}://{parsed_url.netloc}", "terms-of-service")
            privacy_url = urljoin(f"{parsed_url.scheme}://{parsed_url.netloc}", "privacy-policy")
            tos_allowed = check_additional_pages(tos_url)
            privacy_allowed = check_additional_pages(privacy_url)
            return tos_allowed and privacy_allowed
    except requests.RequestException as e:
        # 例外が発生した場合、スクレイピングが許可されていないとみなす
        print(f"Error accessing robots.txt: {e}")
        return False

def check_additional_pages(url):
    """
    指定されたURLに対してスクレイピングに関する記述があるかを確認する関数。

    Parameters
    ----------
    url : str
        チェック対象のURL。

    Returns
    -------
    bool
        スクレイピングが許可されている場合はTrue、そうでない場合はFalseを返す。
    """
    try:
        # 利用規約またはプライバシーポリシーページにアクセスして内容を取得
        response = requests.get(url)
        if response.status_code == 200:
            content = response.text.lower()
            # 特定のキーワードを含む場合はスクレイピングが禁止されていると判断
            if any(keyword in content for keyword in ["scraping", "crawl", "bot"]):
                print(f"Scraping disallowed according to content at {url}")
                return False
        return True
    except requests.RequestException as e:
        # 例外が発生した場合、ページが存在しないとみなす
        print(f"Error accessing page {url}: {e}")
        return True

def fetch_text_from_url(url_link, retries=5):
    """
    指定されたURLからテキストコンテンツを取得する関数。
    
    Parameters
    ----------
    url_link : str
        取得対象のURL。
    retries : int, optional
        リトライ回数 (default is 5)

    Returns
    -------
    str
        取得したテキストコンテンツ、またはエラーメッセージ。
    """
    try:
        response = requests.get(url_link, allow_redirects=True, timeout=10)
        if response.status_code != 200:
            if retries > 0:
                time.sleep(5)  # 5秒待機
                print(f"Retrying... attempts left: {retries}")
                return fetch_text_from_url(url_link, retries - 1)  # 再帰
            else:
                return 'Error: Failed to retrieve the content after multiple attempts'
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        paragraphs = soup.find_all('p')
        list_items = soup.find_all('li')
        
        text = ' '.join([para.get_text() for para in paragraphs + list_items])
        return text

    except requests.exceptions.TooManyRedirects:
        print("Too many redirects encountered.")
        return 'Error: Too many redirects'
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return f'Error: {e}'



```


```python
# スクレイピング禁止使用例
url = 'https://www.amazon.co.jp/%E3%80%8E%E3%82%B4%E3%82%B8%E3%83%A9-1-0%EF%BC%8FC%E3%80%8F-DVD-%E5%B1%B1%E5%B4%8E%E8%B2%B4/dp/B0CX171MJV/ref=tmm_dvd_swatch_0?_encoding=UTF8&qid=&sr='

if is_scraping_allowed(url):
    content = fetch_text_from_url(url)
    if content.startswith('Error'):
        print(content)
    else:
        print(content)
else:
    print("スクレイピングは許可されていません。")
```

    Accessing robots.txt at https://www.amazon.co.jp/robots.txt
    Scraping disallowed for path: /
    スクレイピングは許可されていません。



```python
# スクレイピング禁止使用例
url = 'https://x.com/elonmusk'

if is_scraping_allowed(url):
    content = fetch_text_from_url(url)
    if content.startswith('Error'):
        print(content)
    else:
        print(content)
else:
    print("スクレイピングは許可されていません。")
```

    Accessing robots.txt at https://x.com/robots.txt
    Scraping disallowed for path: /
    スクレイピングは許可されていません。



```python
# スクレイピングしない例 利用規約とプライバシーポリシーを確認
url = 'https://www.kaggle.com/'

if is_scraping_allowed(url):
    content = fetch_text_from_url(url)
    if content.startswith('Error'):
        print(content.startswith('Error'))
        print(content)
    else:
        print(content[:160])
else:
    print("スクレイピングは許可されていません。")
```

    No robots.txt found at https://www.kaggle.com/robots.txt, checking additional pages
    



```python
# スクレイピング使用可能例
url = 'https://ja.wikipedia.org/wiki/Twitter'

if is_scraping_allowed(url):
    content = fetch_text_from_url(url)
    if content.startswith('Error'):
        print(content)
    else:
        print(content[:160])
else:
    print("スクレイピングは許可されていません。")
```

    Accessing robots.txt at https://ja.wikipedia.org/robots.txt
    Scraping allowed according to robots.txt
    Twitter（ツイッター）、現在のX（エックス）は、アメリカ合衆国のX社が運営するソーシャルメディア、ソーシャル・ネットワーキング・サービス[10][11][12]。2023年7月24日に「X」へ名称変更した。投稿は、Twitterでは「ツイート」、Xでは「ポスト」と呼ばれ、限られた文字数だけで投稿できる[注釈 1]



```python
check_url = ['https://www.amazon.co.jp/%E3%80%8E%E3%82%B4%E3%82%B8%E3%83%A9-1-0%EF%BC%8FC%E3%80%8F-DVD-%E5%B1%B1%E5%B4%8E%E8%B2%B4/dp/B0CX171MJV/ref=tmm_dvd_swatch_0?_encoding=UTF8&qid=&sr=',
            'https://x.com/elonmusk',
            'https://www.kaggle.com/',
            'https://ja.wikipedia.org/wiki/Twitter']

for url in check_url:
    if is_scraping_allowed(url):
        content = fetch_text_from_url(url)
        if content.startswith('Error'):
            print(content.startswith('Error'))
            print(content)
        else:
            print(content[:160])
    else:
        print("スクレイピングは許可されていません。")
```

    Accessing robots.txt at https://www.amazon.co.jp/robots.txt
    Scraping disallowed for path: /
    スクレイピングは許可されていません。
    Accessing robots.txt at https://x.com/robots.txt
    Scraping disallowed for path: /
    スクレイピングは許可されていません。
    No robots.txt found at https://www.kaggle.com/robots.txt, checking additional pages
    
    Accessing robots.txt at https://ja.wikipedia.org/robots.txt
    Scraping allowed according to robots.txt
    Twitter（ツイッター）、現在のX（エックス）は、アメリカ合衆国のX社が運営するソーシャルメディア、ソーシャル・ネットワーキング・サービス[10][11][12]。2023年7月24日に「X」へ名称変更した。投稿は、Twitterでは「ツイート」、Xでは「ポスト」と呼ばれ、限られた文字数だけで投稿できる[注釈 1]



```python

```
