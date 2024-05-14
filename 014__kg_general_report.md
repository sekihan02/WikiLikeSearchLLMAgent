- Refarence
    - [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/pdf/2402.14207)
    - [Graph Chain-of-Thought: Augmenting Large Language Models by Reasoning on Graphs](https://arxiv.org/pdf/2404.07103)
    - [A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts](https://read-agent.github.io/)


```python
!pip install arxiv==2.1.0
!pip install python-dotenv tiktoken
!pip install openai==1.30.0

!pip install wikipedia==1.4.0
```

    Requirement already satisfied: arxiv==2.1.0 in /usr/local/lib/python3.10/dist-packages (2.1.0)
    Requirement already satisfied: feedparser==6.0.10 in /usr/local/lib/python3.10/dist-packages (from arxiv==2.1.0) (6.0.10)
    Requirement already satisfied: requests==2.31.0 in /usr/local/lib/python3.10/dist-packages (from arxiv==2.1.0) (2.31.0)
    Requirement already satisfied: sgmllib3k in /usr/local/lib/python3.10/dist-packages (from feedparser==6.0.10->arxiv==2.1.0) (1.0.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests==2.31.0->arxiv==2.1.0) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests==2.31.0->arxiv==2.1.0) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests==2.31.0->arxiv==2.1.0) (2.2.1)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests==2.31.0->arxiv==2.1.0) (2024.2.2)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mRequirement already satisfied: python-dotenv in /usr/local/lib/python3.10/dist-packages (1.0.1)
    Requirement already satisfied: tiktoken in /usr/local/lib/python3.10/dist-packages (0.7.0)
    Requirement already satisfied: regex>=2022.1.18 in /usr/local/lib/python3.10/dist-packages (from tiktoken) (2023.12.25)
    Requirement already satisfied: requests>=2.26.0 in /usr/local/lib/python3.10/dist-packages (from tiktoken) (2.31.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (2.2.1)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (2024.2.2)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mRequirement already satisfied: openai==1.30.0 in /usr/local/lib/python3.10/dist-packages (1.30.0)
    Requirement already satisfied: anyio<5,>=3.5.0 in /usr/local/lib/python3.10/dist-packages (from openai==1.30.0) (4.3.0)
    Requirement already satisfied: distro<2,>=1.7.0 in /usr/lib/python3/dist-packages (from openai==1.30.0) (1.7.0)
    Requirement already satisfied: httpx<1,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from openai==1.30.0) (0.27.0)
    Requirement already satisfied: pydantic<3,>=1.9.0 in /usr/local/lib/python3.10/dist-packages (from openai==1.30.0) (1.10.14)
    Requirement already satisfied: sniffio in /usr/local/lib/python3.10/dist-packages (from openai==1.30.0) (1.3.1)
    Requirement already satisfied: tqdm>4 in /usr/local/lib/python3.10/dist-packages (from openai==1.30.0) (4.66.2)
    Requirement already satisfied: typing-extensions<5,>=4.7 in /usr/local/lib/python3.10/dist-packages (from openai==1.30.0) (4.10.0)
    Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.10/dist-packages (from anyio<5,>=3.5.0->openai==1.30.0) (3.6)
    Requirement already satisfied: exceptiongroup>=1.0.2 in /usr/local/lib/python3.10/dist-packages (from anyio<5,>=3.5.0->openai==1.30.0) (1.2.0)
    Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from httpx<1,>=0.23.0->openai==1.30.0) (2024.2.2)
    Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.10/dist-packages (from httpx<1,>=0.23.0->openai==1.30.0) (1.0.5)
    Requirement already satisfied: h11<0.15,>=0.13 in /usr/local/lib/python3.10/dist-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai==1.30.0) (0.14.0)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mRequirement already satisfied: wikipedia==1.4.0 in /usr/local/lib/python3.10/dist-packages (1.4.0)
    Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from wikipedia==1.4.0) (4.12.3)
    Requirement already satisfied: requests<3.0.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from wikipedia==1.4.0) (2.31.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3.0.0,>=2.0.0->wikipedia==1.4.0) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3.0.0,>=2.0.0->wikipedia==1.4.0) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3.0.0,>=2.0.0->wikipedia==1.4.0) (2.2.1)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3.0.0,>=2.0.0->wikipedia==1.4.0) (2024.2.2)
    Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->wikipedia==1.4.0) (2.5)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m


```python
import re
import os
import json
import warnings

import arxiv
import openai
from openai import OpenAI
from dotenv import load_dotenv
import wikipedia

# すべての警告を無視する
warnings.filterwarnings('ignore')
```


```python
def get_wikipedia_articles_for_keywords(keywords, num_articles=3, lang='ja'):
    """
    与えられたキーワードのリストに対し、各キーワードについてWikipedia記事を検索し、記事の情報を取得する。

    Parameters
    ----------
    keywords : list of str
        検索するキーワードのリスト
    num_articles : int, optional
        各キーワードに対して取得する記事の数 (default is 3)
    lang : str, optional
        使用する言語 (default is 'ja' for Japanese)

    Returns
    -------
    all_articles : list of dict
        各キーワードについて取得した記事の情報を含む辞書のリスト。
        各辞書はキーワード、タイトル、URL、記事の全文を含む。
    -------
    articles = get_wikipedia_articles_for_keywords(keywords)
    for article in articles:
        print('キーワード: ', article['keyword'])
        print('タイトル: ', article['title'])
        print('URL: ', article['url'])
        print('内容: ', article['content'])
        print('\n')
    """
    
    wikipedia.set_lang(lang)  # 言語を設定
    all_articles = []  # 全記事情報を保持するリスト

    try:
        titles = wikipedia.search(keywords, results=num_articles)  # キーワードでWikipediaを検索
        articles = []

        for title in titles:  # 取得した各タイトルに対して
            page = wikipedia.page(title)  # ページ情報を取得
            articles.append({  # 記事情報を辞書として追加
                'keyword': keywords,  # 検索キーワード
                'title': title,  # 記事のタイトル
                'url': page.url,  # 記事のURL
                'summary': page.summary,  # 記事の概要
                # 'summary': wikipedia.summary(title),  # 記事の概要
                'content': page.content  # 記事の全文
            })
        all_articles.extend(articles)  # 全記事情報リストに追加
    except wikipedia.DisambiguationError as e:  # 曖昧さ回避ページがヒットした場合のエラーハンドリング
        print(f"DisambiguationError for keyword {keywords}: {e.options}")  # エラーメッセージを出力
        
    return all_articles  # 全記事情報を返す

```


```python
load_dotenv()
```




    True




```python
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-3.5-turbo-0125"
# MODEL_NAME = "gpt-3.5-turbo-instruct"
# MODEL_NAME = "gpt-4-0125-preview"
# MODEL_NAME = "gpt-4-turbo-2024-04-09"

MODEL4o_NAME = "gpt-4o-2024-05-13"

TEMPERATURE = 0.7
# OpenAIクライアントの初期化
client = OpenAI()
```


```python
message = "ディアドコイ戦争や後継者戦争とも言われる\nディアドコイ戦争について教えてください"
```


```python
def generate_search_queries(model_name, text, count):
    # テキストからWikipedia検索クエリを生成する
    prompt = [
        {"role": "system", "content": "You want to answer the question using search . What do you type in the search box ?"},
        {"role": "system", "content": f"Please formulate {count} distinct search queries based on the content of the Input text."},
        {"role": "system", "content": "Please ensure that the output is in English."},
        {"role": "system", "content": "Write the queries you will use in the following"},
        {"role": "system", "content": "format :\n query 1\n query 2\n..."},
        {"role": "user", "content": f"Input text: {text}"},
        {"role": "user", "content": "format :\n"}
    ]
    
    # 概要と提案手法名抽出用のプロンプトテンプレートを作成
    response = client.chat.completions.create(
        model=model_name, # model = "deployment_name".
        messages=prompt,
        temperature=TEMPERATURE,
    )
    
    search_queries = response.choices[0].message.content
    return search_queries
```


```python
search_query = generate_search_queries(MODEL_NAME, message, "one")
search_query
```




    'Diadochi Wars timeline'




```python
keywords_list = search_query.split('\n')
articles_overview = []
for keyword in keywords_list:
    article = get_wikipedia_articles_for_keywords(keyword, num_articles=1, lang='en')  # ここでは英語で検索しています
    articles_overview.append(article)
```


```python
# 結果を出力
for article in articles_overview:
    article = article[0]
    print('キーワード:', article['keyword'])
    print('タイトル:', article['title'])
    print('URL:', article['url'])
    print('概要:', article['summary'])
    print('\n')
```

    キーワード: Diadochi Wars timeline
    タイトル: Wars of the Diadochi
    URL: https://en.wikipedia.org/wiki/Wars_of_the_Diadochi
    概要: The Wars of the Diadochi (Ancient Greek: Πόλεμοι τῶν Διαδόχων Pólemoi tōn Diadóchōn, literally War of the Crown Princes), or Wars of Alexander's Successors, were a series of conflicts fought between the generals of Alexander the Great, known as the Diadochi, over who would rule his empire following his death. The fighting occurred between 322 and 281 BC.
    
    



```python
def generate_wiki_questions(model_name, summary_text):
    # Wikipediaの概要から質問を生成する関数
    prompt = [
        {"role": "system", "content": "You are an experienced Wikipedia writer and want to edit a specific page."},
        {"role": "system", "content": "Besides your identity as a Wikipedia writer, you have a specific focus when researching the topic."},
        {"role": "system", "content": "Now, you are chatting with an expert to get information. Ask good questions to get more useful information."},
        {"role": "system", "content": "When you have no more question to ask, say 'Thank you so much for your help!' to end the conversation."},
        {"role": "system", "content": "Please only ask one question at a time and don't ask what you have asked before."},
        {"role": "system", "content": "Your questions should be related to the topic you want to write."},
        {"role": "user", "content": f"Input summary: {summary_text}"},
        {"role": "system", "content": "Please generate three distinct questions based on the input summary."},
        {"role": "system", "content": "The questions should be formatted as:\nQuestion 1: [Your question here]\nQuestion 2: [Your question here]\nQuestion 3: [Your question here]"}
    ]

    
    # 会話を開始するためにチャットAPIを使用する
    response = client.chat.completions.create(
        model=model_name, # モデル名を指定
        messages=prompt,
        temperature=TEMPERATURE,
    )
    
    questions = response.choices[0].message.content
    
    return questions


```


```python
# 記事から生成された質問を格納するリストを初期化
all_questions = []

# 各記事の要約に対して質問を生成
for article in articles_overview:
    article = article[0]
    questions = generate_wiki_questions(MODEL_NAME, article['summary'])
    all_questions.append({
        'title': article['title'],
        'questions': questions
    })
```


```python

# 生成された質問のリストを出力
for item in all_questions:
    print(f"Title: {item['title']}")
    print(f"questions: {item['questions']}")
    # for question in item['questions']:
    #     print(question)
    print('\n')  # 質問の間に空行を挿入

```

    Title: Wars of the Diadochi
    questions: Question 1: What were the main reasons behind the Wars of the Diadochi, also known as the Wars of Alexander's Successors?
    Question 2: Can you explain the significance of the conflicts fought between the Diadochi, the generals of Alexander the Great, in determining the ruler of his empire after his death?
    Question 3: How long did the Wars of the Diadochi last, and what were the major events that took place during this period from 322 to 281 BC?
    
    



```python
import requests

# APIキー設定
api_key = os.getenv("BING_API_KEY")
# APIリクエストを送信
url = 'https://api.bing.microsoft.com/v7.0/search'
```


```python
all_questions
```




    [{'title': 'Wars of the Diadochi',
      'questions': "Question 1: What were the main reasons behind the Wars of the Diadochi, also known as the Wars of Alexander's Successors?\nQuestion 2: Can you explain the significance of the conflicts fought between the Diadochi, the generals of Alexander the Great, in determining the ruler of his empire after his death?\nQuestion 3: How long did the Wars of the Diadochi last, and what were the major events that took place during this period from 322 to 281 BC?"}]




```python
all_questions[0]['questions'].split("\n")[0].split(": ")[-1]
```




    "What were the main reasons behind the Wars of the Diadochi, also known as the Wars of Alexander's Successors?"




```python
all_questions[0]['questions'].split("\n")
```




    ["Question 1: What were the main reasons behind the Wars of the Diadochi, also known as the Wars of Alexander's Successors?",
     'Question 2: Can you explain the significance of the conflicts fought between the Diadochi, the generals of Alexander the Great, in determining the ruler of his empire after his death?',
     'Question 3: How long did the Wars of the Diadochi last, and what were the major events that took place during this period from 322 to 281 BC?']




```python
item['questions']
```




    "Question 1: What were the main reasons behind the Wars of the Diadochi, also known as the Wars of Alexander's Successors?\nQuestion 2: Can you explain the significance of the conflicts fought between the Diadochi, the generals of Alexander the Great, in determining the ruler of his empire after his death?\nQuestion 3: How long did the Wars of the Diadochi last, and what were the major events that took place during this period from 322 to 281 BC?"




```python
# 前はほとんど一言しか回答せず、せっかく全文検索しているメリットがないので
# 少し説明部分も抽出する形で一まとめに要約するように生成
def generate_search_content_summary(model_name, search_quary, content):
    
    prompt = [
        {"role": "system", "content": f"Please extract and summarize the following sentences with keywords and the parts that explain the keywords."},
        {"role": "user", "content": f"keyword: {search_quary}"},
        {"role": "user", "content": f"Sentence: {content}"},
        {"role": "user", "content": "Shortened sentences:"},
    ]
    
    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=TEMPERATURE,
    )
    
    detailed_outline = response.choices[0].message.content
    
    return detailed_outline

```


```python
import requests
from bs4 import BeautifulSoup
import time

def fetch_text_from_url(url_link, retries=5):
    try:
        response = requests.get(url_link, allow_redirects=True, timeout=10)
        # レスポンスのステータスコードが200以外の場合はエラーを表示して処理を終了
        if response.status_code != 200:
            if retries > 0:
                time.sleep(5)  # 5秒待機
                print(f"retries: {retries}")
                return fetch_text_from_url(url_link, retries - 1)  # 再帰
            else:
                return 'Error: Failed to retrieve the content after multiple attempts'
        
        # HTMLの解析
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # <p> と <li> タグのテキストを取得する
        paragraphs = soup.find_all('p')
        list_items = soup.find_all('li')
        
        # <p> と <li> のテキストを結合する
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
# 生成された質問のリストごとに検索を実施
articles_q = []

# 空白の要素を除外する
questions = all_questions[0]['questions'].split("\n")
filtered_questions = [question for question in questions if question.strip()]

for question in filtered_questions:
    search_query = generate_search_queries(MODEL_NAME, question.split(": ")[-1], "one")
    
    # Wikipediaを除外する検索クエリの追加
    search_query += " -site:wikipedia.org"

    params = {'q': search_query, 'mkt': 'en', 'count': 3}
    # params = {'q': search_query, 'mkt': 'en', 'count': 1}
    headers = {'Ocp-Apim-Subscription-Key': api_key}
    r = requests.get(url, headers=headers, params=params)

    # 検索結果を取得
    results = r.json()['webPages']['value']
    # 結果を連結して回答を生成
    
    links = []
    for result in results:
        links.append(result['url'])
        
    url_text = ""
    for search_link in links:
        ur_fetch = fetch_text_from_url(search_link)
        if ur_fetch == 'Error: Failed to retrieve the content after multiple attempts':
            continue
        else:
            url_text += fetch_text_from_url(search_link)
    
    short_ans = generate_search_content_summary(MODEL4o_NAME, search_query, url_text)
    
    articles_q.append({
        'questions': question,
        'answer' : short_ans,
        'links' : links,
    })
```


```python
# 結果を出力
for article in articles_q:
    print('questions:', article['questions'])
    print('answer:', article['answer'])
    print('links:', article['links'])
    print('\n')
```

    questions: Question 1: What were the main reasons behind the Wars of the Diadochi, also known as the Wars of Alexander's Successors?
    answer: 1. **Keywords: Alexander the Great's Death, Leadership Void**
       - *On June 10, 323 BCE, Alexander the Great died in Babylon, leaving his empire without clear leadership.*
    
    2. **Keywords: Military Commanders, Territorial Conflict**
       - *Alexander's military commanders fought over territories, leading to the Wars of Succession or Wars of the Diadochi.*
    
    3. **Keywords: Rivalry, Dynasties**
       - *Over three decades of intense rivalry ended with the establishment of three dynasties that lasted until Roman times.*
    
    4. **Keywords: Alexander’s Exile Decree, Rebellion**
       - *Alexander’s proposed Exile Decree aimed at returning Greek exiles caused unrest, with many troops protesting and some satraps being executed for treason.*
    
    5. **Keywords: Revolts, Lamian War**
       - *After Alexander's death, Athens and Aetolia revolted, leading to the Lamian War, which ended with intervention by Antipater and Craterus.*
    
    6. **Keywords: Alexander’s Successor, Perdiccas**
       - *On his deathbed, Alexander handed his signet ring to Perdiccas, leading to debates over his final words and the choice of a successor.*
    
    7. **Keywords: Succession Dispute, Arrhidaeus, Alexander IV**
       - *The succession struggle centered around Alexander's half-brother Arrhidaeus and his unborn son Alexander IV, with commanders divided over support.*
    
    8. **Keywords: Perdiccas, Roxanne, Assassination**
       - *Perdiccas supported Roxanne and Alexander IV but faced betrayal, leading Roxanne to kill Alexander's other wife and her sister.*
    
    9. **Keywords: Babylon Partition, Territorial Division**
       - *The Partition of Babylon divided Alexander's empire among key commanders, including Antipater, Craterus, Ptolemy, Lysimachus, Eumenes, and Antigonus.*
    
    10. **Keywords: Successor Wars, Antigonus, Seleucus, Ptolemy**
        - *The Successor Wars involved major figures like Antigonus, Seleucus, and Ptolemy, resulting in shifting alliances and territorial control.*
    
    11. **Keywords: Perdiccas vs. Ptolemy, Nile Crossing**
        - *Perdiccas attempted to invade Egypt but failed, leading to his assassination by his own troops, possibly with Seleucus's involvement.*
    
    12. **Keywords: Treaty of Triparadeisus, New Regents**
        - *The Treaty of Triparadeisus in 321 BCE reallocated territories, with Antipater replacing Perdiccas as regent.*
    
    13. **Keywords: Second and Third Successor Wars, Cassander, Polyperchon**
        - *The Second and Third Successor Wars saw conflicts between Cassander and Polyperchon, with shifting alliances and territorial gains.*
    
    14. **Keywords: Battle of Ipsus, Antigonus’s Death**
        - *The Battle of Ipsus in 301 BCE resulted in Antigonus's death and the final division of his territories among other commanders.*
    
    15. **Keywords: Antigonids, Ptolemies, Seleucids**
        - *The wars led to the establishment of the Antigonid, Ptolemaic, and Seleucid dynasties, which lasted until the Roman period.*
    
    16. **Keywords: Seleucus, Assassination**
        - *Seleucus, who had gained significant territory, was assassinated by Ptolemy's son in 281 BCE.*
    
    17. **Keywords: Persistent Conflict, Alexander's Legacy**
        - *The wars that ensued from Alexander's death created lasting dynasties but never reunited the empire Alexander had built.*
    links: ['https://www.worldhistory.org/Wars_of_the_Diadochi/', 'https://www.thecollector.com/who-were-the-diadochi-of-alexander-the-great/', 'https://www.livius.org/articles/concept/diadochi/']
    
    
    questions: Question 2: Can you explain the significance of the conflicts fought between the Diadochi, the generals of Alexander the Great, in determining the ruler of his empire after his death?
    answer: 1. **Alexander the Great's Diadochi conflicts**: Successors fought over his vast empire from Greece to India in bloody conflicts, shaping the Hellenistic World.
    2. **Significance of Diadochi age**: One of the bloodiest periods in Greek history, marked by intrigue, treachery, and bloodshed.
    3. **Empire ruler after death**: Alexander died on June 11, 323 BCE, in Babylon; his generals asked who would rule, and he said, “to the strongest.”
    4. **Empire's vastness**: Alexander's empire spanned from the Adriatic Sea to the Indus River and from Libya to modern-day Tajikistan.
    5. **Sudden death impact**: Alexander's sudden death led to no clear successor, causing shock and instability in the empire.
    6. **Diadochi wars**: From 323 to 281 BCE, Macedonian generals fought bloody wars known as the Diadochi wars.
    7. **Initial succession plan**: Generals agreed the successor would be Alexander and Roxana’s unborn child (if male) or his brother-in-law, Philip III.
    8. **Perdiccas' role after death**: Perdiccas became the empire’s regent until Alexander IV could rule, enjoying legitimacy as Alexander gave him his ring before dying.
    9. **Key figures**: Ptolemy, Antigonus, Antipater, Seleucus, and Lysimachus proved resilient in administrative roles.
    10. **Power shift after Perdiccas**: Perdiccas was murdered in 321 BCE, with Ptolemy securing Egypt and Alexander’s body, gaining prestige.
    11. **Triparadisus partition**: In 321 BCE, the empire was partitioned among the Diadochi, still united under Alexander IV and Philip III.
    12. **Antigonus' dominance**: From 320-301 BCE, Antigonus sought to reunite Alexander's empire, becoming the most formidable power.
    13. **End of Alexander’s bloodline**: Cassander assassinated Alexander IV in 311 BCE, solidifying the division into four kingdoms.
    14. **Battle of Ipsos**: In 301 BCE, the allied forces defeated Antigonus, leading to the final division between Europe and Asia.
    15. **Lysimachus’ and Seleucus’ fates**: Lysimachus expanded and was later killed by Seleucus in 281 BCE, who was then assassinated by Ptolemy Keraunos.
    16. **Antigonus II Gonatas’ rise**: Took advantage of the chaos to become king of Thessaly and Macedonia in 276 BCE.
    17. **End of Diadochi wars**: The Hellenistic World stabilized until Roman conquest, with the Antigonids ruling Macedonia, the Ptolemies Egypt, and the Seleucids Syria, Mesopotamia, and Iran.
    18. **Ptolemy I Soter’s achievements**: Secured Egypt, constructed Alexandria's tomb, expanded his realm, and established the library and museum of Alexandria.
    19. **Seleucus’ rise**: Gained control of Babylon, expanded his territory, and established cities like Antioch and Seleucia.
    20. **Antigonus’ ambitions**: Tried to reunite Alexander's empire, fought multiple Diadochi, and was defeated at Ipsos in 301 BCE.
    21. **Cassander’s ruthlessness**: Murdered Alexander’s family members, secured Macedonia, and founded cities like Thessalonica.
    22. **Lysimachus’ rule**: Controlled Thrace and parts of Asia Minor, was killed by Seleucus in 281 BCE.
    23. **Alexander’s final words**: “To the best,” indicating the strongest should succeed him, leading to rivalry and wars among his generals.
    links: ['https://www.thecollector.com/who-were-the-diadochi-of-alexander-the-great/', 'https://www.worldhistory.org/Wars_of_the_Diadochi/', 'https://www.ancient-origins.net/history-important-events/diadochi-0016823']
    
    
    questions: Question 3: How long did the Wars of the Diadochi last, and what were the major events that took place during this period from 322 to 281 BC?
    answer: 1. **Death of Alexander the Great (323 BCE)**: Alexander the Great died in Babylon, leaving no clear successor, leading to the Wars of the Diadochi.
    
    2. **Lamian War (323-322 BCE)**: Athens and Aetolia rebelled upon hearing of Alexander's death, ending with the Battle at Crannon where Antipater and Craterus intervened.
    
    3. **Partition of Babylon**: Alexander's kingdom was divided among his commanders: Antipater and Craterus received Macedon and Greece, Ptolemy took Egypt, Lysimachus got Thrace, Eumenes received Cappadocia, and Antigonus retained Phrygia.
    
    4. **First Successor War (322-320 BCE)**: Conflict arose over territorial disputes, leading to the death of Perdiccas and the Treaty of Triparadeisus, securing territories for the commanders.
    
    5. **Second Successor War (319-315 BCE)**: Cassander and Polyperchon clashed over Macedon and Greece, with Cassander eventually establishing control with Antigonus's help.
    
    6. **Third Successor War (314-311 BCE)**: Antigonus and Eumenes fought for control, culminating in Eumenes's betrayal and execution.
    
    7. **Babylonian War (311-309 BCE)**: Seleucus, with Ptolemy's support, regained Babylonia from Antigonus and his son Demetrius.
    
    8. **Fourth Successor War (308-301 BCE)**: Cassander, Ptolemy, Lysimachus, and Seleucus allied against Antigonus and Demetrius, resulting in the Battle of Ipsus and the death of Antigonus.
    
    9. **Death of Lysimachus (281 BCE)**: Lysimachus was defeated by Seleucus at Corupedium, marking the end of the major conflicts among Alexander's successors.
    links: ['https://www.worldhistory.org/Wars_of_the_Diadochi/', 'https://www.worldhistory.org/timeline/Wars_of_the_Diadochi/', 'https://history-maps.com/story/Seleucid-Empire/event/Wars-of-the-Diadochi']
    
    



```python
# 関連する検索結果のまとめ
related_search_results = ""
related_search_links = []
for article in articles_q:
    related_search_results += article['answer']
    for link in article['links']:
        related_search_links.append(link)

```


```python
articles_overview[0][0]['summary']
```




    "The Wars of the Diadochi (Ancient Greek: Πόλεμοι τῶν Διαδόχων Pólemoi tōn Diadóchōn, literally War of the Crown Princes), or Wars of Alexander's Successors, were a series of conflicts fought between the generals of Alexander the Great, known as the Diadochi, over who would rule his empire following his death. The fighting occurred between 322 and 281 BC."




```python
articles_overview[0][0]['content']
```




    'The Wars of the Diadochi (Ancient Greek: Πόλεμοι τῶν Διαδόχων Pólemoi tōn Diadóchōn, literally War of the Crown Princes), or Wars of Alexander\'s Successors, were a series of conflicts fought between the generals of Alexander the Great, known as the Diadochi, over who would rule his empire following his death. The fighting occurred between 322 and 281 BC.\n\n\n== Background ==\n\nAlexander the Great died on June 10, 323 BC, leaving behind an empire that stretched from Macedon and the rest of Greece in Europe to the Indus valley in South Asia. The empire had no clear successor, with the Argead family, at this point, consisting of Alexander\'s mentally disabled half-brother, Arrhidaeus; his unborn son Alexander IV; his reputed illegitimate son Heracles; his mother Olympias; his sister Cleopatra; and his half-sisters Thessalonike and Cynane.\nAlexander\'s death was the catalyst for the disagreements that ensued between his former generals resulting in a succession crisis. Two main factions formed after the death of Alexander. The first of these was led by Meleager, who supported the candidacy of Alexander\'s half-brother, Arrhidaeus. The second was led by Perdiccas, the leading cavalry commander, who believed it would be best to wait until the birth of Alexander\'s unborn child, by Roxana. Both parties agreed to a compromise, wherein Arrhidaeus would become king as Philip III and rule jointly with Roxana\'s child, providing it was a male heir. Perdiccas was designated as regent of the empire, with Meleager acting as his lieutenant. However, soon after, Perdiccas had Meleager and the other leaders who had opposed him murdered, and he assumed full control.\nThe  generals who had supported Perdiccas were rewarded in the partition of Babylon by becoming satraps of the various parts of the empire. Ptolemy received Egypt; Laomedon received Syria and Phoenicia; Philotas took Cilicia; Peithon took Media; Antigonus received Phrygia, Lycia and Pamphylia; Asander received Caria; Menander received Lydia; Lysimachus received Thrace; Leonnatus received Hellespontine Phrygia; and Neoptolemus had Armenia. Macedon and the rest of Greece were to be under the joint rule of Antipater, who had governed them for Alexander, and Craterus, a lieutenant of Alexander. Alexander\'s secretary, Eumenes of Cardia, was to receive Cappadocia and Paphlagonia.\nIn the east, Perdiccas largely left Alexander\'s arrangements intact – Taxiles and Porus ruled over their kingdoms in India; Alexander\'s father-in-law Oxyartes ruled Gandara; Sibyrtius ruled Arachosia and Gedrosia; Stasanor ruled Aria and Drangiana; Philip ruled Bactria and Sogdiana; Phrataphernes ruled Parthia and Hyrcania; Peucestas governed Persis; Tlepolemus had charge over Carmania; Atropates governed northern Media; Archon got Babylonia; and, Arcesilas ruled northern Mesopotamia.\n\n\n== Lamian War ==\n\nThe news of Alexander\'s death inspired a revolt in Greece, known as the Lamian War. Athens and other cities formed a coalition and besieged Antipater in the fortress of Lamia, however, Antipater was relieved by a force sent by Leonnatus, who was killed in battle. The Athenians were defeated at the Battle of Crannon on September 5, 322 BC by Craterus and his fleet.\nAt this time, Peithon suppressed a revolt of Greek settlers in the eastern parts of the empire, and Perdiccas and Eumenes subdued Cappadocia.\n\n\n== First War of the Diadochi, 321–319 BC ==\n\nPerdiccas, who was already betrothed to the daughter of Antipater, attempted to marry Alexander\'s sister, Cleopatra, a marriage which would have given him claim to the Macedonian throne. In 322 BC, Antipater, Craterus and Antigonus all formed a coalition against Perdiccas\'s growing power. Soon after, Antipater would send his army, under the command of Craterus, into Asia Minor. In late 322 or early 321 BC, Ptolemy stole Alexander\'s body on its way to Macedonia and then joined the coalition. A force under Eumenes defeated Craterus at the battle of the Hellespont, however, Perdiccas was soon after murdered by his own generals Peithon, Seleucus, and Antigenes during his invasion of Egypt, after a failed attempt to cross the Nile.\nPtolemy came to terms with Perdiccas\' murderers, making Peithon and Arrhidaeus regents in Perdiccas\'s place, but soon these came to a new agreement with Antipater at the Treaty of Triparadisus. Antipater was made Regent of the Empire, and the two kings were moved to Macedon. Antigonus was made Strategos of Asia and remained in charge of Phrygia, Lycia, and Pamphylia, to which was added Lycaonia. Ptolemy retained Egypt, Lysimachus retained Thrace, while the three murderers of Perdiccas—Seleucus, Peithon, and Antigenes—were given the provinces of Babylonia, Media, and Susiana respectively. Arrhidaeus, the former regent, received Hellespontine Phrygia. Antigonus was charged with the task of rooting out Perdiccas\'s former supporter, Eumenes. In effect, Antipater retained for himself control of Europe, while Antigonus, as Strategos of the East, held a similar position in Asia.\nAlthough the First War ended with the death of Perdiccas, his cause lived on. Eumenes was still at large with a victorious army in Asia Minor. So were Alcetas, Attalus, Dokimos and Polemon who had also gathered their armies in Asia Minor. In 319 BC Antigonus, after receiving reinforcements from Antipater\'s European army, first campaigned against Eumenes (see: battle of Orkynia), then against the combined forces of Alcetas, Attalus, Dokimos and Polemon (see: battle of Cretopolis), defeating them all.\n\n\n== Second War of the Diadochi, 318–316 BC ==\n\nAnother war soon broke out between the Diadochi. At the start of 318 BC Arrhidaios, the governor of Hellespontine Phrygia, tried to take the city of Cyzicus. Antigonus, as the Strategos of Asia, took this as a challenge to his authority and recalled his army from their winter quarters. He sent an army against Arrhidaios while he himself marched with the main army into Lydia against its governor Cleitus whom he drove out of his province.\nCleitus fled to Macedon and joined Polyperchon, the new Regent of the Empire, who decided to march his army south to force the Greek cities to side with him against Cassander and Antigonus. Cassander, reinforced with troops and a fleet by Antigonus, sailed to Athens and thwarted Polyperchon\'s efforts to take the city. From Athens Polyperchon marched on Megalopolis which had sided with Cassander and besieged the city. The siege failed and he had to retreat losing a lot of prestige and most of the Greek cities. Eventually Polyperchon retreated to Epirus with the infant King Alexander IV. There he joined forces with Alexander\'s mother Olympias and was able to re-invade Macedon. King Philip Arrhidaeus, Alexander\'s half-brother, having defected to Cassander\'s side at the prompting of his wife, Eurydice, was forced to flee, only to be captured in Amphipolis, resulting in the execution of himself and the forced suicide of his wife, both purportedly at the instigation of Olympias. Cassander rallied once more, and seized Macedon. Olympias was murdered, and Cassander gained control of the infant King and his mother. Eventually, Cassander became the dominant power in the European part of the Empire, ruling over Macedon and large parts of Greece.\nMeanwhile, Eumenes, who had gathered a small army in Cappadocia, had entered the coalition of Polyperchon and Olympias. He took his army to the royal treasury at Kyinda in Cilicia where he used its funds to recruit mercenaries. He also secured the loyalty of 6,000 of Alexander\'s veterans, the Argyraspides (the Silver Shields) and the Hypaspists, who were stationed in Cilicia. In the spring of 317 BC he marched his army to Phoenica and began to raise a naval force on the behalf of Polyperchon. Antigonus had spent the rest of 318 BC consolidating his position and gathering a fleet. He now used this fleet (under the command of Nicanor who had returned from Athens) against Polyperchon\'s fleet in the Hellespont. In a two-day battle near Byzantium, Nicanor and Antigonus destroyed Polyperchon\'s fleet. Then, after settling his affairs in western Asia Minor, Antigonus marched against Eumenes at the head of a great army. Eumenes hurried out of Phoenicia and marched his army east to gather support in the eastern provinces. In this he was successful, because most of the eastern satraps joined his cause (when he arrived in Susiana) more than doubling his army. They marched and counter-marched throughout Mesopotamia, Babylonia, Susiana and Media until they faced each other on a plain in the country of the Paraitakene in southern Media. There they fought a great battle −the battle of Paraitakene− which ended inconclusively. The next year (315) they fought another great but inconclusive battle −the battle of Gabiene− during which some of Antigonus\'s troops plundered the enemy camp. Using this plunder as a bargaining tool, Antigonus bribed the Argyraspides who arrested and handed over Eumenes. Antigonus had Eumenes and a couple of his officers executed. With Eumenes\'s death, the war in the eastern part of the Empire ended.\nAntigonus and Cassander had won the war. Antigonus now controlled Asia Minor and the eastern provinces, Cassander controlled Macedon and large parts of Greece, Lysimachus controlled Thrace, and Ptolemy controlled Egypt, Syria, Cyrene and Cyprus. Their enemies were either dead or seriously reduced in power and influence.\n\n\n== Third War of the Diadochi, 315–311 BC ==\nThough his authority had seemed secure with his victory over Eumenes, the eastern dynasts were unwilling to see Antigonus rule all of Asia. In 314 BC they demanded from Antigonus that he cede Lycia and Cappadocia to Cassander, Hellespontine Phrygia to Lysimachus, all of Syria to Ptolemy, and Babylonia to Seleucus, and that he share the treasures he had captured. Antigonus only answer was to advise them to be ready, then, for war. In this war, Antigonus faced an alliance of Ptolemy (with Seleucus serving him), Lysimachus, and Cassander. At the start of the campaigning season of 314 BC Antigonus invaded Syria and Phoenicia, which were under Ptolemy\'s control, and besieged Tyre. Cassander and Ptolemy started supporting Asander (satrap of Caria) against Antigonus who ruled the neighbouring provinces of Lycia, Lydia and Greater Phrygia. Antigonus then sent Aristodemus with 1,000 talents to the Peloponnese to raise a mercenary army to fight Cassander, he allied himself to Polyperchon, who still controlled parts of the Peloponnese, and he proclaimed freedom for the Greeks to get them on their side. He also sent his nephew Ptolemaios with an army through Cappadocia to the Hellespont to cut Asander off from Lysimachus and Cassander. Polemaios was successful, securing the northwest of Asia Minor for Antigonus, even invading Ionia/Lydia and bottling up Asander in Caria, but he was unable to drive his opponent from his satrapy.\nEventually Antigonus decided to campaign against Asander himself, leaving his oldest son Demetrius to protect Syria and Phoenica against Ptolemy. Ptolemy and Seleucus invaded from Egypt and defeated Demetrius in the Battle of Gaza. After the battle, Seleucus went east and secured control of Babylon (his old satrapy), and then went on to secure the eastern satrapies of Alexander\'s empire. Antigonus, having defeated Asander, sent his nephews Telesphorus and Polemaios to Greece to fight Cassander, he himself returned to Syria/Phoenica, drove off Ptolemy, and sent Demetrius east to take care of Seleucus. Although Antigonus now concluded a compromise peace with Ptolemy, Lysimachus, and Cassander, he continued the war with Seleucus, attempting to recover control of the eastern reaches of the empire. Although he went east himself in 310 BC, he was unable to defeat Seleucus (he even lost a battle to Seleucus) and had to give up the eastern satrapies.\nAt about the same time, Cassander had young King Alexander IV and his mother Roxane murdered, ending the Argead dynasty, which had ruled Macedon for several centuries. As Cassander did not publicly announce the deaths, all of the various generals continued to recognize the dead Alexander as king, however, it was clear that at some point, one or all of them would claim the kingship. At the end of the war there were five Diadochi left: Cassander ruling Macedon and Thessaly, Lysimachus ruling Thrace, Antigonus ruling Asia Minor, Syria and Phoenicia, Seleucus ruling the eastern provinces and Ptolemy ruling Egypt and Cyprus. Each of them ruled as kings (in all but name).\n\n\n== Babylonian War, 311–309 BC ==\n\nThe Babylonian War was a conflict fought between 311 and 309 BC between the Diadochi kings Antigonus I Monophthalmus and Seleucus I Nicator, ending in a victory for the latter, Seleucus I Nicator. The conflict ended any possibility of restoration of the empire of Alexander the Great, a result confirmed in the Battle of Ipsus.\n\n\n== Fourth War of the Diadochi, 307–301 BC ==\nPtolemy had been expanding his power into the Aegean and to Cyprus, while Seleucus went on a tour of the east to consolidate his control of the vast eastern territories of Alexander\'s empire. Antigonus resumed the war, sending his son Demetrius to regain control of Greece. In 307 he took Athens, expelling Demetrius of Phaleron, Cassander\'s governor, and proclaiming the city free again. Demetrius now turned his attention to Ptolemy, invading Cyprus and defeating Ptolemy\'s fleet at the Battle of Salamis. In the aftermath of this victory, Antigonus and Demetrius both assumed the crown, and they were shortly followed by Ptolemy, Seleucus, Lysimachus, and eventually Cassander.\nIn 306, Antigonus attempted to invade Egypt, but storms prevented Demetrius\' fleet from supplying him, and he was forced to return home. Now, with Cassander and Ptolemy both weakened, and Seleucus still occupied in the East, Antigonus and Demetrius turned their attention to Rhodes, which was besieged by Demetrius\'s forces in 305 BC. The island was reinforced by troops from Ptolemy, Lysimachus, and Cassander. Ultimately, the Rhodians reached a compromise with Demetrius – they would support Antigonus and Demetrius against all enemies, save their great ally Ptolemy. Ptolemy took the title of Soter ("Savior") for his role in preventing the fall of Rhodes, but the victory was ultimately Demetrius\'s, as it left him with a free hand to attack Cassander in Greece.\nAt the beginning of 304, Cassander managed to capture Salamis and besieged Athens. Athens petitioned Antigonus and Demetrius to come to their aid. Demetrius gathered a large fleet and landed his army in Boeotia in the rear of Cassander\'s forces. He freed the cities of Chalkis and Eretria, renewed the alliance with the Boeotian League and the Aetolian League, raised the siege of Athens and drove Cassander\'s forces from central Greece. In the spring of 303, Demetrius marched his army into the Peloponnese and took the cities of Sicyon and Corinth, he then campaigned in Argolis, Achaea and Arcadia, bringing the northern and central Peloponnese into the Antigonid camp. In 303–302 Demetrius formed a new Hellenic League, the League of Corinth, with himself and his father as presidents, to "defend" the Greek cities against all enemies (and particularly Cassander).\nIn the face of these catastrophes, Cassander sued for peace, but Antigonus rejected the claims, and Demetrius invaded Thessaly, where he and Cassander battled in inconclusive engagements. But now Cassander called in aid from his allies, and Anatolia was invaded by Lysimachus, forcing Demetrius to leave Thessaly and send his armies to Asia Minor to assist his father. With assistance from Cassander, Lysimachus overran much of western Anatolia, but was soon (301 BC) isolated by Antigonus and Demetrius near Ipsus. Here came the decisive intervention from Seleucus, who arrived in time to save Lysimachus from disaster and utterly crush Antigonus at the Battle of Ipsus. Antigonus was killed in the fight, and Demetrius fled back to Greece to attempt to preserve the remnants of his rule there. Lysimachus and Seleucus divided up Antigonus\'s Asian territories between them, with Lysimachus receiving western Asia Minor and Seleucus the rest, except Cilicia and Lycia, which went to Cassander\'s brother Pleistarchus.\n\n\n== The struggle over Macedon, 298–285 BC ==\nThe events of the next decade and a half were centered around various intrigues for control of Macedon itself.  Cassander died in 298 BC, and his sons, Antipater and Alexander, proved weak kings.  After quarreling with his older brother, Alexander V called in Demetrius, who had retained control of Cyprus, the Peloponnese, and many of the Aegean islands, and had quickly seized control of Cilicia and Lycia from Cassander\'s brother, as well as Pyrrhus, the King of Epirus.  After Pyrrhus had intervened to seize the border region of Ambracia, Demetrius invaded, killed Alexander, and seized control of Macedon for himself (294 BC).  While Demetrius consolidated his control of mainland Greece, his outlying territories were invaded and captured by Lysimachus (who recovered western Anatolia), Seleucus (who took most of Cilicia), and Ptolemy (who recovered Cyprus, eastern Cilicia, and Lycia).\nSoon, Demetrius was forced from Macedon by a rebellion supported by the alliance of Lysimachus and Pyrrhus, who divided the Kingdom between them, and, leaving Greece to the control of his son, Antigonus Gonatas, Demetrius launched an invasion of the east in 287 BC.  Although initially successful, Demetrius was ultimately captured by Seleucus (286 BC), drinking himself to death two years later.\n\n\n== The struggle of Lysimachus and Seleucus, 285–281 BC ==\nAlthough Lysimachus and Pyrrhus had cooperated in driving Antigonus Gonatas from Thessaly and Athens, in the wake of Demetrius\'s capture they soon fell out, with Lysimachus driving Pyrrhus from his share of Macedon. Dynastic struggles also rent Egypt, where Ptolemy decided to make his younger son Ptolemy Philadelphus his heir rather than the elder, Ptolemy Ceraunus.  Ceraunus fled to Seleucus.  The eldest Ptolemy died peacefully in his bed in 282 BC, and Philadelphus succeeded him.\nIn 282 BC Lysimachus had his son Agathocles murdered, possibly at the behest of his second wife, Arsinoe II. Agathocles\'s widow, Lysandra, fled to Seleucus, who after appointing his son Antiochus ruler of his Asian territories, defeated and killed Lysimachus at the Battle of Corupedium in Lydia in 281 BC. Selucus hoped to take control of Lysimachus\' European territories, and in 281 BC, soon after arriving in Thrace, he was assassinated by Ptolemy Ceraunus, for reasons that remain unclear.\n\n\n== The Gallic invasions and consolidation, 280–275 BC ==\nPtolemy Ceraunus did not rule Macedon for very long. The death of Lysimachus had left the Danube border of the Macedonian kingdom open to barbarian invasions, and soon tribes of Gauls were rampaging through Macedon and Greece, and invading Asia Minor. Ptolemy Ceraunus was killed by the invaders, and after several years of chaos, Demetrius\'s son Antigonus Gonatas emerged as ruler of Macedon. In Asia, Seleucus\'s son, Antiochus I, also managed to defeat the Celtic invaders, who settled down in central Anatolia in the part of eastern Phrygia that would henceforward be known as Galatia after them.\nNow, almost fifty years after Alexander\'s death, some sort of order was restored. Ptolemy ruled over Egypt, southern Syria (known as Coele-Syria), and various territories on the southern coast of Asia Minor. Antiochus ruled the Asian territories of the empire, while Macedon and Greece (with the exception of the Aetolian League) fell to Antigonus.\n\n\n== Aftermath ==\n\n\n== References ==\n\nShipley, Graham (2000) The Greek World After Alexander. Routledge History of the Ancient World. (Routledge, New York)\nWalbank, F. W. (1984) The Hellenistic World, The Cambridge Ancient History, volume VII. part I.  (Cambridge)\nWaterfield, Robin (2011). Dividing the Spoils – The War for Alexander the Great\'s Empire (hardback). New York: Oxford University Press. pp. 273 pages. ISBN 978-0-19-957392-9.\n\n\n== External links ==\nAlexander\'s successors: the Diadochi from Livius.org (Jona Lendering)\nWiki Classical Dictionary: "Successors" category and Diadochi entry\nT. Boiy, "Dating Methods During the Early Hellenistic Period", Journal of Cuneiform Studies, Vol. 52, 2000 PDF format. A recent study of primary sources for the chronology of eastern rulers during the period of the Diadochi.'




```python
articles_overview[0][0]['url']
```




    'https://en.wikipedia.org/wiki/Wars_of_the_Diadochi'




```python
summary_text = articles_overview[0][0]['summary']
```


```python
def generate_wiki_outline(model_name, summary_text, related_search_results):
    # Wikipediaページのアウトラインを生成する関数
    prompt = [
        {"role": "system", "content": "You are an experienced Wikipedia writer and want to edit a specific page."},
        {"role": "system", "content": "Your task is to write an outline for a Wikipedia page based on a question summary and related search results."},
        {"role": "system", "content": "The outline should prioritize the question summary over related search results."},
        {"role": "system", "content": "Here is the format of your writing:"},
        {"role": "system", "content": "1. Use '#' Title ' to indicate section title, '##' Title ' to indicate subsection title, '###' Title ' to indicate subsubsection title, and so on."},
        {"role": "system", "content": "2. Do not include other information."},
        {"role": "user", "content": f"Question Summary: {summary_text}"},
        {"role": "user", "content": f"Related Search Results: {related_search_results}"},
        {"role": "system", "content": "Based on this information, please create a structured outline for the Wikipedia page."},
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=TEMPERATURE,
    )
    
    outline = response.choices[0].message.content
    
    return outline

```


```python
%%time
# outline_text = generate_wiki_outline(MODEL_NAME, summary_text, related_search_results)
outline_text = generate_wiki_outline(MODEL4o_NAME, summary_text, related_search_results)

print(outline_text)
```

    # Wars of the Diadochi
    ## Introduction
    ### Definition and Terminology
    ### Historical Context
    ## Death of Alexander the Great (323 BCE)
    ### Circumstances of Alexander’s Death
    ### Immediate Aftermath and Leadership Void
    ## Initial Succession Plans
    ### Alexander’s Exile Decree and Rebellion
    ### Perdiccas and the Signet Ring
    ### Division Among Generals
    ## The Lamian War (323-322 BCE)
    ### Causes and Key Events
    ### Athens and Aetolia's Revolt
    ### Intervention by Antipater and Craterus
    ## Partition of Babylon (322 BCE)
    ### Territorial Division
    ### Key Figures and Their Assignments
    ## First Successor War (322-320 BCE)
    ### Territorial Disputes
    ### Perdiccas vs. Ptolemy
    ### Treaty of Triparadeisus
    ## Second Successor War (319-315 BCE)
    ### Cassander vs. Polyperchon
    ### Shifting Alliances
    ## Third Successor War (314-311 BCE)
    ### Antigonus’s Ambitions
    ### Conflict with Eumenes
    ### Betrayal and Execution of Eumenes
    ## Babylonian War (311-309 BCE)
    ### Seleucus’s Return to Power
    ### Role of Ptolemy and Antigonus
    ## Fourth Successor War (308-301 BCE)
    ### Coalition Against Antigonus and Demetrius
    ### Battle of Ipsus
    ### Death of Antigonus
    ## Establishment of Dynasties
    ### Antigonid Dynasty
    ### Ptolemaic Dynasty
    ### Seleucid Dynasty
    ## Death of Lysimachus (281 BCE)
    ### Battle of Corupedium
    ### Assassination of Seleucus
    ## Conclusion
    ### Legacy of the Wars
    ### Impact on the Hellenistic World
    CPU times: user 10.9 ms, sys: 57 µs, total: 11 ms
    Wall time: 8.29 s



```python
def generate_detailed_outline(model_name, outline, summary_text, related_search_results):
    """
    Given a summary and related search results, enhance a Wikipedia page outline with detailed descriptions.
    
    Parameters:
    model_name (str): The model to be used for generating the descriptions.
    summary_text (str): Summary text providing a concise overview of the topic.
    related_search_results (str): Text containing related search results or additional contextual information.
    outline (str): The basic outline of the Wikipedia page.
    
    Returns:
    str: The enhanced outline with detailed descriptions for each section.
    """
    prompt = [
        {"role": "system", "content": "You are tasked with enhancing a Wikipedia page outline by integrating detailed descriptions based on a given summary and related search results."},
        {"role": "system", "content": "Here is the basic outline of the page you need to expand with detailed explanations:"},
        {"role": "user", "content": outline},
        {"role": "system", "content": "Use the following summary and related search results to provide detailed descriptions for each section of the outline."},
        {"role": "user", "content": f"Summary: {summary_text}"},
        {"role": "user", "content": f"Related Search Results: {related_search_results}"},
        {"role": "system", "content": "Based on the summary and related search results, please add a comprehensive explanation for each section of the outline that includes historical context, significance of events, and relevant interpretations."}
    ]
    
    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=TEMPERATURE,
    )
    
    detailed_outline = response.choices[0].message.content
    
    return detailed_outline

```


```python
%%time
# 生成したアウトラインに詳細な説明を加える
detailed_outline = generate_detailed_outline(MODEL4o_NAME, outline_text, summary_text, related_search_results)
print(detailed_outline)
```

    # Wars of the Diadochi
    
    ## Introduction
    
    ### Definition and Terminology
    The Wars of the Diadochi, derived from the Ancient Greek term "Πόλεμοι τῶν Διαδόχων" (Pólemoi tōn Diadóchōn), literally translates to the Wars of Alexander the Great's Successors. These conflicts spanned from 322 to 281 BCE and were primarily fought among Alexander's generals and their descendants over control of his expansive empire following his untimely death.
    
    ### Historical Context
    Alexander the Great's empire stretched from the Adriatic Sea to the Indus River, encompassing a diverse range of cultures and territories. His sudden death in 323 BCE left a power vacuum that his generals, known as the Diadochi, vied to fill. The resulting conflicts profoundly shaped the political landscape of the Hellenistic world, leading to the establishment of several enduring dynasties.
    
    ## Death of Alexander the Great (323 BCE)
    
    ### Circumstances of Alexander’s Death
    On June 10, 323 BCE, Alexander the Great died in the palace of Nebuchadnezzar II in Babylon at the age of 32. The cause of his death remains uncertain, with theories ranging from natural causes such as fever or poisoning to assassination. His death marked the end of his unprecedented conquests and left his empire without a clear successor.
    
    ### Immediate Aftermath and Leadership Void
    Alexander's death triggered immediate chaos and uncertainty. His generals, or Diadochi, were left to decide the fate of his vast empire. The lack of a clear succession plan led to intense rivalry and conflict. Alexander's final words, reportedly "to the strongest," only fueled the power struggle among his top commanders.
    
    ## Initial Succession Plans
    
    ### Alexander’s Exile Decree and Rebellion
    Prior to his death, Alexander had proposed the Exile Decree, which aimed to return Greek exiles to their homes. This decree caused unrest and rebellion among his ranks. Many troops protested, and several satraps were executed for alleged treason, highlighting the tension and potential instability even before Alexander's death.
    
    ### Perdiccas and the Signet Ring
    On his deathbed, Alexander handed his signet ring to Perdiccas, one of his trusted generals. This gesture was interpreted by many as Alexander's endorsement of Perdiccas as his successor, though it did not quell the disputes over leadership. Perdiccas assumed the role of regent, but his authority was immediately challenged by other generals.
    
    ### Division Among Generals
    The initial succession struggle saw commanders divided in their support for different claimants to the throne. Key figures included Alexander's half-brother Arrhidaeus, who became Philip III, and his unborn son, Alexander IV. The disagreements among the generals set the stage for the protracted Wars of the Diadochi.
    
    ## The Lamian War (323-322 BCE)
    
    ### Causes and Key Events
    The Lamian War was one of the immediate conflicts following Alexander's death. Athens and Aetolia seized the opportunity to rebel against Macedonian control, aiming to restore their autonomy. The war was driven by the desire to reject Macedonian hegemony and capitalize on the power vacuum.
    
    ### Athens and Aetolia's Revolt
    The revolt saw significant battles, including the Siege of Lamia, where Greek forces initially gained the upper hand. However, the lack of unified leadership and resources ultimately hampered their efforts.
    
    ### Intervention by Antipater and Craterus
    Antipater, the Macedonian regent in Europe, and Craterus, one of Alexander's key generals, intervened to suppress the rebellion. The decisive Battle of Crannon in 322 BCE resulted in the defeat of the Greek forces, firmly reestablishing Macedonian control over Greece.
    
    ## Partition of Babylon (322 BCE)
    
    ### Territorial Division
    In 322 BCE, the Partition of Babylon was agreed upon by Alexander's generals to divide his empire into manageable territories. This agreement aimed to maintain a semblance of unity while acknowledging the practicalities of governance.
    
    ### Key Figures and Their Assignments
    The partition allocated regions to various commanders: Antipater and Craterus received Macedon and Greece, Ptolemy secured Egypt, Lysimachus took Thrace, Eumenes was given Cappadocia, and Antigonus retained Phrygia. This division laid the groundwork for future conflicts as each general sought to expand their influence.
    
    ## First Successor War (322-320 BCE)
    
    ### Territorial Disputes
    The First Successor War erupted over territorial disputes and ambitions. Perdiccas, as regent, faced opposition from other generals who were dissatisfied with their allocations and sought greater power.
    
    ### Perdiccas vs. Ptolemy
    Perdiccas's attempt to invade Egypt and overthrow Ptolemy ended in failure. His troops mutinied, leading to his assassination in 321 BCE. This event underscored the fragility of alliances and the volatility of the power struggle.
    
    ### Treaty of Triparadeisus
    Following Perdiccas's death, the Treaty of Triparadeisus in 321 BCE reallocated territories and appointed new regents, with Antipater becoming the new regent of the empire. This treaty temporarily stabilized the situation but did not resolve the underlying conflicts.
    
    ## Second Successor War (319-315 BCE)
    
    ### Cassander vs. Polyperchon
    The Second Successor War saw Cassander and Polyperchon vying for control over Macedon and Greece. Polyperchon, initially appointed as regent, faced opposition from Cassander, who sought to establish his authority.
    
    ### Shifting Alliances
    Alliances during this period were fluid, with various commanders switching sides to gain strategic advantages. The war ended with Cassander consolidating his power, though the conflict left the region politically fragmented and unstable.
    
    ## Third Successor War (314-311 BCE)
    
    ### Antigonus’s Ambitions
    Antigonus, one of the most formidable Diadochi, sought to reunite Alexander's empire under his rule. His ambitions led to confrontations with other generals who viewed him as a threat to their own power.
    
    ### Conflict with Eumenes
    Eumenes, a loyal supporter of Alexander's family, emerged as a key rival to Antigonus. Despite initial successes, Eumenes was ultimately betrayed by his own troops and executed in 316 BCE, solidifying Antigonus's position.
    
    ### Betrayal and Execution of Eumenes
    Eumenes's betrayal highlighted the precarious nature of loyalty among the Diadochi. His execution marked a significant victory for Antigonus but also intensified the rivalries among the remaining generals.
    
    ## Babylonian War (311-309 BCE)
    
    ### Seleucus’s Return to Power
    Seleucus, initially sidelined, managed to regain control of Babylonia with the support of Ptolemy. This marked the beginning of Seleucus's rise to prominence and the establishment of the Seleucid Empire.
    
    ### Role of Ptolemy and Antigonus
    The Babylonian War saw Seleucus and Ptolemy forming an alliance against Antigonus and his son Demetrius. This conflict further fragmented Alexander's empire and set the stage for future wars.
    
    ## Fourth Successor War (308-301 BCE)
    
    ### Coalition Against Antigonus and Demetrius
    The Fourth Successor War witnessed a coalition of Cassander, Ptolemy, Lysimachus, and Seleucus forming against Antigonus and Demetrius. The coalition aimed to curb Antigonus's growing power and prevent him from reuniting the empire.
    
    ### Battle of Ipsus
    The decisive Battle of Ipsus in 301 BCE resulted in the defeat and death of Antigonus. His territories were divided among the victors, marking the end of his ambitions and the solidification of the Hellenistic kingdoms.
    
    ### Death of Antigonus
    Antigonus's death at Ipsus marked a turning point in the Wars of the Diadochi. It led to the final partition of Alexander's empire and the emergence of three dominant Hellenistic dynasties.
    
    ## Establishment of Dynasties
    
    ### Antigonid Dynasty
    The Antigonid dynasty, established by Antigonus's descendants, ruled Macedon and parts of Greece. Despite their initial setbacks, they managed to maintain control until the Roman conquest.
    
    ### Ptolemaic Dynasty
    The Ptolemaic dynasty, founded by Ptolemy I Soter, ruled Egypt and established Alexandria as a major cultural and intellectual center. The dynasty lasted until the Roman annexation of Egypt in 30 BCE.
    
    ### Seleucid Dynasty
    The Seleucid dynasty, founded by Seleucus I Nicator, controlled a vast territory stretching from Asia Minor to the Indus Valley. The Seleucids played a crucial role in the cultural and political landscape of the Hellenistic world.
    
    ## Death of Lysimachus (281 BCE)
    
    ### Battle of Corupedium
    Lysimachus, one of the last surviving Diadochi, was defeated and killed by Seleucus at the Battle of Corupedium in 281 BCE. This battle marked the end of significant conflicts among Alexander's immediate successors.
    
    ### Assassination of Seleucus
    Following his victory, Seleucus was assassinated by Ptolemy Keraunos, the son of Ptolemy I, in 281 BCE. This event signaled the end of the Diadochi era and the beginning of a new phase in the Hellenistic period.
    
    ## Conclusion
    
    ### Legacy of the Wars
    The Wars of the Diadochi profoundly shaped the course of history. They led to the fragmentation of Alexander's empire and the establishment of Hellenistic kingdoms that influenced the cultural, political, and economic landscape of the ancient world.
    
    ### Impact on the Hellenistic World
    The conflicts and the resulting dynasties fostered the spread of Greek culture and ideas across a vast region. The Hellenistic period, characterized by a blend of Greek and local cultures, laid the groundwork for the subsequent rise of the Roman Empire and the enduring legacy of Alexander's conquests.
    CPU times: user 14.5 ms, sys: 867 µs, total: 15.4 ms
    Wall time: 24.2 s



```python

```


```python
# outlineの分解
```


```python

```


```python
def translate_to_japanese(model_name, detailed_outline):
    """
    Translates the detailed outline of a Wikipedia page from English to Japanese.
    
    Parameters:
    model_name (str): The model to be used for translation.
    detailed_outline (str): The detailed outline in English.
    
    Returns:
    str: The translated outline in Japanese.
    """
    prompt = [
        {"role": "system", "content": "You need to translate the following English text into Japanese."},
        {"role": "user", "content": detailed_outline},
        {"role": "system", "content": "Please provide the translation of the entire text into Japanese, maintaining the accuracy and context of the original information."}
    ]
    
    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=TEMPERATURE,
    )
    
    translated_outline = response.choices[0].message.content
    
    return translated_outline
```


```python
%%time
translated_outline = translate_to_japanese(MODEL_NAME, detailed_outline)
```

    CPU times: user 11 ms, sys: 66 µs, total: 11 ms
    Wall time: 1min 11s



```python
print(translated_outline)
```

    ＃ディアドコイの戦争
    
    ## 序文
    
    ### 定義と用語
    ディアドコイの戦争は、古代ギリシャ語の「Πόλεμοι τῶν Διαδόχων」（Pólemoi tōn Diadóchōn）に由来し、文字通りアレクサンダー大王の後継者たちの戦争を意味します。これらの紛争は紀元前322年から紀元前281年まで続き、アレクサンダー大王の死後、彼の広大な帝国の支配権を巡って主にアレクサンダーの将軍たちとその子孫の間で戦われました。
    
    ### 歴史的背景
    アレクサンダー大王の帝国はアドリア海からインダス川まで広がり、多様な文化と領土を包含していました。彼の突然の死（紀元前323年）は、彼の将軍たちであるディアドコイがその帝国の支配権を巡って競い合う機会を残しました。その結果、ヘレニズム世界の政治的景観が大きく変化し、いくつかの持続的な王朝が確立されました。
    
    ## アレクサンダー大王の死（紀元前323年）
    
    ### アレクサンダーの死の状況
    紀元前323年6月10日、アレクサンダー大王はバビロンのネブカドネザル2世の宮殿で32歳で亡くなりました。彼の死因は不明で、熱病や毒殺から暗殺などの諸説があります。彼の死は彼の前例のない征服の終わりを告げ、彼の帝国には明確な後継者がいませんでした。
    
    ### 直後の混乱と指導者不在
    アレクサンダーの死は直ちに混乱と不確実性を引き起こしました。彼の将軍たち、またはディアドコイたちは、彼の広大な帝国の運命を決定することになりました。明確な後継計画がなかったことから、激しいライバル関係と紛争が生じました。アレクサンダーの最期の言葉、「最強なる者に」と伝えられたのは、彼のトップ指揮官たちの権力争いを火に油を注ぎました。
    
    ## 最初の後継計画
    
    ### アレクサンダーの追放令と反乱
    死の直前にアレクサンダーは、ギリシャ人亡命者を故国に戻すことを目的とした追放令を提案していました。この令が原因で、多くの兵士が抗議し、いくつかのサトラップが裏切りの罪で処刑されました。アレクサンダーの死前からすでに緊張と潜在的な不安定性が浮き彫りにされていました。
    
    ### ペルディッカスと指輪
    アレクサンダーは死の床で、彼の信頼する将軍の一人であるペルディッカスに指輪を手渡ししました。多くの人がこれをアレクサンダーによるペルディッカスの後継者という認識したが、それでも指導権をめぐる争いは収まりませんでした。ペルディッカスは摂政の役割を引き受けましたが、直ちに他の将軍たちからその権威が挑戦されました。
    
    ### 将軍たちの分裂
    最初の後継争いでは、指揮官たちは王位継承者候補に対する支持で分かれました。主要な人物には、アレクサンダーの異母兄弟であるアリダイオス（フィリッポス3世）や、まだ胎児であったアレクサンダー4世などが含まれていました。将軍たちの間の意見の不一致が、長期にわたるディアドコイの戦争の舞台を築き上げました。
    
    ## ラミア戦争（紀元前323-322年）
    
    ### 起因と主要な出来事
    ラミア戦争は、アレクサンダーの死の直後に起こった紛争の一つでした。アテナとアイトリアは、マケドニアの支配に反抗し、自治を回復することを目指しました。この戦争は、マケドニアの覇権を拒否し、権力の空白を利用しようとした欲望によって引き起こされました。
    
    ### アテナとアイトリアの反乱
    反乱では、ギリシャ軍が初期に優位を占めたラミア包囲戦など、重要な戦闘が繰り広げられました。しかし、統一された指導者不在や資源の不足が最終的に彼らの努力を阻害しました。
    
    ### アンティパトロスとクラテロスの介入
    マケドニアの摂政であるアンティパトロスとアレクサンダーの主要な将軍の一人であるクラテロスが反乱を鎮圧するために介入しました。紀元前322年のクラノンの戦いでは、ギリシャ軍の敗北により、マケドニアに対する支配がしっかりと再確立されました。
    
    ## バビロンの分割（紀元前322年）
    
    ### 地域の分割
    紀元前322年、アレクサンダーの将軍たちによって彼の帝国を管理可能な地域に分割するためのバビロンの分割が合意されました。この合意は、統一の様相を保ちつつ、統治の実際的な側面を認識することを目的としていました。
    
    ### 主要人物と彼らの任務
    分割により、さまざまな指揮官に地域が割り当てられました。アンティパトロスとクラテロスがマケドンとギリシャを受け取り、プトレマイオスがエジプトを確保し、リュシマコスがトラキアを、エウメネスがカッパドキアを、アンティゴノスがフリギアを担当しました。この分割は、各将軍が影響を拡大しようとする中で将来の紛争の基盤を築きました。
    
    ## 初期の後継戦争（紀元前322-320年）
    
    ### 領土紛争
    初期の後継戦争は、領土紛争と野心によって勃発しました。摂政であるペルディッカスは、他の将軍たちからの不満を抱え、彼らの割り当てに不満を持ち大きな権力を求められました。
    
    ### ペルディッカス対プトレマイオス
    ペルディッカスがエジプトに侵攻してプトレマイオスを打倒しようとした試みは失敗に終わりました。彼の兵士たちが暴動を起こし、結果的に彼は紀元前321年に暗殺されました。この出来事は同盟関係の脆弱性と権力争いの不安定性を浮き彫りにしました。
    
    ### トリパラデイソスの条約
    ペルディッカスの死後、紀元前321年にトリパラデイソスの条約が締結され、領土が再分配され、新たな摂政が任命されました。この条約は一時的に状況を安定させましたが、根本的な紛争を解決することはありませんでした。
    
    ## 第二次後継戦争（紀元前319-315年）
    
    ### カッサンドロス対ポリペルコン
    第二次後継戦争では、カッサンドロスとポリペルコンがマケドンとギリシャの支配権を巡って争いました。最初は摂政として任命されたポリペルコンは、カッサンドロスからの反対に直面し、自らの権威を確立しようとしました。
    
    ### 変動する同盟関係
    この時期の同盟関係は流動的で、さまざまな指揮官が戦略的な利点を得るために陣営を変えました。戦争はカッサンドロスが権力を固めることで終結しましたが、この紛争は地域を政治的に分断し、不安定にしました。
    
    ## 第三次後継戦争（紀元前314-311年）
    
    ### アンティゴノスの野望
    アンティゴノスは、最も手ごわいディアドコイの一人としてアレクサンダーの帝国を再統一しようとしました。彼の野望は、他の将軍たちにとって自らの権力に対する脅威と見なされ、対立を引き起こしました。
    
    ### エウメネスとの対立
    アレクサンダーの家族の忠実な支持者であるエウメネスは、アンティゴノスにとっての主要なライバルとして浮上しました。初期の成功にもかかわらず、エウメネスは最終的に自らの部下に裏切られ、紀元前316年に処刑されました。これによりアンティゴノスの地位が固定されました。
    
    ### エウメネスの裏切りと処刑
    エウメネスの裏切りは、ディアドコイの間の忠誠心の不確かさを示しました。彼の処刑はアンティゴノスにとって重要な勝利をもたらしましたが、他の将軍たちの間の対立を激化させました。
    
    ## バビロニア戦争（紀元前311-309年）
    
    ### セレウコスの権力回復
    最初は脇に置かれていたセレウコスは、プトレマイオスの支援を受けてバビロニアの支配権を回復しました。これはセレウコスの隆盛とセレウコス朝の確立の始まりでした。
    
    ### プトレマイオスとアンティゴノスの役割
    バビロニア戦争では、セレウコスとプトレマイオスがアンティゴノスと彼の息子デメトリオスに対抗して同盟を組みました。この紛争はアレクサンダーの帝国をさらに分裂させ、将来の戦争の舞台を設定しました。
    
    ## 第四次後継戦争（紀元前308-301年）
    
    ### アンティゴノスとデメトリオスに対する連合
    第四次後継戦争では、カッサンドロス、プトレマイオス、リュシマコス、セレウコスからなる連合がアンティゴノスとデメトリオスに対抗しました。この連合は、アンティゴノスの権力拡大を抑制し、帝国を再統一させないようにすることを目指しました。
    
    ### イプソスの戦い
    紀元前301年のイプソスの戦いは、アンティゴノスの敗北と死をもたらしました。彼の領土は勝者たちに分割され、彼の野望の終わりとヘレニズム諸王国の確立を示すものでした。
    
    ### アンティゴノスの死
    イプソスでのアンティゴノスの死は、ディアドコイの戦争の転換点となりました。これはアレクサンダーの帝国の最終的な分割と、三つの主要なヘレニズム王朝の台頭をもたらしました。
    
    ## 王朝の確立
    
    ### アンティゴノス朝
    アンティゴノス朝は、アンティゴノスの子孫によって建てられ、マケドンとギリシャの一部を支配しました。初期の挫折にもかかわらず、彼らはローマの征服まで支配を維持しました。
    
    ### トレマー朝
    プトレマイオス1世ソテルによって創設されたプトレマイオス朝は、エジプトを支配し、アレクサンドリアを主要な文化・知的中心地として確立しました。この王朝は紀元前30年のエジプト併合まで続きました。
    
    ### セレウコス朝
    セレウコス1世ニカトルによって創設されたセレウコス朝は、アジア小アジアからインダス川流域まで広がる広大な領土を支配しました。



```python
def remove_duplicates_keep_order(original_list):
    seen = set()
    unique_list = []
    for item in original_list:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    return unique_list
#     # 全リンクを一つのセットに統合する
#     unique_links = set()
#     for sublist in original_list:
#         unique_links.update(sublist)  # 各サブリストのリンクを追加

#     # セットをリストに変換
#     consolidated_list = list(unique_links)
    
#     return consolidated_list


# 重複を削除
last_search_links = []
last_search_links.append(articles_overview[0][0]['url'])

for link in related_search_links:
    last_search_links.append(link)


# 重複を削除
last_search_links = remove_duplicates_keep_order(last_search_links)
```


```python
last_search_links
```




    ['https://en.wikipedia.org/wiki/Wars_of_the_Diadochi',
     'https://www.worldhistory.org/Wars_of_the_Diadochi/',
     'https://www.thecollector.com/who-were-the-diadochi-of-alexander-the-great/',
     'https://www.livius.org/articles/concept/diadochi/',
     'https://www.ancient-origins.net/history-important-events/diadochi-0016823',
     'https://www.worldhistory.org/timeline/Wars_of_the_Diadochi/',
     'https://history-maps.com/story/Seleucid-Empire/event/Wars-of-the-Diadochi']




```python
translated_outline += "\n\n## 参考リンク\n"

for link in last_search_links:
    if len(link) > 5:
        translated_outline += "- " + str(link) + "\n"
```


```python
print(translated_outline)
```

    ＃ディアドコイの戦争
    
    ## 序文
    
    ### 定義と用語
    ディアドコイの戦争は、古代ギリシャ語の「Πόλεμοι τῶν Διαδόχων」（Pólemoi tōn Diadóchōn）に由来し、文字通りアレクサンダー大王の後継者たちの戦争を意味します。これらの紛争は紀元前322年から紀元前281年まで続き、アレクサンダー大王の死後、彼の広大な帝国の支配権を巡って主にアレクサンダーの将軍たちとその子孫の間で戦われました。
    
    ### 歴史的背景
    アレクサンダー大王の帝国はアドリア海からインダス川まで広がり、多様な文化と領土を包含していました。彼の突然の死（紀元前323年）は、彼の将軍たちであるディアドコイがその帝国の支配権を巡って競い合う機会を残しました。その結果、ヘレニズム世界の政治的景観が大きく変化し、いくつかの持続的な王朝が確立されました。
    
    ## アレクサンダー大王の死（紀元前323年）
    
    ### アレクサンダーの死の状況
    紀元前323年6月10日、アレクサンダー大王はバビロンのネブカドネザル2世の宮殿で32歳で亡くなりました。彼の死因は不明で、熱病や毒殺から暗殺などの諸説があります。彼の死は彼の前例のない征服の終わりを告げ、彼の帝国には明確な後継者がいませんでした。
    
    ### 直後の混乱と指導者不在
    アレクサンダーの死は直ちに混乱と不確実性を引き起こしました。彼の将軍たち、またはディアドコイたちは、彼の広大な帝国の運命を決定することになりました。明確な後継計画がなかったことから、激しいライバル関係と紛争が生じました。アレクサンダーの最期の言葉、「最強なる者に」と伝えられたのは、彼のトップ指揮官たちの権力争いを火に油を注ぎました。
    
    ## 最初の後継計画
    
    ### アレクサンダーの追放令と反乱
    死の直前にアレクサンダーは、ギリシャ人亡命者を故国に戻すことを目的とした追放令を提案していました。この令が原因で、多くの兵士が抗議し、いくつかのサトラップが裏切りの罪で処刑されました。アレクサンダーの死前からすでに緊張と潜在的な不安定性が浮き彫りにされていました。
    
    ### ペルディッカスと指輪
    アレクサンダーは死の床で、彼の信頼する将軍の一人であるペルディッカスに指輪を手渡ししました。多くの人がこれをアレクサンダーによるペルディッカスの後継者という認識したが、それでも指導権をめぐる争いは収まりませんでした。ペルディッカスは摂政の役割を引き受けましたが、直ちに他の将軍たちからその権威が挑戦されました。
    
    ### 将軍たちの分裂
    最初の後継争いでは、指揮官たちは王位継承者候補に対する支持で分かれました。主要な人物には、アレクサンダーの異母兄弟であるアリダイオス（フィリッポス3世）や、まだ胎児であったアレクサンダー4世などが含まれていました。将軍たちの間の意見の不一致が、長期にわたるディアドコイの戦争の舞台を築き上げました。
    
    ## ラミア戦争（紀元前323-322年）
    
    ### 起因と主要な出来事
    ラミア戦争は、アレクサンダーの死の直後に起こった紛争の一つでした。アテナとアイトリアは、マケドニアの支配に反抗し、自治を回復することを目指しました。この戦争は、マケドニアの覇権を拒否し、権力の空白を利用しようとした欲望によって引き起こされました。
    
    ### アテナとアイトリアの反乱
    反乱では、ギリシャ軍が初期に優位を占めたラミア包囲戦など、重要な戦闘が繰り広げられました。しかし、統一された指導者不在や資源の不足が最終的に彼らの努力を阻害しました。
    
    ### アンティパトロスとクラテロスの介入
    マケドニアの摂政であるアンティパトロスとアレクサンダーの主要な将軍の一人であるクラテロスが反乱を鎮圧するために介入しました。紀元前322年のクラノンの戦いでは、ギリシャ軍の敗北により、マケドニアに対する支配がしっかりと再確立されました。
    
    ## バビロンの分割（紀元前322年）
    
    ### 地域の分割
    紀元前322年、アレクサンダーの将軍たちによって彼の帝国を管理可能な地域に分割するためのバビロンの分割が合意されました。この合意は、統一の様相を保ちつつ、統治の実際的な側面を認識することを目的としていました。
    
    ### 主要人物と彼らの任務
    分割により、さまざまな指揮官に地域が割り当てられました。アンティパトロスとクラテロスがマケドンとギリシャを受け取り、プトレマイオスがエジプトを確保し、リュシマコスがトラキアを、エウメネスがカッパドキアを、アンティゴノスがフリギアを担当しました。この分割は、各将軍が影響を拡大しようとする中で将来の紛争の基盤を築きました。
    
    ## 初期の後継戦争（紀元前322-320年）
    
    ### 領土紛争
    初期の後継戦争は、領土紛争と野心によって勃発しました。摂政であるペルディッカスは、他の将軍たちからの不満を抱え、彼らの割り当てに不満を持ち大きな権力を求められました。
    
    ### ペルディッカス対プトレマイオス
    ペルディッカスがエジプトに侵攻してプトレマイオスを打倒しようとした試みは失敗に終わりました。彼の兵士たちが暴動を起こし、結果的に彼は紀元前321年に暗殺されました。この出来事は同盟関係の脆弱性と権力争いの不安定性を浮き彫りにしました。
    
    ### トリパラデイソスの条約
    ペルディッカスの死後、紀元前321年にトリパラデイソスの条約が締結され、領土が再分配され、新たな摂政が任命されました。この条約は一時的に状況を安定させましたが、根本的な紛争を解決することはありませんでした。
    
    ## 第二次後継戦争（紀元前319-315年）
    
    ### カッサンドロス対ポリペルコン
    第二次後継戦争では、カッサンドロスとポリペルコンがマケドンとギリシャの支配権を巡って争いました。最初は摂政として任命されたポリペルコンは、カッサンドロスからの反対に直面し、自らの権威を確立しようとしました。
    
    ### 変動する同盟関係
    この時期の同盟関係は流動的で、さまざまな指揮官が戦略的な利点を得るために陣営を変えました。戦争はカッサンドロスが権力を固めることで終結しましたが、この紛争は地域を政治的に分断し、不安定にしました。
    
    ## 第三次後継戦争（紀元前314-311年）
    
    ### アンティゴノスの野望
    アンティゴノスは、最も手ごわいディアドコイの一人としてアレクサンダーの帝国を再統一しようとしました。彼の野望は、他の将軍たちにとって自らの権力に対する脅威と見なされ、対立を引き起こしました。
    
    ### エウメネスとの対立
    アレクサンダーの家族の忠実な支持者であるエウメネスは、アンティゴノスにとっての主要なライバルとして浮上しました。初期の成功にもかかわらず、エウメネスは最終的に自らの部下に裏切られ、紀元前316年に処刑されました。これによりアンティゴノスの地位が固定されました。
    
    ### エウメネスの裏切りと処刑
    エウメネスの裏切りは、ディアドコイの間の忠誠心の不確かさを示しました。彼の処刑はアンティゴノスにとって重要な勝利をもたらしましたが、他の将軍たちの間の対立を激化させました。
    
    ## バビロニア戦争（紀元前311-309年）
    
    ### セレウコスの権力回復
    最初は脇に置かれていたセレウコスは、プトレマイオスの支援を受けてバビロニアの支配権を回復しました。これはセレウコスの隆盛とセレウコス朝の確立の始まりでした。
    
    ### プトレマイオスとアンティゴノスの役割
    バビロニア戦争では、セレウコスとプトレマイオスがアンティゴノスと彼の息子デメトリオスに対抗して同盟を組みました。この紛争はアレクサンダーの帝国をさらに分裂させ、将来の戦争の舞台を設定しました。
    
    ## 第四次後継戦争（紀元前308-301年）
    
    ### アンティゴノスとデメトリオスに対する連合
    第四次後継戦争では、カッサンドロス、プトレマイオス、リュシマコス、セレウコスからなる連合がアンティゴノスとデメトリオスに対抗しました。この連合は、アンティゴノスの権力拡大を抑制し、帝国を再統一させないようにすることを目指しました。
    
    ### イプソスの戦い
    紀元前301年のイプソスの戦いは、アンティゴノスの敗北と死をもたらしました。彼の領土は勝者たちに分割され、彼の野望の終わりとヘレニズム諸王国の確立を示すものでした。
    
    ### アンティゴノスの死
    イプソスでのアンティゴノスの死は、ディアドコイの戦争の転換点となりました。これはアレクサンダーの帝国の最終的な分割と、三つの主要なヘレニズム王朝の台頭をもたらしました。
    
    ## 王朝の確立
    
    ### アンティゴノス朝
    アンティゴノス朝は、アンティゴノスの子孫によって建てられ、マケドンとギリシャの一部を支配しました。初期の挫折にもかかわらず、彼らはローマの征服まで支配を維持しました。
    
    ### トレマー朝
    プトレマイオス1世ソテルによって創設されたプトレマイオス朝は、エジプトを支配し、アレクサンドリアを主要な文化・知的中心地として確立しました。この王朝は紀元前30年のエジプト併合まで続きました。
    
    ### セレウコス朝
    セレウコス1世ニカトルによって創設されたセレウコス朝は、アジア小アジアからインダス川流域まで広がる広大な領土を支配しました。
    
    ## 参考リンク
    - https://en.wikipedia.org/wiki/Wars_of_the_Diadochi
    - https://www.worldhistory.org/Wars_of_the_Diadochi/
    - https://www.thecollector.com/who-were-the-diadochi-of-alexander-the-great/
    - https://www.livius.org/articles/concept/diadochi/
    - https://www.ancient-origins.net/history-important-events/diadochi-0016823
    - https://www.worldhistory.org/timeline/Wars_of_the_Diadochi/
    - https://history-maps.com/story/Seleucid-Empire/event/Wars-of-the-Diadochi
    



```python
MODEL_ENBEDDING_NAME = "text-embedding-3-small"
```


```python
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding
```


```python
message_embedding = get_embedding(message, model=MODEL_ENBEDDING_NAME)
```


```python
# translated_embedding_outline = get_embedding(translated_outline[:7000], model=MODEL_ENBEDDING_NAME)
translated_embedding_outline = get_embedding(translated_outline, model=MODEL_ENBEDDING_NAME)
# translated_embedding_outline = get_embedding(translated_outline, model="text-embedding-3-large")
# translated_embedding_outline = get_embedding(updated_outline_text, model=MODEL_ENBEDDING_NAME)

```


```python
import numpy as np

def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity
```


```python
len(message_embedding), len(translated_embedding_outline)
```




    (1536, 1536)




```python
# Calculating the cosine similarity
cosine_similarity_result = cosine_similarity(message_embedding, translated_embedding_outline)
cosine_similarity_result
```




    0.48425997637627205




```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_fixed(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors ensuring proper 2D shapes.
    """
    vector_a = np.array(vector_a).reshape(1, -1)
    vector_b = np.array(vector_b).reshape(1, -1)
    return cosine_similarity(vector_a, vector_b)[0][0]

def calculate_intersection(embedding_a, embedding_b):
    """
    Approximate the soft cardinality of the intersection of two sets of embeddings.
    """
    return cosine_similarity_fixed(embedding_a, embedding_b)

def soft_cardinality(embedding):
    """
    Calculate the soft cardinality for a set of embeddings.
    Each element's weight is inversely proportional to the sum of its cosine similarities with all other elements.
    """
    embedding = np.array(embedding).reshape(1, -1)  # Ensure embedding is 2D
    cosine_sim = cosine_similarity(embedding, embedding)
    weights = 1 / cosine_sim.sum(axis=1)
    return weights.sum()

def soft_precision(reference_embedding, predicted_embedding):
    """
    Calculate soft precision using embeddings of reference and predicted sets.
    """
    intersection = calculate_intersection(predicted_embedding, reference_embedding)
    predicted_cardinality = soft_cardinality(predicted_embedding)
    return intersection / predicted_cardinality

def soft_recall(reference_embedding, predicted_embedding):
    """
    Calculate soft recall using embeddings of reference and predicted sets.
    """
    intersection = calculate_intersection(predicted_embedding, reference_embedding)
    reference_cardinality = soft_cardinality(reference_embedding)
    return intersection / reference_cardinality

```


```python
# ソフトプレシジョンとソフトリコールの計算
precision = soft_precision(translated_embedding_outline, message_embedding)
recall = soft_recall(translated_embedding_outline, message_embedding)

print("Soft Precision:", precision)
print("Soft Recall:", recall)

```

    Soft Precision: 0.4842599763762731
    Soft Recall: 0.4842599763762729



```python

```
