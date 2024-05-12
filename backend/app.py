import re
import os
import json
import warnings
import requests
from bs4 import BeautifulSoup

from flask import Flask, request, jsonify
from flask_cors import CORS

import openai
from openai import OpenAI
from dotenv import load_dotenv
import wikipedia

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# すべての警告を無視する
warnings.filterwarnings('ignore')

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo-0125"
TEMPERATURE = 0.7
MODEL_ENBEDDING_NAME = "text-embedding-3-small"
# OpenAIクライアントの初期化
client = OpenAI()

# APIキー設定
api_key = os.getenv("BING_API_KEY")
# APIリクエストを送信
url = 'https://api.bing.microsoft.com/v7.0/search'

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
        app.logger.debug(f"DisambiguationError for keyword {keywords}: {e.options}")  # エラーメッセージを出力

    return all_articles  # 全記事情報を返す

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


# def generate_detailed_outline(model_name, outline, summary_text, related_search_results):
#     """
#     Given a summary and related search results, enhance a Wikipedia page outline with detailed descriptions.
    
#     Parameters:
#     model_name (str): The model to be used for generating the descriptions.
#     summary_text (str): Summary text providing a concise overview of the topic.
#     related_search_results (str): Text containing related search results or additional contextual information.
#     outline (str): The basic outline of the Wikipedia page.
    
#     Returns:
#     str: The enhanced outline with detailed descriptions for each section.
#     """
#     prompt = [
#         {"role": "system", "content": "You are tasked with enhancing a Wikipedia page outline by integrating detailed descriptions based on a given summary and related search results."},
#         {"role": "system", "content": "Here is the basic outline of the page you need to expand with detailed explanations:"},
#         {"role": "user", "content": outline},
#         {"role": "system", "content": "Use the following summary and related search results to provide detailed descriptions for each section of the outline."},
#         {"role": "user", "content": f"Summary: {summary_text}"},
#         {"role": "user", "content": f"Related Search Results: {related_search_results}"},
#         {"role": "system", "content": "Based on the summary and related search results, please add a comprehensive explanation for each section of the outline that includes historical context, significance of events, and relevant interpretations."}
#     ]
    
#     response = client.chat.completions.create(
#         model=model_name,
#         messages=prompt,
#         temperature=TEMPERATURE,
#     )
    
#     detailed_outline = response.choices[0].message.content
    
#     return detailed_outline

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

def split_outline_into_sections(outline_text):
    # アウトラインテキストを行に分割
    lines = outline_text.split('\n')
    
    # 各セクションとサブセクションを保存するリスト
    sections = []
    
    # 現在のセクションとサブセクションを追跡
    current_section = None
    current_subsection = None
    
    # 各行を処理
    for line in lines:
        if line.startswith('# '):  # 新しいセクション
            if current_section:
                sections.append(current_section)
            current_section = {'title': line[2:], 'content': [], 'subsections': []}
        elif line.startswith('## '):  # 新しいサブセクション
            if current_subsection:
                if current_section:
                    current_section['subsections'].append(current_subsection)
            current_subsection = {'title': line[3:], 'content': []}
        elif line.startswith('- '):  # サブセクションの内容
            if current_subsection is not None:
                current_subsection['content'].append(line[2:])
        else:
            continue  # 空行やその他の行は無視

    # 最後のセクションとサブセクションを追加
    if current_subsection:
        if current_section:
            current_section['subsections'].append(current_subsection)
    if current_section:
        sections.append(current_section)
    
    return sections

def add_search_content(model_name, title, sub_title, content):
    
    # {"role": "system", "content": f"Generate contexts that are different from the already existing {content}, and are related to {title} and {sub_title}."},
    
    prompt = [
        {"role": "system", "content": f"The title and its subtitles provide the context of the subtitles explaining them. Within this context, further investigative contexts are created that are related to the title and its subtitles."},
        {"role": "system", "content": "Generate contexts that are brief, about five words each, and no more than four types."},
        {"role": "user", "content": f"Title: {title}"},
        {"role": "user", "content": f"Sub_title: {sub_title}"},
        {"role": "user", "content": f"content: {content}"},
    ]
    
    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=TEMPERATURE,
    )
    
    search_content = response.choices[0].message.content
    
    return search_content

def fetch_text_from_url(url_link):
    try:
        response = requests.get(url_link, allow_redirects=True, timeout=10)
        # レスポンスのステータスコードが200以外の場合はエラーを表示して処理を終了
        if response.status_code != 200:
            return 'Error: Failed to retrieve the content'
        
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


def generate_search_content_summary(model_name, search_quary, content):
    
    prompt = [
        {"role": "system", "content": f"Please shorten the following sentences around the keyword description."},
        {"role": "user", "content": f"keyword: {search_quary}"},
        {"role": "user", "content": f"Sentence: {content}"},
        {"role": "user", "content": "Shortened sentences:"},
    ]
    
    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=TEMPERATURE,
    )
    
    short_sentence = response.choices[0].message.content
    
    return short_sentence

def update_subsections_with_descriptions(sections):
    url_link_list = []
    # 各サブセクションに説明を追加
    for section in sections:
        page_title = section['title']
        print("page_title", page_title)
        for subsection in section['subsections']:
            page_sub_title = subsection['title']
            print("page_sub_title", page_sub_title)
            # 既存のcontentリストに対して説明を追加（説明の例は仮のものです）
            updated_content = []
            for content in subsection['content']:
                content_text = content
                # 生成された質問のリストごとに検索を実施
                articles_q_content = []

                query_info = page_title + " " + page_sub_title + " " + content_text
                search_query = generate_search_queries(MODEL_NAME, query_info, "one")
                s_q = search_query
                # Wikipediaを除外する検索クエリの追加
                search_query += " -site:wikipedia.org"

                params = {'q': search_query, 'mkt': 'en', 'count': 3}
                headers = {'Ocp-Apim-Subscription-Key': api_key}
                r = requests.get(url, headers=headers, params=params)

                try:
                    # 検索結果を取得
                    results = r.json()['webPages']['value']
                except KeyError:
                    # 'webPages' キーが見つからない場合、このループの残りの処理をスキップし、次のループへ進む
                    continue
                # 結果を連結して回答を生成
                ans = ""
                links = []
                for result in results:
                    ans += result['snippet']
                    links.append(result['url'])

                articles_q_content.append({
                    'question' : s_q,
                    'answer' : ans,
                    'links' : links,
                })
                url_text = ""
                for article in articles_q_content:
                    url_links = article['links']
                    url_link_list.append(url_links)
                    for url_link in url_links:
                        url_text = url_text + "\n" + fetch_text_from_url(url_link)
                content_summary = generate_search_content_summary(MODEL_NAME, article['question'], url_text)

                content_description = f"{content}:\n{content_summary}"
                print(content_description)
                updated_content.append(content_description)
            subsection['content'] = updated_content
    return url_link_list, sections

def reconstruct_outline(sections):
    # 更新されたアウトラインを再構成
    outline_text = ""
    for section in sections:
        outline_text += f"# {section['title']}\n\n"
        for subsection in section['subsections']:
            outline_text += f"## {subsection['title']}\n"
            for content in subsection['content']:
                if content[0] == "-":
                    outline_text += f"{content}\n"
                else:
                    outline_text += f"- {content}\n"
            outline_text += "\n"
    return outline_text

def reconstruct_outline_japanese(sections):
    # 更新されたアウトラインを再構成
    outline_text = ""
    for section in sections:
        outline_text += f"# {section['title']}\n\n"
        print(f"# {section['title']}")
        for subsection in section['subsections']:
            outline_text += f"## {subsection['title']}\n"
            print(f"# {outline_text}")
            for content in subsection['content']:
                translated_outline = translate_to_japanese(MODEL_NAME, content)
                print(f"- {translated_outline}")
                outline_text += f"- {translated_outline}\n"
                
            outline_text += "\n"
    return outline_text

def remove_duplicates_keep_order(original_list):
    # seen = set()
    # unique_list = []
    # for item in original_list:
    #     if item not in seen:
    #         unique_list.append(item)
    #         seen.add(item)
    # return unique_list
    # 全リンクを一つのセットに統合する
    unique_links = set()
    for sublist in original_list:
        unique_links.update(sublist)  # 各サブリストのリンクを追加

    # セットをリストに変換
    consolidated_list = list(unique_links)
    
    return consolidated_list

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# def cosine_similarity(vector_a, vector_b):
#     dot_product = np.dot(vector_a, vector_b)
#     norm_a = np.linalg.norm(vector_a)
#     norm_b = np.linalg.norm(vector_b)
#     similarity = dot_product / (norm_a * norm_b)
#     # similarity = 0.5167918279102236
#     return similarity

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

app = Flask(__name__)
CORS(app)

# チャットの履歴を保存するリスト
chat_history = []
id_ = 0

@app.route('/reset', methods=['POST'])
def reset():
    global id_
    id_ = 0
    return jsonify(success=True)

@app.route('/chat', methods=['POST'])
def chat():
    global id_  # id_変数をグローバル変数として宣言
    data = request.json
    message = data['message']  # 入力メッセージを受け取る

    # ノードとエッジのリストを初期化
    nodes = []
    edges = []

    # 初期ノード (message) を追加
    current_message_id = id_ + 1
    nodes.append({'id': current_message_id, 'name': message, 'color': '#919191'})
    if current_message_id != 1:
        edges.append({'source': current_message_id, 'target': id_})
    intermediate_id = id_
    
    search_query = generate_search_queries(MODEL_NAME, message, "one")
    keywords_list = search_query.split('\n')
    articles_overview = []
    for keyword in keywords_list:
        article = get_wikipedia_articles_for_keywords(keyword, num_articles=1, lang='en')  # ここでは英語で検索しています
        articles_overview.append(article)
    
    # 記事から生成された質問を格納するリストを初期化
    all_questions = []

    # 各記事の要約に対して質問を生成
    # for article in articles_overview:
        # article = article[0]
    article = articles_overview[0][0]
    # get_wikipedia_articles_for_keywords中間ノードを追加
    # 中間ノードを追加
    intermediate_id = current_message_id + 1
    wikipedia_articles_for_keywords = current_message_id
    nodes.append({'id': intermediate_id, 'name': f"wiki articles summary for message\n{article['summary']}", 'color': '#7373a0'})
    edges.append({'source': current_message_id, 'target': intermediate_id})
    app.logger.debug(f"wiki articles summary for message\n{article['summary']}")

    # 各記事の要約に対して質問を生成
    questions = generate_wiki_questions(MODEL_NAME, article['summary'])
    all_questions.append({
        'title': article['title'],
        'questions': questions
    })
    current_message_id = intermediate_id

    # 各記事の要約に対して質問のまとめ
    # 生成された質問のリストごとに検索を実施
    # 生成された質問のリストごとに検索を実施
    articles_q = []

    for question in all_questions[0]['questions'].split("\n"):
        search_query = generate_search_queries(MODEL_NAME, question.split(": ")[-1], "one")
        
        # Wikipediaを除外する検索クエリの追加
        search_query += " -site:wikipedia.org"

        # params = {'q': question, 'mkt': 'en', 'count': 3}
        params = {'q': search_query, 'mkt': 'en', 'count': 1}
        headers = {'Ocp-Apim-Subscription-Key': api_key}
        r = requests.get(url, headers=headers, params=params)

        try:
            # 検索結果を取得
            results = r.json()['webPages']['value']
        except KeyError:
            # 'webPages' キーが見つからない場合、このループの残りの処理をスキップし、次のループへ進む
            continue
        ans = ""
        links = []
        for result in results:
            ans += result['snippet']
            links.append(result['url'])
        
        articles_q.append({
            'questions': question,
            'answer' : ans,
            'links' : links,
        })

    # 関連する検索結果のまとめ
    related_search_results = ""
    related_search_links = []
    next_node_id = []
    last_i = 0
    for i, article in enumerate(articles_q, start=1):
        related_search_results += article['answer']
        related_search_links.append(article['links'][0])

        # generate_wiki_questions中間ノードを追加
        # 中間ノードを追加
        intermediate_id = current_message_id + i
        next_node_id.append(intermediate_id)
        nodes.append({'id': intermediate_id, 'name': f"questions: {article['questions']}\nanswer: {article['answer']}", 'color': '#5454ac'})
        edges.append({'source': current_message_id, 'target': intermediate_id})
        app.logger.debug(f"questions: {article['questions']}\nanswer: {article['answer']}")
        last_i = i
    current_message_id = intermediate_id

    summary_text = articles_overview[0][0]['summary']
    outline_text = generate_wiki_outline(MODEL_NAME, summary_text, related_search_results)

    # 中間ノードを追加
    intermediate_id = current_message_id + 1
    nodes.append({'id': intermediate_id, 'name': f"outline\n{outline_text}...", 'color': '#3636b9'})
    edges.append({'source': wikipedia_articles_for_keywords+1, 'target': intermediate_id})
    
    # 最後のノードを対象に過去のノードからのエッジを追加
    for j in range(0, last_i):
        if current_message_id - j + 1 >= 0:  # 無効なIDを参照しないためのチェック
            edges.append({'source': current_message_id + ((-1) * j), 'target': intermediate_id})
    # edges.append({'source': current_message_id-4, 'target': intermediate_id})
    # edges.append({'source': current_message_id-3, 'target': intermediate_id})
    # edges.append({'source': current_message_id-2, 'target': intermediate_id})
    # edges.append({'source': current_message_id-1, 'target': intermediate_id})
    # edges.append({'source': current_message_id, 'target': intermediate_id})
    current_message_id = intermediate_id
    app.logger.debug(f"outline\n{outline_text}...")
    # 生成したアウトラインに詳細な説明を加える
    # detailed_outline = generate_detailed_outline(MODEL_NAME, outline_text, summary_text, related_search_results)
    sections = split_outline_into_sections(outline_text)
    # sections リストをループして各 section にアクセス
    for section in sections:
        print("Section:", section['title'])
        # 各 section の中の subsections リストをループして各 subsection にアクセス
        for subsection in section['subsections']:
            print("  Subsection:", subsection['title'])

            # subsection の現在の content を取得
            content_list = subsection.get('content', []).copy()

            # 追加するコンテンツを生成
            search_content = add_search_content(MODEL_NAME, section['title'], subsection['title'], str(content_list))
            print(search_content.split("\n"))
            
            # 追加するコンテンツを既存の content_list に統合
            content_list.extend(search_content.split("\n"))
            
            # 更新した content_list を subsection の content に代入
            subsection['content'] = content_list

            # 更新後の content を表示
            print("Updated Content:", subsection['content'])
    url_link_descript_list, updated_sections = update_subsections_with_descriptions(sections)
    # 更新されたアウトラインを再構成
    detailed_outline = reconstruct_outline(updated_sections)

    # 中間ノードを追加
    intermediate_id = current_message_id + 1
    nodes.append({'id': intermediate_id, 'name': f"add detailed outline\n{detailed_outline[:64]}...", 'color': '#1717c5'})
    edges.append({'source': current_message_id, 'target': intermediate_id})
    current_message_id = intermediate_id
    app.logger.debug(f"add detailed outline\n{detailed_outline[:64]}...")

    # translated_outline = translate_to_japanese(MODEL_NAME, detailed_outline)
    # 更新されたアウトラインを再構成
    translated_outline = reconstruct_outline_japanese(updated_sections)
    last_search_links = []
    last_search_links.append(articles_overview[0][0]['url'])

    # コンテキスト追加リンク
    for context_link in url_link_descript_list:
        last_search_links.append(context_link)

    for link in related_search_links:
        last_search_links.append(link)

    # 重複を削除
    last_search_links = remove_duplicates_keep_order(last_search_links)
    translated_outline += "\n\n## 参考リンク\n"

    # for link in last_search_links:
    #     translated_outline += "- " + str(link) + "\n"
    for link in last_search_links:
        if len(link) > 5:
            translated_outline += "- " + str(link) + "\n"

    # 最終ノード (final_message) を追加
    final_similarity = current_message_id + 1
    final_id = final_similarity + 3
    id_ = final_id
    final_message = translated_outline  # 最終メッセージは入力メッセージのコピー
    nodes.append({'id': final_id, 'name': f"{final_message[:64]}...", 'color': '#0f0f5d'})
    edges.append({'source': current_message_id, 'target': final_id})
    app.logger.debug(f"{final_message[:64]}...")

    # 評価結果
    message_embedding = get_embedding(message, model=MODEL_ENBEDDING_NAME)
    # translated_embedding_outline = get_embedding(translated_outline, model=MODEL_ENBEDDING_NAME)
    translated_embedding_outline = get_embedding(translated_outline[:7000], model=MODEL_ENBEDDING_NAME)
    cosine_similarity_result = cosine_similarity(np.array(message_embedding).reshape(1, -1), np.array(translated_embedding_outline).reshape(1, -1))
    nodes.append({'id': final_similarity, 'name': f"cosine similarity: {str(cosine_similarity_result)}", 'color': '#0f0f5d'})
    edges.append({'source': current_message_id, 'target': final_similarity})

    precision = soft_precision(translated_embedding_outline, message_embedding)
    nodes.append({'id': final_similarity+1, 'name': f"Soft precision: {str(precision)}", 'color': '#0f0f5d'})
    edges.append({'source': current_message_id, 'target': final_similarity+1})
    
    recall = soft_recall(translated_embedding_outline, message_embedding)
    nodes.append({'id': final_similarity+2, 'name': f"Soft recall: {str(recall)}", 'color': '#0f0f5d'})
    edges.append({'source': current_message_id, 'target': final_similarity+2})

    # 前回の最終メッセージを保存
    chat_history.append(final_message)
    return jsonify(nodes=nodes, links=edges, final_message=final_message)

if __name__ == '__main__':
    # ログレベルを設定 (例えば、デバッグ情報も出力する)
    app.logger.setLevel('DEBUG')
    # アプリケーションを実行
    app.run(debug=True, host='0.0.0.0', port=5000)
