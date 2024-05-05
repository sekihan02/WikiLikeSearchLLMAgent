# WikiLikeSearchLLMAgent

WikiLikeSearchLLMAgentは、Wikipediaの記事風の調査報告書をWikipedia記事とBing検索、`gpt-3.5-turbo-0125`のLLMモデルを使用して作成するLLMエージェントです。

![AutoResearcherGraphLLM_general_report.gif](./AutoResearcherGraphLLM_general_report.gif)

途中の出力結果は力学グラフで出力します。
評価として`text-embedding-3-small`を使用して入力内容と最終的な報告書の出力結果を埋め込みベクトル化して`cosine similarity`, `Soft Precision`, `Soft Recall`を算出しています


---

WikiLikeSearchLLMAgent is an LLM agent designed to create research reports in the style of Wikipedia articles, utilizing data from Wikipedia entries, Bing searches, and the `gpt-3.5-turbo-0125` LLM model.

![AutoResearcherGraphLLM_general_report.PNG](./AutoResearcherGraphLLM_general_report.PNG)


The intermediate outputs are visualized using dynamic graphs. For evaluation, the project employs the `text-embedding-3-small` model to vectorize both the input data and the final report outputs, calculating metrics such as cosine similarity, Soft Precision, and Soft Recall to assess the quality and relevance of the generated content.

---

## 主な機能

- 応答の生成: app.py では、入力した調査報告書を作成したい内容について検索クエリを作成。その内容をWikipediaで検索し、概要を取得。概要から質問を3つ程度作成。
作成した質問に対して検索クエリを作り、Bing検索します。
Wikipediaの検索、Bing検索の結果を使用してWikipedia記事風のアウトラインを作成、アウトラインを説明するテキストを追加作成して日本語に翻訳したものを最終的な調査報告書として出力します。
- グラフによるインタラクションの可視化: 入力されたメッセージと最終的な出力以外の途中の生成結果は力学グラフで出力します。

## 参考内容

- [Graph Chain-of-Thought: Augmenting Large Language Models by Reasoning on Graphs](https://arxiv.org/abs/2404.07103)
- [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/abs/2402.14207)
- [Soft precision and recall](https://www.sciencedirect.com/science/article/pii/S0167865523000296)

---

## Key Features

- Generating Responses: In app.py, a search query is crafted based on the input about the desired research report content. This content is searched on Wikipedia to gather summaries and from these summaries, about three questions are generated. These questions are then used to formulate search queries for Bing. Using the results from both Wikipedia and Bing searches, a Wikipedia-style article outline is created, and additional explanatory text is added. This content is then translated into Japanese to produce the final research report.
- Visualization of Interactions through Graphs: The intermediate outputs, apart from the initial input message and the final output, are visualized using dynamic graphs.

---


## アクセスポイント

- フロントエンド: ユーザーインターフェースは http://localhost:3000 でアクセス可能です。ここからユーザーはメッセージを送信し、グラフの可視化を見ることができます。
- バックエンド: サーバーは http://localhost:5000 で稼働しており、フロントエンドからのリクエストを処理します。具体的には、http://localhost:5000/chat にPOSTリクエストを送ることで、ユーザーからのメッセージが処理され、応答が返されます。

---

## Access Points

- Frontend: The user interface is accessible at http://localhost:3000, where users can send messages and view the graphical visualization of the interactions.
- Backend: The server operates at http://localhost:5000, handling requests from the frontend. Specifically, POST requests sent to http://localhost:5000/chat process user messages and return responses.

## 動作イメージ
## Functional Image

![](./EchoChatGraph.gif)


- ファイル構成は以下です
- File Structure

```
WikiLikeSearchLLMAgent/
│  docker-compose.yml
│  Dockerfile
│  package.json
│  README.md
│
├─backend
│      .env
│      .prettierrc
│      app.py
│      Dockerfile
│      requirements.txt
│
└─frontend
    │  config-overrides.js
    │  Dockerfile
    │  package.json
    │  tsconfig.json
    │
    ├─public
    │      index.html
    │
    └─src
            App.css
            App.tsx
            ForceGraph.js
            index.css
            index.tsx
```

.envには、OPENAI_API_KEYとBING_API_KEYをそれぞれ設定してください

Please ensure the following keys are set in your .env file:

```
OPENAI_API_KEY=****
BING_API_KEY=****
```
