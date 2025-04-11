# 📰 Topic Modeling & Aspect-Based Sentiment Analysis of BBC News

![Python](https://img.shields.io/badge/Python-3.12.6-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

🚀 **Analyzing political news from the BBC using Topic Modeling and Sentiment Analysis.**  
We use **BERTopic** with `all-MiniLM-L12-v2` embeddings from HuggingFace to uncover hidden topics and extract insights from the text.

---

## 📌 **Project Overview**
- **Data Source:** BBC News articles from **BigQuery** (`bigquery-public-data`)  
- **Models Used:** `BERTopic`, `all-MiniLM-L12-v2`, `facebook/bart-large-cnn`, `gemini-2.0-flash`
- **Visualization:** Interactive topic visualizations (barcharts, clusters, heatmaps)
- **Text Summarisation:**  
            - summarised very long articles to avoid BERTopic bias with `facebook/bart-large-cnn`
            - summarised articles with `facebook/bart-large-cnn` & `gemini-2.0-flash` for comparison
            
    - *Topic-Level Summarisation:* labelled topics with `gemini-2.0-flash`
 
- **Named Entity Recognition** extraction of entities with RoBERTa and spaCy
- **Sentiment Analysis:** Aspect-based sentiment evaluation of political entities where **aspect labels = topic labels** 
- **Reference dataset**: [![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-fhamborg/news_sentiment_newsmtsc-yellow)](https://huggingface.co/datasets/fhamborg/news_sentiment_newsmtsc)
- **Sentiment Labels**: Positive, Negative, Neutral

---

## ⚡ **Quick Setup**

    pip install -r requirements.txt

#### **Get Topics & Summaries by BART**

    cd src/
    python main.py

---

#### **Get Summaries by Gemini**
    cd src/
    python summary_llm.py

#### **Get Named Entities**

    pip install -r ner_requirements.txt
    cd src/
    python entities.py


