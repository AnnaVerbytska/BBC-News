# 📰 Topic Modeling & Aspect-Based Sentiment Analysis of BBC News

![Python](https://img.shields.io/badge/Python-3.12.6-blue)
![BERTopic](https://img.shields.io/badge/BERTopic-MiniLM--L12--v2-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

🚀 **Analyzing political news from the BBC using Topic Modeling and Sentiment Analysis.**  
We use **BERTopic** with `all-MiniLM-L12-v2` embeddings from HuggingFace to uncover hidden topics and extract insights from the text.

---

## 📌 **Project Overview**
- **Data Source:** BBC News articles from **BigQuery** (`bigquery-public-data`)  
- **Models Used:** `BERTopic`, `all-MiniLM-L12-v2`
- **Visualization:** Interactive topic visualizations (barcharts, clusters, heatmaps)
- **Text Summarisation:** `facebook/bart-large-cnn`
            - *Article-Level Summarisaiton:* summarised very long articles to avoid BERTopic bias
            - *Topic-Level Summarisation:* summarised bunches of articles categories as certain topics to label them. 
- **Named Entity Recognition:** `distilBERT`  
- **Sentiment Analysis:** Aspect-based sentiment evaluation of political entities where **aspect labels = topic labels**

---

## ⚡ **Quick Setup**
```bash
pip install -r requirements.txt

## **Get Topics**
cd src/
python main.py 

