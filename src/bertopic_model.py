# main.py
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from query import bbc_news_politics
import joblib
from dotenv import load_dotenv
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Fix for Hugging Face Tokenizers issue

# Load environment variables
load_dotenv()

# Fetch data from BigQuery
df = bbc_news_politics()

# Pre-compute embeddings
embedding_model = SentenceTransformer("all-MiniLM-L12-v2")
embeddings = embedding_model.encode(df['body'], show_progress_bar=False)

# Initialize UMAP, HDBSCAN, and BERTopic parameters
umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
representation_model = MaximalMarginalRelevance(diversity=0.5, top_n_words=15)

# Initialize BERTopic model
topic_model = BERTopic(
  embedding_model=embedding_model,
  umap_model=umap_model,
  hdbscan_model=hdbscan_model,
  vectorizer_model=vectorizer_model,
  ctfidf_model=ctfidf_model,
  representation_model=representation_model,
  nr_topics=None,
  min_topic_size=20,
  verbose=True,
  top_n_words=15
)

# Fit the model
topics, probabilities = topic_model.fit_transform(df['body'], embeddings)

# Add a new column filled with topics
df['topic'] = topics
print(df.head(5))

# Print number of topics and keywords
freq = topic_model.get_topic_info()
print(f"Number of topics: {len(freq)}")
print(freq.head(15))

# Save DataFrame as CSV in the data folder
df.to_csv("../data/news_politics_topics.csv", index=False)
print("DataFrame saved successfully!")

# Visualize topic keywords and save as png
fig1 = topic_model.visualize_barchart(n_words=10)
pio.write_image(fig1, '../plots/topic_keywords_barchart.png')  
fig2 = topic_model.visualize_topics(title="Topic Clusters Visualization")
pio.write_image(fig2, '../plots/topic_clusters.png') 
fig3 = topic_model.visualize_hierarchy(top_n_topics=11)
pio.write_image(fig3, '../plots/topic_hierarchy.png') 
fig4 = topic_model.visualize_heatmap(top_n_topics=30)
pio.write_image(fig4, '../plots/topic_similarity_heatmap.png') 

# Save the BERTopic model for future use. Note: it takes a lot of space to upload on GitHub
# joblib.dump(topic_model, '../models/bertopic_model.pkl')
# print("Model saved as 'bertopic_model.pkl'")