import pandas as pd
import os
from google import genai
from dotenv import load_dotenv #pip install python-dotenv
from google.genai import types
load_dotenv()  # Loads variables from .env


from google.api_core import retry
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
genai.models.Models.generate_content = retry.Retry(
    predicate=is_retriable)(genai.models.Models.generate_content)

# Set up your API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)

def generate_topic_label(topic_id, summaries):
    prompt = f"""You are given a list of summaries of news articles that all belong to the same topic.
Return a concise, human-readable label (2–4 words) that best describes the overall topic.
Here are the summaries:
{summaries}
"""
    config = types.GenerateContentConfig(temperature=0.2, max_output_tokens=50)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=config
    )
    return response.text.strip()

# Generate topic labels
topic_labels = {}
for topic_id in df['topic'].unique():
    summaries = "\n".join(df[df['topic'] == topic_id]['gemini_summary'].dropna().tolist()[:20])  # optionally limit to 20
    label = generate_topic_label(topic_id, summaries)
    topic_labels[topic_id] = label

# Fetch the data 
df = pd.read_csv("../data/topics_summaries.csv")

# Map labels to a new column
df['topic_label'] = df['topic'].map(topic_labels)
print(df[['topic', 'topic_label']].head(10))

# Save the updated DataFrame with summaries
df.to_csv("../data/topics_summaries.csv", index=False)
print("✅ DataFrame with topic labels saved successfully!")

