import pandas as pd
import re
import nltk
nltk.download('punkt')
from transformers import pipeline



def clean_text(text):
    """Preprocess text by removing extra spaces, line breaks, and special characters."""
    if isinstance(text, str):  # Check if the text is a string
        text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with one space
        return text.strip()
    return text  # If the text is not a string, return it as-is (e.g., handle NaN or non-string entries)

def drop_columns(df, columns):
    """Drops specified columns from a DataFrame."""
    return df.drop(columns=columns, errors='ignore')


def filter_articles_by_length(df):
    """Filters articles with word length between 150 and 1000."""
    df['word_count'] = df['body'].apply(lambda x: len(nltk.word_tokenize(x)))
    filtered_df = df[(df['word_count'] >= 150) & (df['word_count'] <= 1000)].reset_index(drop=True)
    return filtered_df

# Initialize BART summarization pipeline
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to summarize articles longer than 1000 words
def summarize_long_articles(df):
    """
    - Summarizes articles longer than 1000 words and replaces long text in 'body' column with summary.
    - Move articles that were not summarized to a new column 'body_shorter'.
    """
    
    # Create a new column for word count
    df['word_count'] = df['body'].apply(lambda x: len(nltk.word_tokenize(x)))
    
    # Apply BART summarization to long articles
    def summarize_article(text):
        return summarization_pipeline(text, max_length=1000, min_length=150, do_sample=False)[0]['shorter_text']
    # Replace 'body' with summary for long articles only
    long_mask = df['word_count'] > 1000
    df.loc[long_mask, 'body'] = df.loc[long_mask, 'body'].apply(summarize_article)

    return df