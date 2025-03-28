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
    # Initialize the 'summary' column with NaN or body text
    df['body_shorter'] = None  # or df['body'] to keep the original body text in summary for shorter articles
    # Identify long articles (more than 1000 words)
    long_articles = df[df['word_count'] > 1000].reset_index(drop=True)  # Reset index to preserve it as a column
    # Apply BART summarization to long articles
    def summarize_article(text):
        return summarization_pipeline(text, max_length=1000, min_length=150, do_sample=False)[0]['body_shorter_text']
    # Apply summarization to long articles and add the 'summary' column
    long_articles['body_shorter'] = long_articles['body'].apply(summarize_article) 
    # Replace the 'body' column for long articles with the summary
    df.loc[df['word_count'] > 1000, 'body'] = long_articles['body_shorter'] 
    # Fill 'summary' for articles that were not summarized (i.e., shorter articles) with the original 'body'
    df['body_shorter'] = df['body_shorter'].fillna(df['body'])  
    return df