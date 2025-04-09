import pandas as pd
import re
import nltk
nltk.download('punkt')
from transformers import pipeline
from tqdm import tqdm



def clean_text(text):
    """Preprocess text by removing extra spaces, line breaks, and special characters."""
    if isinstance(text, str):  # Check if the text is a string
        text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with one space
        return text.strip()
    return text  # If the text is not a string, return it as-is (e.g., handle NaN or non-string entries)

def drop_columns(df):
    """Drops specified columns from a DataFrame."""
    return df.drop(columns=['filename'], errors='ignore')


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

# Function to summarize articles and create a new column for summaries
def summarize_articles_bart(df, text_column='body', summary_column='article_summary', max_length=300, min_length=100):
    """
    Summarizes articles using facebook/bart-large-cnn and stores the result in a new column.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the articles.
        text_column (str): The name of the column with the full article text.
        summary_column (str): The name of the new column to store the summaries.
        max_length (int): Max token length for the summary.
        min_length (int): Min token length for the summary.
    
    Returns:
        pd.DataFrame: The DataFrame with the added summary column.
    """
    # Load summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    summaries = []
    for text in tqdm(df[text_column].fillna("").tolist(), desc="Summarizing articles"):
        if isinstance(text, str) and len(text.strip()) > 50:
            try:
                summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            except Exception as e:
                print(f"Summarization failed: {e}")
                summary = ""
        else:
            summary = ""
        summaries.append(summary)

    # Add summary column to the DataFrame
    df[summary_column] = summaries
    return df