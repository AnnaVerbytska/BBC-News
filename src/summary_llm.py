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


#####################################################
# Article-Level Summarization by Gemini 2.0 Falsh #
####################################################

# Summarize an article using Gemini 2.0 Flash
def summarize_article(text):
    """
    Summarizes the article text using Gemini 2.0 Flash.
    
    Args:
        text (str): The article text to be summarized.
        max_output_tokens (int): The maximum number of tokens for the summary.
    
    Returns:
        str: The summarized text.
    """
    try:
        # Prepare the prompt for summarization
        prompt = f"Summarize the following article into a concise summary:\n\n{text}"

        config = types.GenerateContentConfig(temperature=0.2, max_output_tokens=200)

        # Call the Gemini API to generate the summary
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=config        
            )

        # Return the summary
        return response.text.strip()

    except Exception as e:
        print(f"Error during summarization: {e}")
        return ""

# Main function to apply summarization to all articles in a DataFrame
def summarize_articles_in_df(df, body_col='body'):
    """
    Summarizes the articles in the specified DataFrame column and adds the summaries to a new column.

    Args:
        df (pd.DataFrame): DataFrame with the articles.
        body_col (str): The name of the column containing article text to summarize.
    
    Returns:
        pd.DataFrame: DataFrame with a new 'article_summary' column containing the summaries.
    """
    # Apply the summarization function to each article in the 'body' column
    df['gemini_summary'] = df[body_col].apply(lambda text: summarize_article(text))

    return df

# Fetch the data 
df = pd.read_csv("../data/topics_summaries.csv")

# Summarize the articles in the 'body' column (Run time - mins)
df = summarize_articles_in_df(df)
df[['body', 'gemini_summary']].head(10)

# Save the updated DataFrame with summaries
df.to_csv("topics_summaries.csv", index=False)
print("✅ DataFrame with article summaries saved successfully!")