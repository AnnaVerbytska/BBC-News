# Import libraries
import pandas as pd
from google.cloud import bigquery #pip install google-cloud-bigquery for SQL
# Setting secret credentials
from dotenv import load_dotenv #pip install python-dotenv
import os
load_dotenv()

def get_bbc_news():
    """"
    This function fetches the BBC News dataset from BigQuery and returns it as a DataFrame.
    """
    # Setting up the credentials
    project = os.getenv('BBC_PROJECT_ID')
    # Initialize a BigQuery Client
    client=bigquery.Client(project=project)
    # Construct a reference to a dataset 
    dataset_ref=client.dataset('bbc_news', project='bigquery-public-data')
    # Fetch the dataset
    dataset=client.get_dataset(dataset_ref)
    # List all the tables
    tables=list(client.list_tables(dataset))
    # Print their names
    list_of_tables=[table.table_id for table in tables]
    print(list_of_tables)
    # Construct the reference to a table
    table_ref=dataset_ref.table('fulltext')
    # Get info on the columns
    table=client.get_table(table_ref)
    table.schema
    query = """
    SELECT *
    FROM  `bigquery-public-data.bbc_news.fulltext`
    ORDER BY title ASC
    """
    dry_run_config = bigquery.QueryJobConfig(dry_run=True)
    # Run dry run
    dry_run_query_job = client.query(query, job_config=dry_run_config)
    # Check the number of bytes the query would process
    print(f"This query will process {dry_run_query_job.total_bytes_processed} bytes.")
    safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = 10**10) # 1MB limit  # 10 GB limit - maximum_bytes_billed=10**10
    safe_query_job=client.query(query, job_config=safe_config)
    safe_query_job.to_dataframe()
    # Put the table to a DataFrame and preview
    bbc_news = client.list_rows(table).to_dataframe()
    return bbc_news

def bbc_news_politics(category="politics"):
    """"
    This function fetches the BBC News dataset on the political topic from BigQuery and returns it as a DataFrame.
    """
    # Setting up the credentials
    project = os.getenv('BBC_PROJECT_ID')
    # Initialize a BigQuery Client
    client = bigquery.Client(project=project)
    
    # Construct the SQL query with filtering
    query = f"""
    SELECT * 
    FROM `bigquery-public-data.bbc_news.fulltext`
    WHERE category = 'politics'
    ORDER BY title ASC
    """

    # Dry run to estimate data processed
    dry_run_config = bigquery.QueryJobConfig(dry_run=True)
    dry_run_query_job = client.query(query, job_config=dry_run_config)
    print(f"This query will process {dry_run_query_job.total_bytes_processed} bytes.")

    # Execute query with a safety limit (e.g., 10 GB)
    safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10) 
    query_job = client.query(query, job_config=safe_config)
    
    # Convert to DataFrame
    bbc_news_politics = query_job.to_dataframe()
    return bbc_news_politics