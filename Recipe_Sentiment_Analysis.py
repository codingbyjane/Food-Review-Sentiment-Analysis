# Importing necessary libraries

# Data manipulation & visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from venny4py.venny4py import venny4py  # For plotting Venn diagrams of sentiment overlaps between ingredients

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string

nltk.download('punkt')
nltk.download('stopwords')

# NLP & Deep Learning
import torch
from datasets import Dataset
from transformers import (BartTokenizer, BartForSequenceClassification, Trainer, TrainingArguments) # For BART-based sentiment analysis
from transformers import pipeline, AutoTokenizer, set_seed # For zero-shot classification and reproducibility

# NLP utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Dataset fetching
from ucimlrepo import fetch_ucirepo

recipe_reviews_and_user_feedback = fetch_ucirepo(id=911) # Importing the "Recipe Reviews and User Feedback" dataset from UCI Machine Learning Repository (UCI ID: 911)

# Access the data as a pandas DataFrame
reviews_df = recipe_reviews_and_user_feedback.data.features
sentiments_df = recipe_reviews_and_user_feedback.data.target

full_dataset_df = pd.concat([reviews_df, sentiments_df], axis=1) # Combine reviews and sentiments into a single DataFrame

# Renaming columns for clarity
full_dataset_df = full_dataset_df.rename(columns={'stars':'rating', 'text':'review_text'})

# Display the first few rows of the dataset
print(full_dataset_df.head(10))


# Data Preprocessing: lowercasing, removing punctuation, stopwords, html artifacts, and tokenization
stopwords_set = set(stopwords.words('english'))

# Define a function to clean the review text
def preprocess_text(text): 

    # Ensure the input is a string
    if not isinstance(text, str):
        text = "" # Replace with empty string

    text = text.lower() # Convert to lowercase
    text = re.sub(r"<.*?>", "", text) # Remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", "", text) # Remove special characters
    text = re.sub(r"\d+", "", text) # Remove numbers
    tokens = word_tokenize(text) # Tokenization
    tokens = [word for word in tokens if word not in stopwords_set] # Remove stopwords using a list comprehension
    cleaned_text = " ".join(tokens) # Join tokens back into a single string by adding a space between them
    return cleaned_text


# Apply the function to the 'review_text' column of the DataFrame
full_dataset_df['cleaned_review_text'] = full_dataset_df['review_text'].apply(preprocess_text)

# Normalize sentiment labels to three classes: positive (4-5 stars), neutral (3 stars), and negative (1-2 stars)
full_dataset_df['rating'] = full_dataset_df['rating'].replace({
    0 : "NEGATIVE",
    1 : "NEGATIVE",
    2 : "NEGATIVE",
    3 : "NEUTRAL",
    4 : "POSITIVE",
    5 : "POSITIVE"
})

print(full_dataset_df['rating'].value_counts())

# Convert the pandas DataFrame into a Hugging Face Dataset for easier integration with NLP models
full_dataset_hf = Dataset.from_pandas(full_dataset_df)