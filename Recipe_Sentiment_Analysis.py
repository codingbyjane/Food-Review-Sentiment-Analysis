# Importing necessary libraries

# Data manipulation & visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from venny4py.venny4py import venny4py  # For plotting Venn diagrams of sentiment overlaps between ingredients

# NLP & Deep Learning
import torch
import nltk
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

full_dataset = pd.concat([reviews_df, sentiments_df], axis=1) # Combine reviews and sentiments into a single DataFrame

# Display the first few rows of the dataset
print(full_dataset.head(20))

# Renaming columns for clarity
full_dataset = full_dataset.rename(columns={'stars':'rating', 'text':'review_text'})