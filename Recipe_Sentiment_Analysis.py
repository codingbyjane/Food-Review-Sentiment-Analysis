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
from transformers import pipeline

# NLP utilities
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


# Display the first few rows of the dataset to understand its structure
print(full_dataset_df.head())

# Check the distribution of ratings in the dataset to understand the balance of sentiment classes
print(full_dataset_df['rating'].value_counts(dropna=False))

# Check for reviews with 0 rating, which may indicate missing data
zero_reviews = full_dataset_df[full_dataset_df['rating'] == 0]
print(zero_reviews[['rating', 'review_text']].head()) # Note: the assumtion that 0 rating represent strong negative sentiment is false, because displayed 0 rated reviews contain positive comments


# Data Preprocessing: lowercasing, removing punctuation, stopwords, html artifacts, and tokenization
stopwords_set = set(stopwords.words('english'))

# Define a function to clean the review text
def preprocess_text(text): 

    # Ensure the input is a string
    if not isinstance(text, str):
        text = "" # Replace NaNs and other invalid value with an empty string to prevent errors during processing

    text = text.lower() # Convert to lowercase
    text = re.sub(r"<.*?>", "", text) # Remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", "", text) # Remove special characters
    text = re.sub(r"\d+", "", text) # Remove numbers
    tokens = word_tokenize(text) # Tokenization
    tokens = [word for word in tokens if word not in stopwords_set] # Remove stopwords using a list comprehension
    cleaned_text = " ".join(tokens) # Join tokens back into a single string by adding a space between them
    return cleaned_text


# Apply the function to the 'review_text' column of the DataFrame
full_dataset_df['cleaned_review_text'] = full_dataset_df['review_text'].apply(preprocess_text) # creates a new column 'cleaned_review_text'

# Remove the reviews with 0 rating, as they are missing data and could distort the sentiment analysis results
full_dataset_df = full_dataset_df[full_dataset_df['rating'] != 0]

# Normalize sentiment labels to three classes: positive (4-5 stars), neutral (3 stars), and negative (1-2 stars)
full_dataset_df['rating'] = full_dataset_df['rating'].replace({
    1 : "NEGATIVE",
    2 : "NEGATIVE",
    3 : "NEUTRAL",
    4 : "POSITIVE",
    5 : "POSITIVE"
})

# Convert the pandas DataFrame into a Hugging Face Dataset for easier integration with NLP models
full_dataset_hf = Dataset.from_pandas(full_dataset_df.reset_index(drop=True)) # Reset index to ensure a clean dataset without the old indexes

pizza_keywords = ['pizza', 'pepperoni', 'margherita', 'hawaiian'] # Define a list of keywords related to pizza to identify relevant reviews
sushi_keywords = ['sushi', 'sashimi', 'nigiri', 'maki'] # Define a list of keywords related to sushi to identify relevant reviews
ramen_keywords = ['ramen', 'noodles', 'broth', 'tonkotsu'] # Define a list of keywords related to ramen to identify relevant reviews

# Filtering the dataset to create susets for the startup's main products: pizza, suchi, and ramen
pizza_reviews = full_dataset_hf.filter(lambda review: any(keyword in review['cleaned_review_text'] for keyword in pizza_keywords))
ramen_reviews = full_dataset_hf.filter(lambda review: any(keyword in review['cleaned_review_text'] for keyword in ramen_keywords)) 
sushi_reviews = full_dataset_hf.filter(lambda review: any(keyword in review['cleaned_review_text'] for keyword in sushi_keywords))

print(len(pizza_reviews))  # How many reviews mention pizza
print(len(sushi_reviews))  # How many reviews mention sushi
print(len(ramen_reviews))  # How many reviews mention ramen

# Create a pipeline for sentiment analysis using a pre-trained BERT model from Hugging Face's Transformers library
pipe = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

pizza_review_result = pipe(list(pizza_reviews['cleaned_review_text']), truncation=True, max_length=512) # Analyze the sentiment of pizza reviews using the pipeline, truncating long reviews to fit the model's input size
sushi_review_result = pipe(list(sushi_reviews['cleaned_review_text']), truncation=True, max_length=512) # Analyze the sentiment of sushi reviews
ramen_review_result = pipe(list(ramen_reviews['cleaned_review_text']), truncation=True, max_length=512) # Analyze the sentiment of ramen reviews

# Convert the sentiment analysis results into pandas DataFrames for easier data manipulation and visualization
pizza_result_df = pd.DataFrame(pizza_review_result)
sushi_result_df = pd.DataFrame(sushi_review_result)
ramen_result_df = pd.DataFrame(ramen_review_result)

# Map the sentiment labels from the BERT model's star system to the original rating categories (POSITIVE, NEUTRAL, NEGATIVE)
label_mapping = {
    '1 star' : 'NEGATIVE',
    '2 stars': 'NEGATIVE',
    '3 stars': 'NEUTRAL',
    '4 stars': 'POSITIVE',
    '5 stars': 'POSITIVE'
}

# Apply mapping to each results DataFrame by adding a new column that contains the new sentiment labels
pizza_result_df['sentiment'] = pizza_result_df['label'].replace(label_mapping)
sushi_result_df['sentiment'] = sushi_result_df['label'].replace(label_mapping)
ramen_result_df['sentiment'] = ramen_result_df['label'].replace(label_mapping)

# Covert the Hugging Face Datasets back to pandas DataFrames for merging
pizza_reviews_df = pizza_reviews.to_pandas()
sushi_reviews_df = sushi_reviews.to_pandas()
ramen_reviews_df = ramen_reviews.to_pandas()

# Merge the sentiment results with the original reviews to create a clean DataFrame that contains both the original review text and the predicted sentiment labels for each product
pizza_reviews_df['sentiment'] = pizza_result_df['sentiment'] 
sushi_reviews_df['sentiment'] = sushi_result_df['sentiment']
ramen_reviews_df['sentiment'] = ramen_result_df['sentiment'] 

# Tagging each review with the corresponding product category for later analysis
pizza_reviews_df['product'] = 'Pizza'
sushi_reviews_df['product'] = 'Sushi'
ramen_reviews_df['product'] = 'Ramen'

# Merge all product reviews into a single DataFrame for cleaner analysis and visualization.
product_reviews_df = pd.concat([pizza_reviews_df, sushi_reviews_df, ramen_reviews_df], ignore_index=True)


# Isolate the positive & negative instances by filtering the 'sentiment' column
positive_reviews_df = product_reviews_df[product_reviews_df['sentiment'] == 'POSITIVE']
negative_reviews_df = product_reviews_df[product_reviews_df['sentiment'] == 'NEGATIVE']

print(f"Positive Reviews:\n{positive_reviews_df[['product', 'cleaned_review_text', 'sentiment']].head()}\n") # Display the first few positive reviews for each product
print(f"Negative Reviews:\n{negative_reviews_df[['product', 'cleaned_review_text', 'sentiment']].head()}\n") # Display the first few negative reviews for each product


# Create a general ingredient list by extracting commonly mentioned ingredients from the reviews
general_ingredient_list = [

    # Pizza ingredients
    'broccoli', 'corn', 'zucchini', 'red peppers', 'bacon', 'tomato sauce', 'olives', 'onions', 'mushrooms', 'spinach', 'garlic', 'basil', 'oregano', 'mozzarella', 'parmesan', 'cheese', 'tomato', 'pineapple', 'ham',

    # Sushi ingredients
    'soy sauce', 'wasabi', 'ginger', 'avocado', 'cucumber', 'carrot', 'cream cheese', 'tempura flakes', 'salmon', 'tuna', 'eel', 'shrimp', 'crab', 'seaweed', 'rice', 'nori', 

    # Ramen ingredients
    'pork', 'chicken', 'beef', 'tofu', 'egg', 'bok choy', 'miso', 'ramen noodles', 'chili oil', 'broth', 'scallions', 'chili', 'kimchi', 'sesame', 'tonkotsu', 'shoyu', 'shio'
]

# Define a function to loop over the review texts and check for the presence pf ingredients from the general ingredient list
def extract_ingredients(reviews,  ingredients):

    # Arguments:
    # Reviews (DataFrame): A DataFrame containing the reviews to analyze.
    # Ingredients (list): A list of ingredient strings to look for in the reviews.

    extracted_ingredients = []

    for review in reviews:
        # Defien an empty list to store the ingredients found in the current review
        found_in_review = []

        for ingredient in ingredients:

            # RegEx to ensure the function matches whole words only, preventing false positives (e.g., "ham" in "hamburger")
            pattern = r'\b' + re.escape(ingredient) + r'\b'

            if re.search(pattern, review):
                found_in_review.append(ingredient) # If the ingredient is found in the review, add it to the list of found ingredients for that review
            
        extracted_ingredients.append(found_in_review) # After checking all ingredients for the current review, add the list of found ingredients to the main list of extracted ingredients

    return extracted_ingredients

# Apply the defined function per product to extract the frequently mentioned ingredients in the reviews for each product category. Outputs ingredient mentions extracted per review, stored as nested lists
pizza_ingredients = extract_ingredients(pizza_reviews_df['cleaned_review_text'], general_ingredient_list)
sushi_ingredients = extract_ingredients(sushi_reviews_df['cleaned_review_text'], general_ingredient_list)
ramen_ingredients = extract_ingredients(ramen_reviews_df['cleaned_review_text'], general_ingredient_list)

pizza_ingredients_positive = extract_ingredients(positive_reviews_df[positive_reviews_df['product'] == 'Pizza']['cleaned_review_text'], general_ingredient_list) # Extract ingredients mentioned in positive pizza reviews
pizza_ingredients_negative = extract_ingredients(negative_reviews_df[negative_reviews_df['product'] == 'Pizza']['cleaned_review_text'], general_ingredient_list) # Extract ingredients mentioned in negative pizza reviews

sushi_ingredients_positive = extract_ingredients(positive_reviews_df[positive_reviews_df['product'] == 'Sushi']['cleaned_review_text'], general_ingredient_list)
sushi_ingredients_negative = extract_ingredients(negative_reviews_df[negative_reviews_df['product'] == 'Sushi']['cleaned_review_text'], general_ingredient_list)

ramen_ingredients_positive = extract_ingredients(positive_reviews_df[positive_reviews_df['product'] == 'Ramen']['cleaned_review_text'], general_ingredient_list)
ramen_ingredients_negative = extract_ingredients(negative_reviews_df[negative_reviews_df['product'] == 'Ramen']['cleaned_review_text'], general_ingredient_list)

# Define a function to flatten the nested lists of ingredients into a single list for easier frequency analysis
def flat_ingredient_list(nested_ingredients):
    # Define an empty list to store the flattened ingredients
    flat_list = []

    for ingredients in nested_ingredients:
        for ingredient in ingredients:
            flat_list.append(ingredient) # Add each ingredient from the nested lists to the flat list

    return flat_list

pizza_ingredients_list = flat_ingredient_list(pizza_ingredients) # Flatten the list of ingredients mentioned in pizza reviews
sushi_ingredients_list = flat_ingredient_list(sushi_ingredients) # Flatten the list of ingredients mentioned in sushi reviews
ramen_ingredients_list = flat_ingredient_list(ramen_ingredients) # Flatten the list of ingredients mentioned in ramen reviews

print(pizza_ingredients_list[:20]) # Display the first 10 ingredients mentioned in pizza reviews to verify the output