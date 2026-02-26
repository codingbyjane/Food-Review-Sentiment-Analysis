# Importing necessary libraries

# Data manipulation & visualization
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter # For counting the frequency of ingredient mentions in reviews

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
print(zero_reviews[['rating', 'review_text']].head()) # Note: the assumption that 0 rating represents strong negative sentiment is false, because displayed 0 rated reviews contain positive comments

# Remove the reviews with 0 rating, as they are missing data and could distort the sentiment analysis results
full_dataset_df = full_dataset_df[full_dataset_df['rating'] != 0]

# display the total number of reviews in the dataset, excluding the 0 rated ones which are missing data
print(f"Total number of reviews in the dataset (excluding 0-rated reviews): {len(full_dataset_df)}")

# Data Preprocessing: lowercasing, removing punctuation, stopwords, html artifacts, and tokenization
stopwords_set = set(stopwords.words('english'))

# Remove "no" and "not" from the stopwords list to preserve negation in sentiment analysis, as they can significantly change the meaning of a review (e.g., "not good" vs "good")
stopwords_set.discard('no')
stopwords_set.discard('not')

# Define a function to clean the review text
def preprocess_text(text): 

    # Ensure the input is a string
    if not isinstance(text, str):
        text = "" # Replace NaNs and other invalid values with an empty string to prevent errors during processing

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

# Filtering the dataset to create subsets for the startup's main products: pizza, suchi, and ramen
pizza_reviews = full_dataset_hf.filter(lambda review: any(keyword in review['cleaned_review_text'] for keyword in pizza_keywords))
ramen_reviews = full_dataset_hf.filter(lambda review: any(keyword in review['cleaned_review_text'] for keyword in ramen_keywords)) 
sushi_reviews = full_dataset_hf.filter(lambda review: any(keyword in review['cleaned_review_text'] for keyword in sushi_keywords))

print(len(pizza_reviews))  # How many reviews mention pizza-related keywords
print(len(sushi_reviews))  # How many reviews mention sushi-related keywords
print(len(ramen_reviews))  # How many reviews mention ramen-related keywords

# Create a pipeline for sentiment analysis using a pre-trained BERT model from Hugging Face's Transformers library
pipe = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

pizza_review_result = pipe(list(pizza_reviews['cleaned_review_text']), truncation=True, max_length=512) # Analyze the sentiment of pizza reviews using the pipeline, truncating long reviews to fit the model's input size, and setting a maximum length of 512 tokens
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

# Tagging each review with the corresponding product category for later accessing
pizza_reviews_df['product'] = 'Pizza'
sushi_reviews_df['product'] = 'Sushi'
ramen_reviews_df['product'] = 'Ramen'

# Merge all product reviews into a single DataFrame for cleaner analysis and visualization.
product_reviews_df = pd.concat([pizza_reviews_df, sushi_reviews_df, ramen_reviews_df], ignore_index=True) # Contains only reviews that mention the products, along with their predicted sentiment labels and product tags


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
    'pork', 'chicken', 'beef', 'tofu', 'egg', 'bok choy', 'miso', 'ramen', 'chili oil', 'broth', 'scallions', 'chili', 'kimchi', 'sesame', 'tonkotsu', 'shoyu', 'shio', 'noodles'
]

# Define a function to loop over the review texts and check for the presence of ingredients from the general ingredient list
def extract_ingredients(reviews,  ingredients):

    # Arguments:
    # Reviews (DataFrame): A DataFrame containing the reviews to analyze.
    # Ingredients (list): A list of ingredient strings to look for in the reviews.

    extracted_ingredients = []

    for review in reviews:
        # Define an empty list to store the ingredients found in the current review
        found_in_review = []

        for ingredient in ingredients:

            # RegEx to ensure the function matches whole words only, preventing false positives (e.g., "ham" in "hamburger")
            pattern = r'\b' + re.escape(ingredient) + r'\b'

            if re.search(pattern, review):
                found_in_review.append(ingredient) # If the ingredient is found in the review, add it to the list of found ingredients for that review
            
        extracted_ingredients.append(found_in_review) # After checking all ingredients for the current review, add the list of found ingredients to the main list of extracted ingredients

    return extracted_ingredients

# Apply the defined function per product category to extract the frequently mentioned ingredients in the positive and negative reviews for each product category
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


pizza_ingredients_positive_list = flat_ingredient_list(pizza_ingredients_positive) # Flatten the list of ingredients mentioned in positive pizza reviews
pizza_ingredients_negative_list = flat_ingredient_list(pizza_ingredients_negative) # Flatten the list of ingredients mentioned in negative pizza reviews. 

# Analogously flatten the lists of ingredients mentioned in positive and negative reviews for sushi and ramen
sushi_ingredients_positive_list = flat_ingredient_list(sushi_ingredients_positive)
sushi_ingredients_negative_list = flat_ingredient_list(sushi_ingredients_negative)

ramen_ingredients_positive_list = flat_ingredient_list(ramen_ingredients_positive)
ramen_ingredients_negative_list = flat_ingredient_list(ramen_ingredients_negative)

# Calculate the frequnency of each ingredient mentioned in the reviews for each product category using the Counter class from the collections module
pizza_pos_ingredient_frequency = Counter(pizza_ingredients_positive_list)
pizza_neg_ingredient_frequency = Counter(pizza_ingredients_negative_list)

sushi_pos_ingredient_frequency = Counter(sushi_ingredients_positive_list)
sushi_neg_ingredient_frequency = Counter(sushi_ingredients_negative_list)

ramen_pos_ingredient_frequency = Counter(ramen_ingredients_positive_list)
ramen_neg_ingredient_frequency = Counter(ramen_ingredients_negative_list)


# Create a general list of ingredients uniquely labeled positive across all product categories
uniquely_positive_ingredients = extract_ingredients(product_reviews_df[product_reviews_df['sentiment'] == 'POSITIVE']['cleaned_review_text'], general_ingredient_list) # Extract ingredients mentioned in all positive reviews across products)
uniquely_negative_ingredients = extract_ingredients(product_reviews_df[product_reviews_df['sentiment'] == 'NEGATIVE']['cleaned_review_text'], general_ingredient_list) # Extract ingredients mentioned in all negative reviews across products)

uniquely_negative_ingredients_set = set(flat_ingredient_list(uniquely_negative_ingredients)) 
uniquely_positive_ingredients_set = set(flat_ingredient_list(uniquely_positive_ingredients))

print(f"Uniquely Positive Ingredients:\n{uniquely_positive_ingredients_set}\n") 
print(f"Uniquely Negative Ingredients:\n{uniquely_negative_ingredients_set}\n") 

# Identify the ingredients that are truly positive (mentioned only in positive reviews) and truly negative (mentioned only in negative reviews) by taking the set difference
truly_positive_ingredients = uniquely_positive_ingredients_set - uniquely_negative_ingredients_set
truly_negative_ingredients = uniquely_negative_ingredients_set - uniquely_positive_ingredients_set

print(f"Truly Positive Ingredients:\n{truly_positive_ingredients}\n")
print(f"Truly Negative Ingredients:\n{truly_negative_ingredients}\n")


# Define the top 10 most frequently mentioned ingredients in positive and negative reviews for each product category. The set wrapper ensures there are no duplicates, the ingredient output of the list comprehension is used to extract only the ingredient names, ignoring the counts
top10_pizza_pos_ingredients = set(ingredient for ingredient, count in pizza_pos_ingredient_frequency.most_common(10))
top10_pizza_neg_ingredients = set(ingredient for ingredient, count in pizza_neg_ingredient_frequency.most_common(10))

top10_sushi_pos_ingredients = set(ingredient for ingredient, count in sushi_pos_ingredient_frequency.most_common(10)) 
top10_sushi_neg_ingredients = set(ingredient for ingredient, count in sushi_neg_ingredient_frequency.most_common(10))

top10_ramen_pos_ingredients = set(ingredient for ingredient, count in ramen_pos_ingredient_frequency.most_common(10))
top10_ramen_neg_ingredients = set(ingredient for ingredient, count in ramen_neg_ingredient_frequency.most_common(10))


top_pizza_ingredients = list(top10_pizza_neg_ingredients.union(top10_pizza_pos_ingredients)) # Create a list of the top pizza ingredients by taking the union of the positive and negative sets
top_sushi_ingredients = list(top10_sushi_neg_ingredients.union(top10_sushi_pos_ingredients)) # Create a list of the top sushi ingredients
top_ramen_ingredients = list(top10_ramen_neg_ingredients.union(top10_ramen_pos_ingredients)) # Create a list of the top ramen ingredients

pizza_positive_counts = [pizza_pos_ingredient_frequency.get(ingredient, 0) for ingredient in top_pizza_ingredients] # Get the frequency counts for the top pizza ingredients in positive reviews
pizza_negative_counts = [-pizza_neg_ingredient_frequency.get(ingredient, 0) for ingredient in top_pizza_ingredients] # Get the frequency counts for the top pizza ingredients in negative reviews. The negative sign is used to plot the negative counts on the left side of the diverging bar chart

sushi_positive_counts = [sushi_pos_ingredient_frequency.get(ingredient, 0) for ingredient in top_sushi_ingredients]
sushi_negative_counts = [-sushi_neg_ingredient_frequency.get(ingredient, 0) for ingredient in top_sushi_ingredients]

ramen_positive_counts = [ramen_pos_ingredient_frequency.get(ingredient, 0) for ingredient in top_ramen_ingredients]
ramen_negative_counts = [-ramen_neg_ingredient_frequency.get(ingredient, 0) for ingredient in top_ramen_ingredients]


# Defining a function for plotting a diverging bar chart to visualize the frequency of top ingredients mentioned in positive and negative reviews for each product category
def plot_bar_chart(ingredients, positive_counts, negative_counts, product_category):

    # Arguments:
    # Ingredients (list): A list of ingredient names to plot on the y-axis.
    # Positive_counts (list): A list of frequency counts for the positive mentions of each ingredient.
    # Negative_counts (list): A list of frequency counts for the negative mentions of each ingredient.
    # Product category (string): 'Pizza', 'Sushi', & 'Ramen'  

    plt.figure(figsize=(12, 8))

    # Plot the negative bars (left side)
    plt.barh(ingredients, negative_counts, color='red', label='Negative Reviews')

    # Plot the positive bars (right side)
    plt.barh(ingredients, positive_counts, color='green', label='Positive Reviews')

    plt.xlabel(f"Frequency of {product_category} Ingredient Mentions")
    plt.title(f"Top {product_category} Ingredients in Positive vs Negative Reviews")
    plt.legend()

    plt.axvline(0) # Add a vertical center line to separate positive and negative sides of the chart
    plt.show()

plot_bar_chart(top_pizza_ingredients, pizza_positive_counts, pizza_negative_counts, 'Pizza') # Plot the bar chart for pizza ingredients
plot_bar_chart(top_sushi_ingredients, sushi_positive_counts, sushi_negative_counts, 'Sushi') # Plot the bar chart for sushi ingredients
plot_bar_chart(top_ramen_ingredients, ramen_positive_counts, ramen_negative_counts, 'Ramen') # Plot the bar chart for ramen ingredients