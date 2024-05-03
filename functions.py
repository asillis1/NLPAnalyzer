# File for defining functions used to run the app

################## VALIDATION FUNCTIONS

#### Need to select a value in each dropdown



#### If state, need to have a minimum of X records



#### If aggregate, need to have a minimum of X records








##################### DATA CLEANING FUNCTIONS


#### If uploading multiple files (how many files will we allow?)

# Check if same headers
# Throw error if not 


# If same headers, concat dataframe to one big dataframe



# Generate dataframe with review, checkout_date
def dataframe(data): 
    reviews = data(subset=['review', 'checkout_date'], inplace=True)
    data = reviews
    return data

# Remove duplicates 
def cleaning(data):
    data = data.dropna(subset=['review'])
    data = data.drop_duplicates(subset=['review', 'checkout_date'], keep='first')
    return data

##################### TOPIC SEGMENTATION

# Create segments from reviews 
import pandas as pd
import spacy

# Load the English language model in spacy
nlp = spacy.load("en_core_web_sm")

def segment_reviews(reviews_df):
    # Function to split review into sentences and return a list of tuples (sentence, checkout_date)
    def split_review_into_sentences(review, checkout_date):
        doc = nlp(review)
        return [(sent.text.strip(), checkout_date) for sent in doc.sents]

    # Initialize a list to hold the segmented reviews
    segmented_reviews = []

    # Iterate over each row in the DataFrame and split reviews into sentences
    for index, row in reviews_df.iterrows():
        segmented_reviews.extend(split_review_into_sentences(row['review'], row['checkout_date']))

    # Create a new DataFrame from the segmented reviews
    segments_df = pd.DataFrame(segmented_reviews, columns=['segment', 'checkout_date'])

    # Drop duplicates based on 'segment' and 'checkout_date'
    segments_df.drop_duplicates(subset=['segment', 'checkout_date'], inplace=True)

    # Remove entries with empty strings or NaNs for the segment
    segments_df = segments_df[segments_df['segment'].notna() & (segments_df['segment'] != '')]

    # Merge original reviews with segments based on 'checkout_date'
    merged_df = pd.merge(segments_df, reviews_df, how='left', left_on='checkout_date', right_on='checkout_date')

    # Reorder columns and rename 'review' column
    merged_df = merged_df[['review', 'segment', 'checkout_date']]
    merged_df.rename(columns={'review': 'Review'}, inplace=True)

    return merged_df

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing configurations
contraction_mapping = {
    "didn't": "did not",
    "don't": "do not",
    "aren't": "are not",
    "couldn't": "could not",
}

# Update the stopwords set by removing negations important for sentiment analysis
custom_stopwords = set(stopwords.words('english')) - {"no", "not", "don't", "aren't", "couldn't", "didn't"}

# Function to clean text data
def clean_text(text):
    # Expand contractions
    for contraction, expansion in contraction_mapping.items():
        text = re.sub(r"\b{}\b".format(contraction), expansion, text)
    # Remove non-alphanumeric characters, preserving apostrophes
    text = re.sub(r"[^\w\s']", ' ', text)
    # Lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Removing stopwords
    tokens = [word for word in tokens if word not in custom_stopwords]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

