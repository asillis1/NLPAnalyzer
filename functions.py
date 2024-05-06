import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from itertools import chain
import nltk
nltk.download('vader_lexicon')

# File for defining functions used to run the app

################## VALIDATION FUNCTIONS

#### Need to select a value in each dropdown


##################### DATA CLEANING FUNCTIONS

#### If uploading multiple files (we allow up to 3)

def merge_csv_files(files):
    data_frames = []
    for file in files:
        # Read each file into a DataFrame
        df = pd.read_csv(file)
        # Select only the necessary columns
        df = df[['checkout_date', 'public_review', 'private_feedback']]
        data_frames.append(df)

    # Concatenate all DataFrames
    if data_frames:
        merged_df = pd.concat(data_frames, ignore_index=True)
        return merged_df
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no files were provided

#### Public, Private, All

def process_files(data, review_type):
    try:
        # Always retain 'checkout_date'
        if review_type == 'Public':
            # Keep 'public_review' and rename it to 'review'
            data = data[['checkout_date', 'public_review']]
            data.rename(columns={'public_review': 'review'}, inplace=True)
        elif review_type == 'Private':
            # Keep 'private_feedback' and rename it to 'review'
            data = data[['checkout_date', 'private_feedback']]
            data.rename(columns={'private_feedback': 'review'}, inplace=True)
        else:
            # Keep both and collate under 'review'
            public = data[['checkout_date', 'public_review']].rename(columns={'public_review': 'review'})
            private = data[['checkout_date', 'private_feedback']].rename(columns={'private_feedback': 'review'})
            data = pd.concat([public, private])

        return data
    except KeyError as e:
        return pd.DataFrame()  # Return an empty DataFrame on KeyError
    except Exception as e:
        return pd.DataFrame()  # General error catch returns an empty DataFrame

#### Remove duplicates 
def cleaning(data):
    data = data.dropna(subset=['review'])
    data = data.drop_duplicates(subset=['review', 'checkout_date'], keep='first')
    return data

##################### TOPIC SEGMENTATION

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

# Categories and their associated keywords
categories = {
    "communication": ["communicat", "contact", "responsive", "response", "reply", "communication", "host", "hospitality", "professional", "responsive", "responded", "request", "getting back", "respond", "available", "kind", "customer service", "accommodating", "helpful"],
    "location": ["accessible", "location", "area", "place", "stay" , "accommodation", "spacious", "setting", "walk", "walkability", "parking", "driving", "drive", "biking", "located", "close", "near", "far", "cozy", "local", "neighbourhood", "neighborhood", "across the street", " view", "next to", "market", "cafe"],
    "cleanliness": ["well kept", "clean", "tidy", "hygiene", "dirty", "cleanliness", "gross", "housekeeping", "stain", "spotless", "beach", "crumb", "dirt", "dust", "bug", "wash", "ant", "maintainence", "infest", "spotless", "sanitized", "sanitary"],
    "accuracy": ["accurate", "exact", "description", "expectation", "picture", "catfish", "described", "space", "money", "expensive", "room", "commensurate", "price", "temperature", "comfortable", "uncomfortable", "photo", "mislead", "luxury", "decorated", "decoration", "advertised", "outside", "interiors", "size", "backyard", "yard", "photos", "incorrect", "inform", "wrong", "instructions"],
    "check-in": ["check-in", "arrival", "welcome", "attention", "checkin", "check in", "check in", "checked", "entry", "keypad", "code", "key"],
    "amenities": ["towels", "blinds", "coffee machine", "kitchenware", "oven", "utensil", "accommodations", "alarms", "tv", "t.v", "tools", "coffee", "tea", "toilet paper", "laundry", "linen", "toiletries", "facilit", "stocked", "equipped", "heater", "refrigerator", "couch", "fridge", "amenities", "water", "thermostat", "broken", "dish soap", "dishwasher", "dishes", "furnishing", "furnished", "interiors", " ac ", "a/c", "a.c.", "air conditioning", "ice maker", "gear", "garbage", "trash", "internet", "blanket", "towel", "shower", "washer", "dryer", "appliances", "decor", "wi-fi", "wifi" "knives", "container", "bowl", "washing machine", "heat", "napkin", "plate", "cup", "pillow", "bed", "sheet", " table", "chair", "glasses", "pool", "pans"],
}

# Adjusted function to categorize cleaned reviews, handling NaN values
def categorize_review(row, categories):
    # Handle rows where 'Cleaned Review' is NaN by treating them as empty strings
    cleaned_review = row['Cleaned Review']
    if pd.isnull(cleaned_review):
        cleaned_review = ''  # Treat NaN values as empty strings

    matched_categories = []
    for category, keywords in categories.items():
        if any(keyword in cleaned_review for keyword in keywords):
            matched_categories.append(category)

    # If no categories matched, add "other"
    if not matched_categories:
        matched_categories.append("other")

    return [(row['Segment'], row['Checkout Date'], cleaned_review, cat) for cat in matched_categories]

##################### POLARITY SCORING

sia = SentimentIntensityAnalyzer()

def get_sentiment_scores(text):
    # Ensure text is a string
    text = str(text)
    return sia.polarity_scores(text)

##################### MASTER FUNCTION FOR NLP 

def master(data):

    # cleaning function to remove duplicates
    cleaned_data = cleaning(data)

    # create segments
    segmented_data = segment_reviews(cleaned_data)

    # categorize reviews 
    segmented_data['category'] = segmented_data['cleaned_text'].apply(lambda x: categorize_review(x, categories))  # Assumes categorize_review is defined

    # apply sentiment analysis
    segmented_data['sentiment'] = segmented_data['cleaned_text'].apply(get_sentiment_scores)  # Assumes get_sentiment_scores is defined

    return segmented_data

##################### AGGREGATE METRICS 

def calculate_metrics(group):
    promoters_count = len(group[group['compound'] >= 0.61])
    detractors_count = len(group[group['compound'] <= 0.20])
    all_segments_count = len(group)
    neutrals_count = all_segments_count - promoters_count - detractors_count
    nps_like_metric = (promoters_count - detractors_count) / all_segments_count
    # Ensure NPS-like Metric is at least 0
    nps_like_metric = max(nps_like_metric, 0)
    return pd.Series({
        'Promoters': promoters_count,
        'Detractors': detractors_count,
        'All Segments': all_segments_count,
        'NPS-like Metric': nps_like_metric
    })

# Ensure 'Checkout Date' is a datetime type
data['Checkout Date'] = pd.to_datetime(data['Checkout Date'])

# Extract 'Year-Month' and 'Quarter'
data['Year-Month'] = data['Checkout Date'].dt.to_period('M')

# Define the calculate_metrics function
def calculate_metrics(group):
    promoters_count = len(group[group['compound'] >= 0.61])
    detractors_count = len(group[group['compound'] <= 0.20])
    all_segments_count = len(group)
    neutrals_count = all_segments_count - promoters_count - detractors_count
    nps_like_metric = (promoters_count - detractors_count) / all_segments_count
    nps_like_metric = max(nps_like_metric, 0)  # Ensure NPS-like Metric is at least 0
    return pd.Series({
        'Promoters': promoters_count,
        'Detractors': detractors_count,
        'All Segments': all_segments_count,
        'Neutrals': neutrals_count,
        'NPS-like Metric': nps_like_metric
    })

# Group by Category and Year-Month, then calculate metrics
monthly_metrics = data.groupby(['Category', 'Year-Month']).apply(calculate_metrics).reset_index()
