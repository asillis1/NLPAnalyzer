import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from itertools import chain
import os
nltk.download('vader_lexicon')

# File for defining functions used to run the app

################## VALIDATION FUNCTIONS

#### Need to select a value in each dropdown


##################### DATA CLEANING FUNCTIONS

#### If uploading multiple files (we allow up to 3)


### OLD CODE
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

### NEW CODE
def process_csv_files(directory, file_names):
    """
    Reads multiple CSV files from a specified directory, retains specific columns, and merges them into one DataFrame.

    Args:
    directory (str): The directory where CSV files are located.
    file_names (list of str): List of CSV file names.

    Returns:
    pd.DataFrame: Merged DataFrame containing specified columns.
    """
    # Specify the columns you want to keep
    columns_to_keep = ['checkout_date', 'property_name', 'public_review', 'private_feedback']
    
    dataframes = []
    for file_name in file_names:
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Filter columns if they are available in the DataFrame
            columns = [col for col in columns_to_keep if col in df.columns]
            df = df[columns]
            dataframes.append(df)
        else:
            print(f"File not found: {file_path}")
    
    if dataframes:
        # Concatenate all DataFrames into one, ignoring the index
        data = pd.concat(dataframes, ignore_index=True)
        return data
    else:
        # Returning an empty DataFrame if no valid DataFrames were created
        return pd.DataFrame()

# New function to process a single CSV file
def process_single_csv_file(data):
    """
    Processes a single CSV file (in DataFrame form) by retaining specific columns and ensuring necessary data cleaning.

    Args:
    data (pd.DataFrame): The DataFrame to process.

    Returns:
    pd.DataFrame: Cleaned and processed DataFrame.
    """
    # Specify the columns you want to keep
    columns_to_keep = ['checkout_date', 'property_name', 'public_review', 'private_feedback']
    
    # Filter columns if they are available in the DataFrame
    columns = [col for col in columns_to_keep if col in data.columns]
    data = data[columns]
    
    return data

#### Create State Column (not optional)

def state(data):
    # Extract the last two letters; the result might contain NaN values
    data['state'] = data['property_name'].str.extract(r'(\w\w)$')

    # Convert only non-null extracted values to uppercase
    data['state'] = data['state'].str.upper().fillna('N/A')  # Optionally handle NaNs by setting a default value like 'Unknown'
    return data


#### Filter for singular state (optional)

def filter_by_state(data, state_abbr):
    """
    Filter the DataFrame based on the state abbreviation.
    
    Args:
    data (pd.DataFrame): The DataFrame to filter.
    state_abbr (str): The state abbreviation to filter by.

    Returns:
    pd.DataFrame: A new DataFrame containing only rows where the 'state' column matches the state_abbr.
    """
    # Filter the DataFrame to only include rows with the specified state abbreviation
    filtered_data  = data[data['state'] == state_abbr]
    return filtered_data 

#### Public, Private, All

def process_files(data, review_type):
    """
    Processes the data based on review type. For 'Public' or 'Private', it renames the relevant column to 'review'.
    For 'All', it creates two entries per original entry: one for the public review and one for the private feedback,
    each in a single 'review' column, doubling the number of entries.

    Args:
    data (pd.DataFrame): The DataFrame to process.
    review_type (str): Type of review to process ('Public', 'Private', or 'All').

    Returns:
    pd.DataFrame: Processed DataFrame with the specified reviews.
    """
    essential_columns = ['checkout_date', 'property_name', 'state']  # Essential columns to retain

    if review_type == 'Public':
        data = data[essential_columns + ['public_review']].rename(columns={'public_review': 'review'})
    elif review_type == 'Private':
        data = data[essential_columns + ['private_feedback']].rename(columns={'private_feedback': 'review'})
    else:  # 'All'
        # Duplicate each row for public and private reviews
        public = data[essential_columns + ['public_review']].rename(columns={'public_review': 'review'})
        private = data[essential_columns + ['private_feedback']].rename(columns={'private_feedback': 'review'})
        data = pd.concat([public, private], ignore_index=True)  # Combine into one DataFrame without merging texts

    return data

##################### DATA CLEANING & TOPIC SEGMENTATION (one large function that calls on subfunctions)

# Remove duplicates or NaNs

def cleaning(data):
    data['review'] = data['review'].replace('', pd.NA)
    data = data.dropna(subset=['review'])
    data = data.drop_duplicates(subset=['review', 'checkout_date', 'property_name'], keep='first')
    return data

# Create segments (sentences)

def expand_reviews_by_sentence(data):
    """
    Expands each review in the DataFrame into multiple entries based on sentences.
    Each sentence becomes a new row with all original column values preserved.

    Args:
    data (pd.DataFrame): The DataFrame containing the reviews.

    Returns:
    pd.DataFrame: Expanded DataFrame with each sentence from the reviews as a new row.
    """
    # Assuming the DataFrame has a 'review' column, split reviews into lists of sentences
    # Here, '.\s+' is used as a simple pattern to split at periods followed by whitespace, assuming this marks the end of sentences
    data['segment'] = data['review'].str.split(r'\.\s+')
    
    # Explode the 'segment' lists into separate rows
    data = data.explode('segment')
    
    # Optional: Remove any trailing periods that might have been left after splitting
    data['segment'] = data['segment'].str.rstrip('.')
    
    # Remove rows where 'segment' is NaN
    data = data.dropna(subset=['segment'])

    return data

# Create cleaned_segment

# Load the English language model from spaCy
nlp = spacy.load("en_core_web_sm")

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Custom stopwords setup, removing typical negations that are significant in sentiment analysis
custom_stopwords = set(stopwords.words('english')) - {"no", "not"}

# Contraction mapping
contraction_mapping = {
        "didn't": "did not",
        "don't": "do not",
        "aren't": "are not",
        "couldn't": "could not",
        # Add more contractions as necessary
    }

def preprocess_segments(data):
    """
    Cleans, tokenizes, and lemmatizes the text in the 'segment' column of the DataFrame.
    Handles custom stopwords and contractions. Adds a new column 'cleaned_segment'.

    Args:
    data (pd.DataFrame): The DataFrame containing the text in the 'segment' column.

    Returns:
    pd.DataFrame: The original DataFrame with an additional column 'cleaned_segment'.
    """
    def clean_text(text):
        # Handle cases where text might be NaN
        if pd.isna(text):
            return ""

        # Expand contractions
        for contraction, expanded in contraction_mapping.items():
            text = re.sub(r'\b' + contraction + r'\b', expanded, text)

        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)

        # Lowercase
        text = text.lower()

        # Tokenization and removal of custom stopwords
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in custom_stopwords]

        # Lemmatization using spaCy (since we're loading it, might as well utilize its powerful lemmatizer)
        doc = nlp(' '.join(filtered_tokens))
        lemmatized = [token.lemma_ for token in doc]

        return ' '.join(lemmatized)

    # Apply the cleaning function to each segment
    data['cleaned_segment'] = data['segment'].apply(lambda x: clean_text(x))
    
    return data

# Categorizes segment (vertical)

def categorize_segments(data, categories):
    """
    Categorizes each segment in the DataFrame based on predefined categories and duplicates rows if a segment
    fits multiple categories.

    Args:
    data (pd.DataFrame): The DataFrame containing the cleaned segments and other associated details.
    categories (dict): A dictionary of categories with their associated keywords.

    Returns:
    pd.DataFrame: The updated DataFrame with a new 'category' column.
    """
    # Function to find matching categories for each cleaned_segment
    def match_categories(cleaned_segment):
        matched_categories = []
        for category, keywords in categories.items():
            if any(keyword in cleaned_segment for keyword in keywords):
                matched_categories.append(category)
        return matched_categories if matched_categories else ['other']

    # Expand the DataFrame by matching categories
    expanded_rows = []
    for _, row in data.iterrows():
        cleaned_segment = row['cleaned_segment'] if pd.notna(row['cleaned_segment']) else ''
        matched_cats = match_categories(cleaned_segment)
        for category in matched_cats:
            new_row = row.copy()
            new_row['category'] = category
            expanded_rows.append(new_row)

    # Create a new DataFrame from the expanded rows
    return pd.DataFrame(expanded_rows)

# Define categories and their associated keywords
categories = {
    "communication": ["communicat", "contact", "responsive", "response", "reply", "communication", "host", "hospitality", "professional", "responsive", "responded", "request", "getting back", "respond", "available", "kind", "customer service", "accommodating", "helpful"],
    "location": ["accessible", "location", "area", "place", "stay" , "accommodation", "spacious", "setting", "walk", "walkability", "parking", "driving", "drive", "biking", "located", "close", "near", "far", "cozy", "local", "neighbourhood", "neighborhood", "across the street", "view", "next to", "market", "cafe"],
    "cleanliness": ["well kept", "clean", "tidy", "hygiene", "dirty", "cleanliness", "gross", "housekeeping", "stain", "spotless", "beach", "crumb", "dirt", "dust", "bug", "wash", "ant", "maintainence", "infest", "spotless", "sanitized", "sanitary"],
    "accuracy": ["accurate", "exact", "description", "expectation", "picture", "catfish", "described", "space", "money", "expensive", "room", "commensurate", "price", "temperature", "comfortable", "uncomfortable", "photo", "mislead", "luxury", "decorated", "decoration", "advertised", "outside", "interiors", "size", "backyard", "yard", "photos", "incorrect", "inform", "wrong", "instructions"],
    "check-in": ["check-in", "arrival", "welcome", "attention", "checkin", "check in", "checked", "entry", "keypad", "code", "key"],
    "amenities": ["towels", "blinds", "coffee machine", "kitchenware", "oven", "utensil", "accommodations", "alarms", "tv", "t.v", "tools", "coffee", "tea", "toilet paper", "laundry", "linen", "toiletries", "facilities", "stocked", "equipped", "heater", "refrigerator", "couch", "fridge", "amenities", "water", "thermostat", "broken", "dish soap", "dishwasher", "dishes", "furnishing", "furnished", "interiors", "ac", "a/c", "a.c.", "air conditioning", "ice maker", "gear", "garbage", "trash", "internet", "blanket", "towel", "shower", "washer", "dryer", "appliances", "decor", "wi-fi", "wifi", "knives", "container", "bowl", "washing machine", "heat", "napkin", "plate", "cup", "pillow", "bed", "sheet", "table", "chair", "glasses", "pool", "pans"]
}

# sentiment score for categorized, cleaned segment
def add_sentiment_scores(data):
    """
    Adds sentiment scores to the DataFrame based on the 'cleaned_segment' column.

    Args:
    data (pd.DataFrame): The DataFrame containing the cleaned segments.

    Returns:
    pd.DataFrame: The original DataFrame with an additional column 'compound_score'.
    """
    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Filter out rows where 'cleaned_segment' is NaN or empty
    data = data[data['cleaned_segment'].notna() & data['cleaned_segment'].str.strip().ne('')]

    # Define a function to get the compound sentiment score for a text
    def get_compound_score(text):
        return sia.polarity_scores(text)['compound']

    # Apply the function to each cleaned_segment to get the compound scores
    data['compound_score'] = data['cleaned_segment'].apply(get_compound_score)
    
    return data

#### MASTER FUNCTION

def master_nlp(data):
    # cleaning (1+2)
    data = cleaning(data)
    # creating segments (3)
    data = expand_reviews_by_sentence(data)
    # process segments (4)
    data = preprocess_segments(data)
    # categorize segments (5) 
    data = categorize_segments(data, categories)
    # add sentiment scores (6 and fin)
    data = add_sentiment_scores(data)
    return data

############################# RESULTS CODE #####################################

def plot_sentiment_trend_by_category(data):
    data['checkout_date'] = pd.to_datetime(data['checkout_date'])
    data['Year-Month'] = data['checkout_date'].dt.to_period('M')

    category_names = ['amenities', 'location', 'check-in', 'cleanliness', 'accuracy', 'communication']
    plots = {}

    for category in category_names:
        category_data = data[data['category'] == category]

        if category_data['Year-Month'].dtype == object:
            category_data['Year-Month'] = category_data['Year-Month'].astype('period[M]')

        year_month_labels = category_data['Year-Month'].astype(str).tolist()

        category_data['classification'] = pd.cut(category_data['compound_score'], bins=[-float('inf'), -0.05, 0.05, float('inf')], labels=['Negative', 'Neutral', 'Positive'])
        sentiment_counts = category_data.groupby(['Year-Month', 'classification']).size().unstack(fill_value=0)

        all_months = pd.period_range(start=sentiment_counts.index.min(), end=sentiment_counts.index.max(), freq='M')
        sentiment_counts = sentiment_counts.reindex(all_months, fill_value=0)

        fig = go.Figure()

        for sentiment in ['Positive', 'Neutral', 'Negative']:
            fig.add_trace(go.Bar(
                x=sentiment_counts.index.astype(str),
                y=sentiment_counts[sentiment],
                name=sentiment,
                marker_color={'Positive': 'green', 'Neutral': 'grey', 'Negative': 'red'}[sentiment]
            ))

        fig.update_layout(
            barmode='stack',
            title=f'Sentiment Trend for {category.capitalize()} Over Time',
            xaxis=dict(title='Year-Month', type='category', tickangle=-45),
            yaxis=dict(title='Count of Segments'),
            autosize=True,
            width=1000,
            height=600,
            legend_title_text='Sentiment'
        )

        plots[category] = pio.to_html(fig, full_html=False)

    return plots

def calculate_metrics(group):
    group['checkout_date'] = pd.to_datetime(data['checkout_date'])
    group['Year-Month'] = data['checkout_date'].dt.to_period('M')
    group['checkout_date'] = data['checkout_date'].dt.strftime('%Y-%m-%d')
    
    # Group by Category and Year-Month, then calculate metrics
    monthly_metrics = data.groupby(['category', 'Year-Month']).apply(lambda group:pd.Series({
      'Promoters': len(group[group['compound_score'] >= 0.61]),
      'Detractors': len(group[group['compound_score'] <= 0.20]),
      'All Segments': len(group),
      'Neutrals': len(group) - len(group[group['compound_score'] >= 0.61]) - len(group[group['compound_score'] <= 0.20]),
      'NPS-like Metric': (len(group[group['compound_score'] >= 0.61]) - len(group[group['compound_score'] <= 0.20])) / len(group)
      })).reset_index()

    return monthly_metrics


def plot_trend(data, category_name):
    category_names = ['amenities', 'location', 'check-in', 'cleanliness', 'accuracy', 'communication']
    plots = {}
    for category in category_names:
        data = data[data['category'] == category]
        # Ensure 'Year-Month' is in the proper datetime format
        if data['Year-Month'].dtype == object:
            data['Year-Month'] = pd.to_datetime(data['Year-Month'], format='%Y-%m')

        # Format year-month labels for the x-axis
        year_month_labels = data['Year-Month'].dt.strftime('%Y-%m').tolist()
        nps_values = data['NPS-like Metric']

        # Create the figure
        fig = go.Figure()

        # Add bar chart
        fig.add_trace(go.Bar(x=year_month_labels, y=nps_values, name='NPS Metric'))

        # Add trend line
        fig.add_trace(go.Scatter(x=year_month_labels, y=nps_values, mode='lines+markers', name='Trend Line', line=dict(color='red')))

        fig.update_layout(
            title='{} NPS Metric Trend'.format(category_name.capitalize()),
            xaxis_title='Year-Month',
            yaxis_title='NPS-like Metric',
            xaxis=dict(
                type='category',
                tickangle=-45),
            autosize=False,
            width=1800,
            height=600,
            margin=dict(l=50,r=50,b=100,t=100,pad=4))
        
        plots[category] = pio.to_html(fig, full_html=False)

    return plots

# This is how it should be used: 
# test = master_nlp(data)
# print(test)
