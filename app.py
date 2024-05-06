from flask import Flask, request, redirect, url_for, render_template, session
from functions import merge_csv_files, process_files, clean_text, categorize_review, get_sentiment_scores, calculate_metrics
import pandas as pd
import time
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

data = pd.DataFrame()  # Initialize global variable data

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return redirect(url_for('loading'))

    return render_template('home.html')

@app.route('/', methods=['POST'])
def handle_form():
    files = request.files.getlist('file1')  # Collect all uploaded files
    if files:
        global data  # Use the global data variable
        # Merge CSV files into one DataFrame
        data = merge_csv_files(files)
        # Process the DataFrame based on the review type
        review_type = request.form.get('review_type')
        data = process_files(data, review_type)
        # Depending on your next steps, here you could proceed with further processing or output
        return "Files processed successfully."
    else:
        return "No files uploaded", 400  # Handle the case where no files were uploaded

@app.route('/loading')
def loading():
    global data  # Use the global data variable
    # Process the DataFrame based on the review type
    data = process_files(data, session.get('review_type'))  # Use session.get() to avoid KeyError

    # Apply the cleaning function
    data['Cleaned Review'] = data['review'].apply(clean_text)

    # Apply categorization and create a new DataFrame for categorized segments
    categorized_data = [categorize_review(row) for index, row in data.iterrows()]
    categorized_data = list(chain.from_iterable(categorized_data))  # Flatten the list of lists

    # Create a new DataFrame from the categorized data
    categorized_df = pd.DataFrame(categorized_data, columns=['Segment', 'Checkout Date', 'Cleaned Review', 'Category'])

    # Polarity Scoring for each segment
    data[['neg', 'neu', 'pos', 'compound']] = categorized_df['Cleaned Review'].apply(lambda review: pd.Series(get_sentiment_scores(review)))

    # Aggregate metrics 
    segment_metrics = data.groupby('Category').apply(calculate_metrics)
    
    # Store results for access in the results route
    session['processed_data'] = categorized_df.to_json()

    # Redirect to results page
    return redirect(url_for('results'))

@app.route('/results')
def results():
    # Logic to generate graphs goes here
    # For now, let's just render the results template
    return render_template('results.html')

if __name__ == "__main__":
    app.run(debug=True)
