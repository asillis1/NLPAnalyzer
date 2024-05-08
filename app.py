from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import functions  # Import your functions module

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get file list from form
        files = request.files.getlist('file')
        geography = request.form['geography']
        review_type = request.form['review_type']
        show_me = request.form['show_me']
        selected_state = request.form.get('stateDropdown', None)

        # Save uploaded files and process them
        file_names = []
        for file in files:
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join('uploads', filename)
                file.save(filepath)
                file_names.append(filepath)

        # Use functions to process files
        data = functions.process_csv_files('uploads', file_names)

        # Filter by state if necessary (optional)
        if geography == 'state' and selected_state:
            data = functions.filter_by_state(data, selected_state)

        # Process files based on review type (not optional)
        data = functions.process_files(data, review_type)

        # Master NLP function
        data = functions.master_nlp(data)

        # You might need to handle how to display data or pass it to another view
        return render_template('results.html', data=data.to_html())

    states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
              "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
              "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
              "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
              "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
    return render_template('index.html', states=states)

@app.route('/results')
def results():
    # Results are displayed here, possibly handle via session or pass data through URL (not recommended for large data)
    return "Results displayed here."

if __name__ == '__main__':
    app.run(debug=True)
