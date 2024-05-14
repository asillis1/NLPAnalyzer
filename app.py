from flask import Flask, render_template, request, redirect, url_for, flash
import io
import pandas as pd
import functions  # Import your functions module

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the single file from form
        file = request.files['file']
        geography = request.form['geography']
        review_type = request.form['review_type']
        show_me = request.form['show_me']
        selected_state = request.form.get('stateDropdown', None)

        if file and allowed_file(file.filename):
            # Read the file into a Pandas DataFrame
            file_content = io.StringIO(file.stream.read().decode('UTF-8'))
            data = pd.read_csv(file_content)

            # Process the single DataFrame
            data = functions.process_single_csv_file(data)

            # Create the 'state' column
            data = functions.state(data)

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
