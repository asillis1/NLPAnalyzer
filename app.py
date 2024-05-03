from flask import Flask, render_template, request, redirect, url_for
from functions import clean_text, contraction_mapping, custom_stopwords


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        geography = request.form.get('geography')
        if geography == 'By State':
            return redirect(url_for('select_state'))
        elif geography == 'All':
            return redirect(url_for('loading'))

    return render_template('home.html')

@app.route('/select_state', methods=['GET', 'POST'])
def select_state():
    states = [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
        "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
        "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
        "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
        "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
        "New Hampshire", "New Jersey", "New Mexico", "New York",
        "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
        "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
        "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
        "West Virginia", "Wisconsin", "Wyoming"
    ]

    if request.method == 'POST':
        selected_state = request.form.get('state')
        # Perform form validation here
        if not selected_state:
            return render_template('select_state.html', states=states, error='Please select a state.')

        # Redirect to loading page
        return redirect(url_for('loading'))

    return render_template('select_state.html', states=states, error=None)

@app.route('/loading')
def loading():
    # Simulate computation delay (for demonstration purposes)
    import time
    time.sleep(3)  # Delay for 3 seconds

    # Redirect to results page
    return redirect(url_for('results'))

@app.route('/results')
def results():
    # Logic to generate graphs goes here
    # For now, let's just render the results template
    return render_template('results.html')

if __name__ == "__main__":
    app.run(debug=True)
