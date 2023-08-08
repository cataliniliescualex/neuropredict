from flask import request, render_template
import requests
import json
import numpy as np

@app.route('/', methods=['POST'])

def preprocess_canvas_data(canvas_data):
    # Assuming canvas_data is a list of lists (28x28 pixels)
    # Flatten and reshape the data
    flattened_data = np.array(canvas_data).flatten().reshape(1, 784)
    # Normalize to [0, 1] range
    normalized_data = flattened_data / 255.0
    return normalized_data.tolist()

def predict_digit():
    # Get canvas data from form
    canvas_data = request.form['canvasData']

    # Preprocess canvas data into appropriate format
    # This depends on how you implemented the drawing on the canvas
    # But it should result in a numpy array of shape (1, 784)
    data = preprocess_canvas_data(canvas_data)

    # Convert to JSON string
    input_data = json.dumps({'data': data.tolist()})

    # Set the content type
    headers = {'Content-Type': 'application/json'}

    # Make the request and display the response
    resp = requests.post(scoring_uri, input_data, headers=headers)
    return render_template('index.html', prediction=resp.text)

@app.route('/')
def home():
    return render_template('index.html')

