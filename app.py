from flask import Flask, request, render_template
import requests
import json
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['POST'])
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

if __name__ == '__main__':
    app.run(debug=True)
