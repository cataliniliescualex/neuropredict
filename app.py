from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import base64
import io
from PIL import Image
import requests

app = Flask(__name__)

AZURE_ENDPOINT = "http://8dfcb175-f4d9-46f8-baa7-ab9a6431dcfc.eastus.azurecontainer.io/score"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    if request.method == 'POST':
        # Get the base64 representation of the drawn image
        canvas_data = request.form.get('canvasData').split(",")[1]
        decoded_img = base64.b64decode(canvas_data)
        
        # Convert it to an image and preprocess for the model
        img = Image.open(io.BytesIO(decoded_img)).convert('L')
        img = img.resize((28, 28))  # Resize image to model's expected input size
        img_array = np.array(img)
        img_array = img_array.reshape(1, 28, 28).tolist()

        # Send the image data to Azure endpoint for inference
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(AZURE_ENDPOINT, json={"data": img_array}, headers=headers)
        if response.status_code == 200:
            prediction = response.json()[0]

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
