
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import re
import base64
import io

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = load_model("mnist_model.h5")

def preprocess_canvas_data(data):
    # Process the canvas data URL to obtain the image data
    data = re.sub('^data:image/.+;base64,', '', data)
    data = base64.b64decode(data)
    data = io.BytesIO(data)
    im = Image.open(data).convert("L")
    im = im.resize((28, 28))
    im = np.array(im)
    im = im.reshape(1, 28, 28, 1)
    im = im/255.0
    return im

@app.route('/', methods=['GET', 'POST'])
def predict_digit():
    if request.method == 'POST':
        # Get canvas data from the POST request
        canvas_data = request.form['canvas_data']
        
        # Preprocess the canvas data
        data = preprocess_canvas_data(canvas_data)
        
        # Make prediction
        prediction = model.predict(data)
        predicted_class = np.argmax(prediction)
        
        # Return the prediction
        return jsonify({'predicted_class': int(predicted_class)})
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
