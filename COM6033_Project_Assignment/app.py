import os
import numpy as np
from flask import Flask, request, render_template # Flask for web app
from tensorflow.keras.models import load_model #Loading the pre-trained model
from PIL import Image #Image handling
from io import BytesIO #Encoding images to base64 to display in HTML
import base64

app = Flask(__name__)

# Load the model
model = load_model("COM6033_Project_Assignment/model/birds.h5")

# Define class names
class_names = ['GoldFinch', 'Magpie', 'Robin', 'Sparrow', 'Swan']

# Allowed image extensions
allowed_extensions = {"png", "jpg", "jpeg"}

# Function to check if file has an allowed extension
def check_extension(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

# Display the html file
@app.route("/")
def index():
    return render_template("index.html")

# Prediction handling
@app.route("/predict", methods=["POST"])
def predict():
    # Checks if a file was uploaded
    if "file" not in request.files:
        return render_template("index.html", error="No file part")
    
    file = request.files["file"] # Get uploaded file

    # Checks if a file was selected
    if file.filename == "":
        return render_template("index.html", error="No selected file")
    
    # Validate file extension
    if file and check_extension(file.filename):
        # Open the image
        img = Image.open(file)
        img = img.resize((128, 128))  # Resized to match input shape of the model
        img_array = np.array(img) / 255.0  # Normalize the image
        
        # Expand dimensions to simulate a batch of size 1
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the model
        predictions = model.predict(img_array)

        # Get all the predictions and their respective class names
        predicted_indices = np.argsort(predictions[0])[::-1]  # Sort the predictions based on confidence
        predicted_confidences = predictions[0][predicted_indices] * 100  # Convert to percentage

        # Main prediction
        top_prediction =(class_names[predicted_indices[0]], predicted_confidences[0])

        # Top 3 predictions
        top_n = 3
        top_predictions = [(class_names[idx], predicted_confidences[i]) for i, idx in enumerate(predicted_indices[:top_n])]

        # Convert the image to base64 for display
        img_buffer = BytesIO()
        img.save(img_buffer, format="PNG") #Convert to png file
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

        # Send the results and image to the template
        return render_template("index.html", 
                            top_prediction=top_prediction, #Main prediction
                            top_predictions=top_predictions, #Other predictions
                            img_base64=img_base64)
    
    else:
        # Display an error if file format is incorrect
        return render_template("index.html", error="Invalid file format")

if __name__ == '__main__':
    app.run(debug=True)