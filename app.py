from flask import Flask, request, render_template, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
import librosa
import numpy as np
import joblib
import os
import smtplib
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"

# Load the pre-trained model and LabelEncoder
model = load_model("my_model.h5")
labelencoder = joblib.load("label_encoder.pkl")

# Ensure the upload folder exists
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# Firebase initialization
cred = credentials.Certificate(
    "C:\\Users\\prabh\\Downloads\\audio-classification-c3e30-firebase-adminsdk-w8397-9e91252ac0.json"
)  # Replace with your Firebase admin JSON file path
firebase_admin.initialize_app(
    cred,
    {"databaseURL": "https://audio-classification-c3e30-default-rtdb.firebaseio.com/"},
)  # Replace with your Realtime Database URL

# Email configuration
EMAIL_ADDRESS = "prabhat.bhasme21@vit.edu"  # Replace with your email
EMAIL_PASSWORD = "welcome@vit69"  # Replace with your email's app password
RECIPIENTS = [
    "prabhatbhasme@gmail.com",
    "sachin.bhargav21@vit.edu",
]


def send_email(predicted_class):
    SUBJECT = f"Alert: {predicted_class} Detected"
    BODY = f"This is an automated alert for the detected class: {predicted_class}. Please take necessary actions."
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        for recipient in RECIPIENTS:
            server.sendmail(EMAIL_ADDRESS, recipient, f"Subject: {SUBJECT}\n\n{BODY}")
    print("Email sent successfully!")


# Function to extract features from an audio file
def features_extractor(file):
    audio, sample_rate = librosa.load(file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=50)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/project")
def project():
    return render_template("project.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(url_for("project"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("project"))

    # Save the uploaded file locally
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Extract features and make prediction
    mfccs_scaled_features = features_extractor(file_path).reshape(1, -1)
    predicted_label = np.argmax(model.predict(mfccs_scaled_features), axis=1)
    prediction_class = labelencoder.inverse_transform(predicted_label)[0]

    # Clean up the uploaded file locally
    os.remove(file_path)

    # Store metadata in Firebase Realtime Database
    timestamp = datetime.now().isoformat()  # Current time in ISO format
    db.reference("audio_records").push(
        {
            "audio_filename": file.filename,
            "classified_audio": prediction_class,
            "timestamp": timestamp,
        }
    )
    print("Data stored in Firebase Realtime Database successfully!")

    # Send email notification
    send_email(predicted_class=prediction_class)

    # Display the prediction result
    return render_template("project.html", prediction=prediction_class)


if __name__ == "__main__":
    app.run(debug=True)
