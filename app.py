from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import os
import pandas as pd
import numpy as np
from datetime import datetime
from deepface import DeepFace
import shutil

app = Flask(__name__)

# Directory paths
faces_directory = "static/faces"
attendance_directory = "Attendance"
if not os.path.exists(faces_directory):
    os.makedirs(faces_directory)
if not os.path.exists(attendance_directory):
    os.makedirs(attendance_directory)

# Helper function to get today's date
def get_date_today():
    return datetime.now().strftime("%Y-%m-%d")

# Helper function to save attendance
def save_attendance(name, roll):
    filename = f"{attendance_directory}/Attendance-{get_date_today()}.csv"
    if not os.path.isfile(filename):
        df = pd.DataFrame(columns=["Name", "Roll", "Time"])
    else:
        df = pd.read_csv(filename)
    if roll not in df["Roll"].values:
        now = datetime.now().strftime("%H:%M:%S")
        df = pd.concat([df, pd.DataFrame([[name, roll, now]], columns=["Name", "Roll", "Time"])])
        df.to_csv(filename, index=False)

# Helper function to resize and crop faces
def resize_and_crop_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        return cv2.resize(face, (160, 160))
    return None

# Route for the main page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        new_username = request.form.get("newusername")
        new_userid = request.form.get("newuserid")
        if new_username and new_userid:
            user_folder = os.path.join(faces_directory, f"{new_username}_{new_userid}")
            if not os.path.exists(user_folder):
                os.makedirs(user_folder)
            # Start capturing images
            cap = cv2.VideoCapture(0)
            count = 0
            while count < 5:  # Capture 5 images
                ret, frame = cap.read()
                if ret:
                    face = resize_and_crop_face(frame)
                    if face is not None:
                        cv2.imwrite(os.path.join(user_folder, f"{new_username}_{count}.jpg"), face)
                        count += 1
                    cv2.imshow("Capturing Faces", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()
            # Train model on captured images
            DeepFace.find(img_path=os.path.join(user_folder, f"{new_username}_0.jpg"), db_path=faces_directory)
            return redirect(url_for("index"))
    return render_template("index.html")

# Route to start face recognition
@app.route("/startrecognition", methods=["GET"])
def start_recognition():
    cap = cv2.VideoCapture(0)
    recognized = False
    while not recognized:
        ret, frame = cap.read()
        if ret:
            face = resize_and_crop_face(frame)
            if face is not None:
                # Temporarily save the face image to a file
                temp_face_path = "temp_face.jpg"
                cv2.imwrite(temp_face_path, face)
                
                try:
                    # Perform the face recognition
                    results = DeepFace.find(img_path=temp_face_path, db_path=faces_directory, enforce_detection=False)
                    
                    # Check if any match was found in the first DataFrame
                    if len(results) > 0 and not results[0].empty:
                        # Get the identity from the first match
                        identity = results[0].iloc[0]["identity"]
                        
                        # Extract the folder name from the full path
                        folder_name = os.path.basename(os.path.dirname(identity))
                        
                        # Split the folder name to get name and roll
                        name, roll = folder_name.split("_")
                        
                        # Save attendance with the name and roll number
                        save_attendance(name, roll)
                        recognized = True
                        break
                except Exception as e:
                    print(f"Error during face recognition: {e}")
            cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"status": "Face recognized and attendance marked." if recognized else "No face recognized."})




if __name__ == "__main__":
    app.run(debug=True)
