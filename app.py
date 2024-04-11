# Import the Flask class from the flask module
from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
from io import BytesIO
import face_recognition

def load_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            image = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return image
        else:
            print(f"Failed to download image from URL: {image_url}")
            return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def find_face_encodings(image_url):
    image = load_image_from_url(image_url)
    if image is not None:
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            return face_encodings[0]
        else:
            print("No face found in the image.")
            return None
    else:
        print("Failed to load image.")
        return None

def compare_images(image1_url, image2_url):
    image1_encoding = find_face_encodings(image1_url)
    image2_encoding = find_face_encodings(image2_url)
    
    if image1_encoding is not None and image2_encoding is not None:
        is_same = face_recognition.compare_faces([image1_encoding], image2_encoding)[0]
        print(f"Are the faces the same? {is_same}")
        
        if is_same:
            distance = face_recognition.face_distance([image1_encoding], image2_encoding)
            accuracy = (1 - distance[0]) * 100
            print(f"Accuracy level: {accuracy:.2f}%")
            return {"isSame" : True, "accuracy" : accuracy}
    else:
        print("Unable to compare images.")
        return {"isSame" : False, "accuracy" : 0}


# Create an instance of the Flask class
app = Flask(__name__)



# Define a route and a function to handle requests to that route
@app.route('/')
def hello_world():
    print("this is a test")
    return 'Hello, Flask World!'

@app.route('/compare_images', methods=['POST'])
def compareImages():
    data = request.get_json()
    image1_url = data.get('image1_url')
    image2_url = data.get('image2_url')
    result = compare_images(image1_url, image2_url)
    print(result)
    return jsonify(result)
# Run the application
if __name__ == '__main__':
    print("Starting the server")
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)