# Import the Flask class from the flask module
from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
from io import BytesIO
import face_recognition
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import os
from io import BytesIO

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

def load_image(image_url):
    response = requests.get(image_url)
    response.raise_for_status()
    input_image = Image.open(BytesIO(response.content))
    resized_image = input_image.resize((224, 224))
    return resized_image

# Extract image embeddings using VGG16
def get_image_embeddings(object_image):
    image_array = np.expand_dims(image.img_to_array(object_image), axis=0)
    image_array = preprocess_input(image_array)  # Normalize the image array
    image_embedding = vgg16.predict(image_array)
    return image_embedding

# Calculate similarity score between two images
def get_similarity_score(first_image_url, second_image_url, n_components = 2):
    first_image = load_image(first_image_url)
    second_image = load_image(second_image_url)

    first_image_vector = get_image_embeddings(first_image)
    second_image_vector = get_image_embeddings(second_image)
    
    similarity_score = cosine_similarity(first_image_vector, second_image_vector).reshape(1,)
    return similarity_score

vgg16 = VGG16(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))

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
    print("This is result",result)
    return jsonify(result)
# Run the application
@app.route("/compare_products", methods=['POST'])
def compareProducts():
    data = request.get_json()
    image1_url = data.get('image_1')
    image2_url = data.get('image_2')
    print("this is image1_url",image1_url)
    similarity_score = get_similarity_score(image1_url, image2_url)
    result = {"similarity" : str(similarity_score[0])}
    print("this is result ++?",result)
    return jsonify(result)


if __name__ == '__main__':
    print("Starting the server")
    from waitress import serve
    port = int(os.environ.get("PORT", 10000))
    serve(app, host="0.0.0.0", port= port)