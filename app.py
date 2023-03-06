import numpy as np
import cv2
import tensorflow_hub as hub
import tensorflow as tf
# from PIL import Image
from flask import Flask, request
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

detector = hub.load(module_handle).signatures['default']

app = Flask(__name__)

# Define the endpoint for receiving image data
@app.route('/image', methods=['POST'])
def upload_image():
    # Decode the image data from the request body
    image_bytes = request.get_data()
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Save the image to a file
    # file_name = 'image.jpg'
    # cv2.imwrite(file_name, image)

    return 'Image saved successfully.'

if __name__ == '__main__':
    app.run()
