import cv2
import numpy as np
import keras
from keras.models import load_model
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer
from mltu.annotations.images import CVImage
from configs import ModelConfigs

keras.config.enable_unsafe_deserialization()

# Load the trained Keras model
model_path = "/Users/aaronshields/Desktop/CSCI-360/Research_project/Models/04_sentence_recognition/Checkpoints/model.keras"
model = load_model(model_path)

configs = ModelConfigs()

# Define the file path for the image
image_path = "/Users/aaronshields/Desktop/CSCI-360/Research_project/readableFiles/usc1.jpeg"

# Read the image using OpenCV
image = cv2.imread(image_path)

# Preprocess the image
# You may need to resize and normalize the image according to the requirements of your model
# Here, I'm assuming you have defined preprocessing functions like ImageReader and ImageResizer
image = ImageReader(CVImage).preprocess(image)
image = ImageResizer.resize_maintaining_aspect_ratio(image, width=configs.width, height=configs.height)

# Perform inference to predict the text
# Make sure to preprocess the image data appropriately based on the model's input requirements
predicted_text = model.predict(np.expand_dims(image, axis=0))

# Postprocess the predicted text (if necessary)
# This might involve decoding the output into human-readable text
# You may also need to handle any special postprocessing steps specific to your model or application

print("Predicted Text:", predicted_text)
